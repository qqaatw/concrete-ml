"""Implements the conversion of a tree model to a numpy function."""
import math
import warnings
from typing import Callable, List, Tuple

import numpy
import onnx
from onnx import numpy_helper

from ..common.debugging.custom_assert import assert_true
from ..common.utils import (
    MAX_BITWIDTH_BACKWARD_COMPATIBLE,
    get_onnx_opset_version,
    is_regressor_or_partial_regressor,
)
from ..onnx.convert import OPSET_VERSION_FOR_ONNX_EXPORT, get_equivalent_numpy_forward_from_onnx
from ..onnx.onnx_model_manipulations import clean_graph_at_node_op_type, remove_node_types
from ..quantization import QuantizedArray
from ..quantization.quantizers import UniformQuantizer

# pylint: disable=wrong-import-position,wrong-import-order

# Silence Hummingbird warnings
warnings.filterwarnings("ignore")
from typing import List

import onnx
from hummingbird.ml import convert as hb_convert  # noqa: E402

# pylint: enable=wrong-import-position,wrong-import-order

# pylint: disable=too-many-branches


def get_onnx_model(model: Callable, x: numpy.ndarray, framework: str) -> onnx.ModelProto:
    """Create ONNX model with Hummingbird convert method.

    Args:
        model (Callable): The tree model to convert.
        x (numpy.ndarray): Dataset used to trace the tree inference and convert the model to ONNX.
        framework (str): The framework from which the ONNX model is generated.
            (options: 'xgboost', 'sklearn')

    Returns:
        onnx.ModelProto: The ONNX model.
    """

    # Silence Hummingbird warnings
    warnings.filterwarnings("ignore")

    extra_config = {
        "tree_implementation": "gemm",
        "onnx_target_opset": OPSET_VERSION_FOR_ONNX_EXPORT,
    }
    if framework != "sklearn":
        extra_config["n_features"] = x.shape[1]

    onnx_model = hb_convert(
        model,
        backend="onnx",
        test_input=x,
        extra_config=extra_config,
    ).model
    return onnx_model


def workaround_squeeze_node_xgboost(onnx_model: onnx.ModelProto):
    """Workaround to fix torch issue that does not export the proper axis in the ONNX squeeze node.

    FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2778
    The squeeze ops does not have the proper dimensions.
    remove the following workaround when the issue is fixed
    Add the axis attribute to the Squeeze node

    Args:
        onnx_model (onnx.ModelProto): The ONNX model.
    """
    target_node_id_list = [
        i for i, node in enumerate(onnx_model.graph.node) if node.op_type == "Squeeze"
    ]
    assert_true(
        len(target_node_id_list) == 1,
        "Multiple Squeeze node found which is unexpected in tree-based models",
    )
    axes_input_name = "axes_squeeze"
    axes_input = onnx.helper.make_tensor(axes_input_name, onnx.TensorProto.INT64, [1], (1,))

    onnx_model.graph.initializer.append(axes_input)
    onnx_model.graph.node[target_node_id_list[0]].input.insert(1, axes_input_name)


def assert_add_node_and_constant_in_xgboost_regressor_graph(onnx_model: onnx.ModelProto):
    """Assert if an Add node with a specific constant exists in the ONNX graph.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model.
    """

    constant_add_name = "_operators.0.base_prediction"
    is_expected_add_node_present = False
    initializer_value_correct = False

    # Find the initializer with the specified name
    initializer = next(
        (init for init in onnx_model.graph.initializer if init.name == constant_add_name), None
    )

    # Check if the initializer exists and its value is 0.5
    if initializer:
        values = onnx.numpy_helper.to_array(initializer)
        if values.size == 1 and values[0] == 0.5:
            initializer_value_correct = True

    # Iterate over all nodes in the model's graph
    for node in onnx_model.graph.node:
        # Check if the node is an "Add" node and has the
        # specified initializer as one of its inputs
        if node.op_type == "Add" and constant_add_name in node.input:
            is_expected_add_node_present = True
            break

    assert_true(
        is_expected_add_node_present and initializer_value_correct,
        "XGBoostRegressor is not supported.",
    )


def add_transpose_after_last_node(onnx_model: onnx.ModelProto):
    """Add transpose after last node.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model.
    """
    # Get the output node
    output_node = onnx_model.graph.output[0]

    # Create the node with perm attribute equal to (2, 1, 0)
    transpose_node = onnx.helper.make_node(
        "Transpose",
        inputs=[output_node.name],
        outputs=["transposed_output"],
        perm=[2, 1, 0],
    )

    onnx_model.graph.node.append(transpose_node)
    onnx_model.graph.output[0].name = "transposed_output"


def preprocess_tree_predictions(
    init_tensor: numpy.ndarray,
    output_n_bits: int,
) -> QuantizedArray:
    """Apply post-processing from the graph.

    Args:
        init_tensor (numpy.ndarray): Model parameters to be pre-processed.
        output_n_bits (int): The number of bits of the output.

    Returns:
        QuantizedArray: Quantizer for the tree predictions.
    """

    # Quantize probabilities and store QuantizedArray
    # IMPORTANT: we must use symmetric signed quantization such that
    # 0 in clear == 0 in quantized.

    quant_args = {}

    # If we have negative values, use a symmetric quantization
    # in order to have a zero zero-point
    if numpy.min(init_tensor) < 0:
        is_signed = is_symmetric = True

    # To ensure the zero-point is 0 we force the
    # range of the quantizer to [0..max(init_tensor)]
    else:
        is_signed = is_symmetric = False
        quant_args["rmax"] = numpy.max(init_tensor)
        quant_args["rmin"] = 0
        quant_args["uvalues"] = []

    q_y = QuantizedArray(
        n_bits=output_n_bits,
        values=init_tensor,
        is_signed=is_signed,
        is_symmetric=is_symmetric,
        **quant_args,
    )
    # Make sure the zero_point is 0 to prevent errors in Hummingbird's GEMM approach.
    # Asymmetric quantization may cause the zero_point to be non-zero
    # which leads to incorrect results.
    assert_true(
        q_y.quantizer.zero_point == 0,
        "Zero point is not 0. Symmetric signed quantization must work.",
    )
    return q_y


def tree_onnx_graph_preprocessing(
    onnx_model: onnx.ModelProto, framework: str, expected_number_of_outputs: int
):
    """Apply pre-processing onto the ONNX graph.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model.
        framework (str): The framework from which the ONNX model is generated.
            (options: 'xgboost', 'sklearn')
        expected_number_of_outputs (int): The expected number of outputs in the ONNX model.
    """
    # Make sure the ONNX version returned by Hummingbird is OPSET_VERSION_FOR_ONNX_EXPORT
    onnx_version = get_onnx_opset_version(onnx_model)
    assert_true(
        onnx_version == OPSET_VERSION_FOR_ONNX_EXPORT,
        f"The ONNX version returned by Hummingbird is {onnx_version} "
        f"instead of {OPSET_VERSION_FOR_ONNX_EXPORT}",
    )

    # Check we do have the correct number of ONNX output.
    # Hummingbird returns two outputs for classification (predict and predict_proba)
    # while a single output for regression (predict)
    assert_true(
        len(onnx_model.graph.output) == expected_number_of_outputs,
        on_error_msg=f"{len(onnx_model.graph.output)} != 2",
    )

    # Check that a XGBoostRegressor onnx graph has the + 0.5 add node.
    if framework == "xgboost":
        # Make sure it is a regression model
        # (by checking it has a single output, as mentioned above)
        if len(onnx_model.graph.output) == 1:
            assert_add_node_and_constant_in_xgboost_regressor_graph(onnx_model)

    # Cut the graph at the ReduceSum node as large sum are not yet supported.
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/451
    clean_graph_at_node_op_type(onnx_model, "ReduceSum")

    if framework == "xgboost":
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2778
        # The squeeze ops does not have the proper dimensions.
        # remove the following workaround when the issue is fixed
        # Add the axis attribute to the Squeeze node
        workaround_squeeze_node_xgboost(onnx_model)
    else:
        # Add a transpose node after the last node.
        # sklearn models apply the reduce sum before the transpose.
        # To have equivalent output between xgboost in sklearn,
        # apply the transpose before returning the output.
        add_transpose_after_last_node(onnx_model)

    # Cast nodes are not necessary so remove them.
    remove_node_types(onnx_model, op_types_to_remove=["Cast"])


def replace_operator_with_rounded_version(onnx_model, lsbs_to_remove):
    """Replace comparisons with rounded comparisons.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model.
        lsbs_to_remove (List[int]): A list of two integers specifying the number of LSBs to remove.

    Returns:
        onnx.ModelProto: The modified ONNX model.
    """

    assert_true(isinstance(lsbs_to_remove, list), "lsbs_to_remove must be a list.")
    assert_true(len(lsbs_to_remove) == 2, "lsbs_to_remove must have exactly two values.")

    # Mapping of original operators to their rounded counterparts
    operator_mapping = {
        "Greater": "RoundedGreater",
        "Less": "RoundedLess",
        "GreaterOrEqual": "RoundedGreaterOrEqual",
        "LessOrEqual": "RoundedLessOrEqual",
        "Equal": "RoundedEqual",
    }

    # Track if the required operators have been replaced
    comparison_replaced = False
    equal_replaced = False

    new_nodes = []

    for node in onnx_model.graph.node:
        if not comparison_replaced and node.op_type in operator_mapping and node.op_type != "Equal":
            # Use the first value in lsbs_to_remove for the comparison operator
            lsbs = lsbs_to_remove[0]
            comparison_replaced = True
        elif not equal_replaced and node.op_type == "Equal":
            # Use the second value in lsbs_to_remove for the Equal operator
            lsbs = lsbs_to_remove[1]
            equal_replaced = True
        else:
            new_nodes.append(node)
            continue

        # Create a new node with the corresponding rounded operator
        rounded_node = onnx.helper.make_node(
            operator_mapping[node.op_type],
            inputs=node.input,
            outputs=node.output,
            lsbs_to_remove=lsbs,
        )
        new_nodes.append(rounded_node)

    # Ensure that both a comparison and an equal operator were replaced
    assert_true(
        comparison_replaced and equal_replaced, "Required operators not found in the model."
    )

    # Replace the graph's node list with the new list
    onnx_model.graph.ClearField("node")
    onnx_model.graph.node.extend(new_nodes)

    return onnx_model


def tree_values_preprocessing(
    onnx_model: onnx.ModelProto,
    framework: str,
    output_n_bits: int,
) -> QuantizedArray:
    """Pre-process tree values.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model.
        framework (str): The framework from which the ONNX model is generated.
            (options: 'xgboost', 'sklearn')
        output_n_bits (int): The number of bits of the output.

    Returns:
        QuantizedArray: Quantizer for the tree predictions.
    """

    # Modify ONNX graph to fit in FHE
    for i, initializer in enumerate(onnx_model.graph.initializer):
        # All constants in our tree should be integers.
        # Tree thresholds can be rounded up or down (depending on the tree implementation)
        # while the final probabilities/regression values must be quantized.
        # We extract the value stored in each initializer node into the init_tensor.
        init_tensor = numpy_helper.to_array(initializer)
        if "weight_3" in initializer.name:
            # weight_3 is the prediction tensor, apply the required pre-processing
            q_y = preprocess_tree_predictions(init_tensor, output_n_bits)

            # Get the preprocessed tree predictions to replace the current (non-quantized)
            # values in the onnx_model.
            init_tensor = q_y.qvalues
        elif "bias_1" in initializer.name:
            if framework == "xgboost":
                # xgboost uses "<" (Less) operator thus we must round up.
                init_tensor = numpy.ceil(init_tensor)
            elif framework == "sklearn":
                # sklearn trees use "<=" (LessOrEqual) operator thus we must round down.
                init_tensor = numpy.floor(init_tensor)
        new_initializer = numpy_helper.from_array(init_tensor.astype(numpy.int64), initializer.name)
        onnx_model.graph.initializer[i].CopyFrom(new_initializer)
    return q_y


# pylint: disable=too-many-locals
def tree_to_numpy(
    model: Callable,
    q_x: numpy.ndarray,
    framework: str,
    use_rounding: bool = False,
    output_n_bits: int = MAX_BITWIDTH_BACKWARD_COMPATIBLE,
) -> Tuple[Callable, List[UniformQuantizer], onnx.ModelProto]:
    """Convert the tree inference to a numpy functions using Hummingbird.

    Args:
        model (Callable): The tree model to convert.
        q_x (numpy.ndarray): The quantized input data.
        use_rounding (bool): Use rounding feature or not.
        framework (str): The framework from which the ONNX model is generated.
            (options: 'xgboost', 'sklearn')
        output_n_bits (int): The number of bits of the output. Default to 8.

    Returns:
        Tuple[Callable, List[QuantizedArray], onnx.ModelProto]: A tuple with a function that takes a
            numpy array and returns a numpy array, QuantizedArray object to quantize and de-quantize
            the output of the tree, and the ONNX model.
    """
    # mypy
    assert output_n_bits is not None

    assert_true(
        framework in ["xgboost", "sklearn"],
        f"framework={framework} is not supported. It must be either 'xgboost' or 'sklearn'",
    )

    # Execute with 1 example for efficiency in large data scenarios to prevent slowdown
    onnx_model = get_onnx_model(model, q_x[:1], framework)

    if use_rounding:
        # compute LSB to remove in stage 1 and 2
        # <!>: List[lsbs_to_remove_stage1, lsbs_to_remove_stage2]
        lsbs_to_remove = compute_lsb_to_remove_for_trees(onnx_model, q_x)
        replace_operator_with_rounded_version(onnx_model, lsbs_to_remove)

    replace_operator_with_rounded_version(onnx_model, [3, 3])

    # Get the expected number of ONNX outputs in the sklearn model.
    expected_number_of_outputs = 1 if is_regressor_or_partial_regressor(model) else 2

    # ONNX graph pre-processing to make the model FHE friendly
    # i.e., delete irrelevant nodes and cut the graph before the final ensemble sum)
    tree_onnx_graph_preprocessing(onnx_model, framework, expected_number_of_outputs)

    # Tree values pre-processing
    # i.e., mainly predictions quantization
    # but also rounding the threshold such that they are now integers
    q_y = tree_values_preprocessing(onnx_model, framework, output_n_bits)

    # Get the numpy inference for the quantized tree (_tree_inference).
    # Use check_model = False here since we have custom onnx operator that won't be recognised.
    _tree_inference, onnx_model = get_equivalent_numpy_forward_from_onnx(
        onnx_model, q_x, check_model=False
    )

    return (_tree_inference, [q_y.quantizer], onnx_model)


# Remove this function once the truncate feature is released
# FIXME: https://github.com/zama-ai/concrete-ml/issues/397
def compute_lsb_to_remove_for_trees(onnx_model: onnx.ModelProto, q_x: numpy.ndarray) -> List[int]:
    """Compute the LSB to remove for the comparison operators in the trees.

    Referring to this paper: https://scnakandala.github.io/papers/TR_2020_Hummingbird.pdf, there are
    2 levels of comparison for trees, one at the level of X.A < B and a second at
    the level of I.C == D.

    Args:
        onnx_model (onnx.ModelProto): The model to clean
        q_x (numpy.ndarray): The quantized inputs

    Returns:
        List: the number of LSB to remove for level 1 and level 2
    """

    def get_bitwidth(array: numpy.ndarray) -> int:
        """Compute the bitwidth required to represent the largest value in `array`.

        Args:
            array (umpy.ndarray): The array for which the bitwidth needs to be checked.

        Returns:
            int: The required bits to represent the array.
        """

        max_val = numpy.max(numpy.abs(array))
        # + 1 is added to include the sign bit
        bitwidth = math.ceil(math.log2(max_val + 1)) + 1
        return bitwidth

    def update_lsbs_if_overflow_detected(array: numpy.ndarray, initial_bitwidth: int) -> int:
        """Update the number of LSBs to remove based on overflow detection.

        Args:
            array (umpy.ndarray): The array for which the bitwidth needs to be checked.
            initial_bitwidth (int): The target bitwidth that should not be exceeded.

        Returns:
            int: The updated LSB to remove.
        """

        lsbs_to_remove = initial_bitwidth

        if lsbs_to_remove > 0:
            half = 1 << (lsbs_to_remove - 1)
            if get_bitwidth(array - half) <= initial_bitwidth:
                lsbs_to_remove -= 1

        return lsbs_to_remove

    quant_params = {
        onnx_init.name: numpy_helper.to_array(onnx_init)
        for onnx_init in onnx_model.graph.initializer
        if "weight" in onnx_init.name or "bias" in onnx_init.name
    }

    key_mat_1 = [key for key in quant_params.keys() if "_1" in key and "weight" in key][0]
    key_bias_1 = [key for key in quant_params.keys() if "_1" in key and "bias" in key][0]

    key_mat_2 = [key for key in quant_params.keys() if "_2" in key and "weight" in key][0]
    key_bias_2 = [key for key in quant_params.keys() if "_2" in key and "bias" in key][0]

    # shape: (nodes, features) or (trees * nodes, features)
    mat_1 = quant_params[key_mat_1]
    # shape: (nodes, 1) or (trees * nodes, 1)
    bias_1 = quant_params[key_bias_1]

    # shape: (trees, leaves, nodes)
    mat_2 = quant_params[key_mat_2]
    # shape: (leaves, 1) or (trees * leaves, 1)
    bias_2 = quant_params[key_bias_2]

    n_features = mat_1.shape[1]
    n_nodes = mat_2.shape[2]
    n_leaves = mat_2.shape[1]

    mat_1 = mat_1.reshape(-1, n_nodes, n_features)
    bias_1 = bias_1.reshape(-1, 1, n_nodes)
    bias_2 = bias_2.reshape(-1, 1, n_leaves)

    # If <= -> stage = biais_1 - (q_x @ mat_1.transpose(0, 2, 1))
    # If < -> stage = (q_x @ mat_1.transpose(0, 2, 1)) - biais_1
    stage_1 = bias_1 - (q_x @ mat_1.transpose(0, 2, 1))

    # The matrix I, as referenced in this paper:
    # https://scnakandala.github.io/papers/TR_2020_Hummingbird.pdf, results from the condition:
    # X.A < B and consists exclusively of binary elements, 1 and 0.
    # Given this assumption, we randomly generate it.
    matrix_q = numpy.random.randint(0, 2, size=(stage_1.shape))

    stage_2 = ((matrix_q @ mat_2.transpose(0, 2, 1)) + bias_2).sum(axis=0)

    lsbs_to_remove_1 = update_lsbs_if_overflow_detected(stage_1, get_bitwidth(stage_1))
    lsbs_to_remove_2 = update_lsbs_if_overflow_detected(stage_2, get_bitwidth(stage_2))

    return [lsbs_to_remove_1, lsbs_to_remove_2]
