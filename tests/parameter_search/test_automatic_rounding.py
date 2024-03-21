
import os
import warnings
from pathlib import Path
import re

import numpy
import pytest
import torch
from sklearn.datasets import make_classification
from concrete.fhe import Configuration

from concrete.ml.pytest.torch_models import TorchAutoRoundingTLUTester
from concrete.ml.common.preprocessors import TLUOptimizer, Exactness
from concrete.ml.torch.compile import compile_torch_model
from concrete.ml.sklearn import NeuralNetClassifier

from concrete.ml.quantization.base_quantized_op import QuantizedMixingOp
from concrete.ml.quantization.post_training import PowerOfTwoScalingRoundPBSAdapter

@pytest.mark.parametrize("is_conv", [True, False])
@pytest.mark.parametrize("tlu_test_mode", ["per_tensor", "per_cell", "per_neuron"])
def test_tlu_analysis_granularity(is_conv, tlu_test_mode):
    if not is_conv and tlu_test_mode == "per_cell":
        # Dense layers produce an output vector with neuron outputs
        # only per_neuron is supported for those layers
        return

    model = TorchAutoRoundingTLUTester(is_conv, tlu_test_mode)

    n_bits = 4
    tlu_optimizer = TLUOptimizer(
        rounding_threshold=n_bits, 
        verbose=True, n_bits_range_search=5, 
        exactness=Exactness.APPROXIMATE,
    )
    cfg = Configuration(
        verbose=True,
        show_optimizer=False,
        additional_pre_processors=[tlu_optimizer]
    )

    input_set = torch.FloatTensor(numpy.concatenate([numpy.random.uniform(size=model.input_shape) for _ in range(10)]))

    # Compile the quantized model
    quantized_numpy_module = compile_torch_model(
        model,
        input_set,
        n_bits=n_bits,
        configuration=cfg,
    )
    
    # Find optimized TLUs
    tlu_nodes = quantized_numpy_module.fhe_circuit.graph.query_nodes(
        custom_filter=lambda node: node.converted_to_table_lookup and "opt_round_a" in node.properties["attributes"],
        ordered=True,
    )

    # A single TLU should be present
    assert(len(tlu_nodes) == 1)

    tlu_node = tlu_nodes[0]
    scales_tlu = tlu_node.properties["attributes"]["opt_round_a"]
    offsets_tlu = tlu_node.properties["attributes"]["opt_round_b"]

    if tlu_test_mode == "per_tensor":
        assert(offsets_tlu.size == 1)
        assert(scales_tlu.size == 1)
    elif tlu_test_mode == "per_cell":
        assert(offsets_tlu.size == model.input_shape[2] * model.input_shape[3])
        assert(scales_tlu.size == model.input_shape[2] * model.input_shape[3])
    elif tlu_test_mode == "per_neuron":
        assert(offsets_tlu.size == model.n_neurons)
        assert(scales_tlu.size == model.n_neurons)
    else:
        assert False, "Invalid tlu granularity test mode"

@pytest.mark.parametrize("n_bits", range(2, 6))
def test_tlu_analysis_optimization(load_data, n_bits):

    params = {
        "module__n_w_bits": n_bits,
        "module__n_a_bits": n_bits,
        "module__n_layers": 2,
        "max_epochs": 10,
    }

    model = NeuralNetClassifier(**params)

    x, y = load_data(NeuralNetClassifier)

    model.fit(x, y)

    adapter = PowerOfTwoScalingRoundPBSAdapter(model.quantized_module_)
    round_pbs_patterns = adapter.process()
    # Remove rounding in the network to perform inference without the optimization.
    # We expect a network that was optimized with the power-of-two adapter
    # to be exactly correct to the non-optimized one
    for (_, node_op) in model.quantized_module_.quant_layers_dict.values():
        if isinstance(node_op, QuantizedMixingOp):
            node_op.rounding_threshold_bits = None
            node_op.lsbs_to_remove = None

    assert (
        len(round_pbs_patterns) == 0
    ), "Expected number of round PBS optimized patterns was not matched"
    assert (
        adapter.num_ignored_valid_patterns == 1
    ), "Expected number of ignored round PBS optimizable patterns was not matched"

    tlu_optimizer = TLUOptimizer(
        rounding_threshold=n_bits, 
        verbose=True, n_bits_range_search=5, 
        exactness=Exactness.APPROXIMATE,
    )
    cfg = Configuration(
        verbose=True,
        show_optimizer=False,
        additional_pre_processors=[tlu_optimizer]
    )

    # Compile the quantized model
    model.compile(
        x,
        configuration=cfg,
        verbose=True
    )
    
    # Find optimized TLUs
    tlu_nodes = model.quantized_module_.fhe_circuit.graph.query_nodes(
        custom_filter=lambda node: node.converted_to_table_lookup and "opt_round_a" in node.properties["attributes"],
        ordered=True,
    )

    # A single TLU should be present
    assert(len(tlu_nodes) == 1)

    tlu_node = tlu_nodes[0]
    scales_tlu = tlu_node.properties["attributes"]["opt_round_a"]
    offsets_tlu = tlu_node.properties["attributes"]["opt_round_b"]

    count_reinterpret = 0
    for line in model.quantized_module_.fhe_circuit.mlir.split("\n"):
        if "reinterpret_precision" in line:
            regex = r"-> tensor<(\d+x)*\!FHE.[A-z]+<(\d+)"
            
            matches = re.finditer(regex, line.strip())
            for matchNum, m in enumerate(matches, start=1):
                bw = int(m.group(2))
                if bw > 24: 
                    continue
                assert(bw == n_bits)
                count_reinterpret += 1

    assert count_reinterpret > 0, "Could not find reinterpret_cast nodes in graph to analyze"

    pass

@pytest.mark.parametrize("n_bits", range(4, 9))
def test_tlu_optimization_cryptoparam_finding(load_data, n_bits):

    params = {
        "module__n_w_bits": n_bits,
        "module__n_a_bits": n_bits,
        "module__n_layers": 2,
        "max_epochs": 10,
    }

    model = NeuralNetClassifier(**params)

    x, y = load_data(NeuralNetClassifier)

    model.fit(x, y)

    compile_ok_with_adjust = True
    try:
        tlu_optimizer = TLUOptimizer(
            rounding_threshold=n_bits, 
            verbose=True, 
            n_bits_range_search=5, 
            exactness=Exactness.APPROXIMATE,
        )
        cfg = Configuration(
            show_optimizer=False,
            additional_pre_processors=[tlu_optimizer]
        )

        # Compile the quantized model
        model.compile(
            x,
            configuration=cfg,
        )
    except RuntimeError as err:
        if len(err.args) and err.args[0] == "NoParametersFound":
            compile_ok_with_adjust = False
        else:
            raise err

    compile_ok_with_no_adjust = True
    try:
        # Compile the quantized model
        cfg = Configuration(
            show_optimizer=False,
        )

        # Compile the quantized model
        model.compile(
            x,
            configuration=cfg,
        )
    except RuntimeError as err:
        if len(err.args) and err.args[0] == "NoParametersFound":
            compile_ok_with_no_adjust = False
        else:
            raise err


    assert(compile_ok_with_adjust == compile_ok_with_no_adjust)
