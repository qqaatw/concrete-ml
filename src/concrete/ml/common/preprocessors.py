"""
Declaration of `TLUOptimizer` graph processor.
"""

from copy import deepcopy
from itertools import product
from typing import Dict, List, Optional, Union

import networkx as nx
import numpy as np
from concrete.fhe import Exactness, round_bit_pattern, truncate_bit_pattern
from concrete.fhe.dtypes import Float, Integer
from concrete.fhe.representation import Graph, GraphProcessor, Node, Operation
from concrete.fhe.representation.evaluator import ConstantEvaluator
from concrete.fhe.values.value_description import ValueDescription
from tqdm.auto import tqdm

P_ERROR_PER_ERROR_SIZE_CACHE = []


def add(x, y):
    return x + y


def subtract(x, y):
    return x - y


def multiply(x, y):
    return x * y


def divide(x, y):
    return x / y


def is_node_tlu(node):
    return node.converted_to_table_lookup


class CycleDetector(GraphProcessor):
    """
    CycleDetector graph processor, to detect cycles.
    """

    def __init__(
        self,
    ):
        pass

    def apply(self, graph: Graph):
        # Get all nodes that will be converted to LUTs
        tlu_nodes = graph.query_nodes(
            custom_filter=is_node_tlu,
        )
        cycles = nx.recursive_simple_cycles(graph.graph)
        if cycles:
            raise Exception()

        for tlu_node in tlu_nodes:
            if "subgraph" in tlu_node.evaluator.properties["kwargs"]:
                tlu_subgraph: Graph = tlu_node.evaluator.properties["kwargs"]["subgraph"]
                cycles = nx.recursive_simple_cycles(tlu_subgraph.graph)
                if cycles:
                    raise Exception()



class InsertRounding(GraphProcessor):
    """
    InsertRounding graph processor, to add rounding before TLUs if desired.
    """

    rounding_threshold: Optional[int]

    def __init__(
        self,
        threshold: Optional[int],
        exactness: Exactness = Exactness.EXACT,
    ):
        self.rounding_threshold = threshold
        self.exactness = exactness

    def apply(self, graph: Graph):
        if self.rounding_threshold is None:
            # No rounding
            return

        # Get all nodes that will be converted to LUTs
        tlu_nodes = graph.query_nodes(
            custom_filter=is_node_tlu,
        )
        for tlu_node in tlu_nodes:
            # Predecessor nodes
            pred_nodes = graph.ordered_preds_of(tlu_node)

            # Only take into accound predecessor's that aren't constants
            variable_input_indices = []
            for pred_index, pred_node in enumerate(pred_nodes):
                if pred_node.operation != Operation.Constant:
                    variable_input_indices.append(pred_index)

            # Only one input should be non-constant per LUT
            # TODO: verify this is actually true
            if len(variable_input_indices) != 1:
                continue

            # Get variable input
            variable_input_index = variable_input_indices[0]
            variable_input_node = pred_nodes[variable_input_index]
            variable_input_dtype = variable_input_node.output.dtype

            if not isinstance(variable_input_dtype, Integer):
                raise ValueError(f"{variable_input_dtype=} is not 'Integer'")

            variable_input_bit_width = variable_input_dtype.bit_width
            if variable_input_bit_width <= self.rounding_threshold:
                # No need to do anything if the bit-width is actually lower or equal
                # to the rounding threshold value
                continue

            # Compute lsbs to remove
            lsbs_to_remove = variable_input_bit_width - self.rounding_threshold

            # Rounding node
            rounding_node = Node.generic(
                "round_bit_pattern",
                [deepcopy(variable_input_node.output)],
                deepcopy(variable_input_node.output),
                round_bit_pattern,
                kwargs={
                    "lsbs_to_remove": lsbs_to_remove,
                    "overflow_protection": False,
                    "exactness": self.exactness,
                },
                attributes={
                    "overflow_protection": False,
                },
            )
            rounding_node.properties["final_lsbs_to_remove"] = lsbs_to_remove
            rounding_node.properties["resulting_bit_width"] = self.rounding_threshold
            rounding_node.properties["overflow_protection"] = False
            rounding_node.properties["overflow_detected"] = False
            rounding_node.properties["exactness"] = self.exactness

            nx_graph = graph.graph
            nx_graph.add_edge(variable_input_node, rounding_node, input_idx=0)

            edge_data = nx_graph.get_edge_data(variable_input_node, tlu_node).values()
            for data in list(edge_data):
                input_idx = data["input_idx"]
                nx_graph.add_edge(rounding_node, tlu_node, input_idx=input_idx)

            nx_graph.remove_edge(variable_input_node, tlu_node)


# We outsource the computation of the subgraph to this function to have faster inference
# This function is currently the bottleneck to the a, b optimization pipeline
def vectorized_graph_eval(graph, *inputs, sorted_nodes: Optional[List] = None):
    # pylint: disable=no-member,too-many-nested-blocks,too-many-branches,too-many-statements

    node_results: Dict[Node, Union[np.bool_, np.integer, np.floating, np.ndarray]] = {}
    if sorted_nodes is None:
        sorted_nodes = list(nx.topological_sort(graph.graph))

    for node in sorted_nodes:
        if node.operation == Operation.Input:
            node_results[node] = node.evaluator(inputs[graph.input_indices[node]])
            continue

        pred_results = [
            deepcopy(node_results[pred]) for pred in graph.ordered_preds_of(node)
        ]
        try:
            node_results[node] = node.evaluator(*pred_results)
        except Exception as error:
            raise RuntimeError(
                "Evaluation of the graph failed\n\n"
                + graph.format(
                    highlighted_nodes={node: ["evaluation of this node failed"]},
                    show_bounds=False,
                )
            ) from error

    result = tuple(node_results[node] for node in graph.ordered_outputs())
    return result if len(result) > 1 else result[0]


# We outsource the computation of the subgraph to this function to have faster inference
# This function is currently the bottleneck to the a, b optimization pipeline
def vectorized_graph_eval_all(graph, *inputs, sorted_nodes: Optional[List] = None):
    # pylint: disable=no-member,too-many-nested-blocks,too-many-branches,too-many-statements

    node_inputs: Dict[Node, Union[np.bool_, np.integer, np.floating, np.ndarray]] = {}
    node_results: Dict[Node, Union[np.bool_, np.integer, np.floating, np.ndarray]] = {}
    if sorted_nodes is None:
        sorted_nodes = list(nx.topological_sort(graph.graph))

    for node in sorted_nodes:
        if node.operation == Operation.Input:
            node_results[node] = node.evaluator(inputs[graph.input_indices[node]])
            continue

        pred_results = [
            deepcopy(node_results[pred]) for pred in graph.ordered_preds_of(node)
        ]
        try:
            node_inputs[node] = node.evaluator(*pred_results)
            node_results[node] = node.evaluator(*pred_results)
        except Exception as error:
            raise RuntimeError(
                "Evaluation of the graph failed\n\n"
                + graph.format(
                    highlighted_nodes={node: ["evaluation of this node failed"]},
                    show_bounds=False,
                )
            ) from error

    return node_results, node_inputs


# TODO: serialize the output of the TLU optimizer to be able to re-use the results without having to re-do the whole search
# i.e. some cache system -> basically if the subgraph is the same and the bounds too
class TLUOptimizer(GraphProcessor):
    """
    TLUOptimizer graph processor, to add approximate rounding and scaling before/in TLUs if desired.
    """

    rounding_threshold: Optional[int]

    def __init__(
        self,
        rounding_threshold: int = 6,
        verbose: bool = True,
        n_bits_range_search: int = 2,
        exactness: Exactness = Exactness.EXACT,
        dataset: Optional[List[np.ndarray]] = None,
        overflow_protection: bool = True,
        rounding_function=round_bit_pattern,
    ):
        self.rounding_threshold = rounding_threshold
        self.verbose = verbose
        self.n_bits_range_search = n_bits_range_search
        self.exactness = exactness
        self.dataset = [dataset] if isinstance(dataset, np.ndarray) else dataset
        self.overflow_protection = overflow_protection
        self.rounding_function = rounding_function
        if self.n_bits_range_search > 3:
            self.dump_all_figures = False

    def apply(self, graph: Graph):
        if self.rounding_threshold is None:
            return

        # If dataset we need to compute the input/output of all tlu nodes
        if self.dataset is not None:
            all_nodes_results, all_node_inputs = vectorized_graph_eval_all(
                graph, *self.dataset
            )
        else:
            all_nodes_results, all_node_inputs = None, None

        del all_nodes_results

        tlu_nodes = graph.query_nodes(
            custom_filter=is_node_tlu,
            ordered=True,
        )

        for tlu_index, tlu_node in enumerate(tlu_nodes):
            # On each tlu we do:
            # 1. Optimize a and b for the subgraph
            # 2. Insert a and b in the graph
            # 3. Insert rounding to the graph
            # 4. Insert a and b in the subgraph

            tlu_subgraph: Graph = tlu_node.evaluator.properties["kwargs"]["subgraph"]

            if self.verbose:
                print(tlu_subgraph.format())

            pred_nodes = graph.ordered_preds_of(tlu_node)
            variable_input_indices = []
            for pred_index, pred_node in enumerate(pred_nodes):
                if pred_node.operation != Operation.Constant:
                    variable_input_indices.append(pred_index)

            if len(variable_input_indices) != 1:
                raise ValueError("TLU node got more than one variable input")

            # TODO: assert that there isn't any rounding nodes before

            variable_input_index = variable_input_indices[0]
            variable_input_node: Node = pred_nodes[variable_input_index]
            variable_input_dtype = variable_input_node.output.dtype

            if not isinstance(variable_input_dtype, Integer):
                raise ValueError("TLU node got input dtype that isn't integer")

            # ---------------------------------------------
            # Search a and b that reduce the MSE of the TLU
            # ---------------------------------------------

            # Extract TLU input bounds
            if variable_input_node.bounds is not None:
                min_bound, max_bound = variable_input_node.bounds
                assert isinstance(min_bound, int) or (
                    isinstance(min_bound, np.ScalarType) and min_bound.dtype == np.int64
                ), f"{type(min_bound)=}"
                assert isinstance(max_bound, int) or (
                    isinstance(max_bound, np.ScalarType) and max_bound.dtype == np.int64
                ), f"{type(max_bound)=}"
            else:
                raise ValueError("Bounds not found")

            # Create input with proper shape on the bounds for optimization
            # TODO: -----------
            # properly reshape the input according to the axis that are actually different
            # in the LUT
            # We should handle shape > 2
            # Actually we could have per element TLU encoded in the subgraph
            # So we should detect which shape we could remove
            # the reduce sum should be along an axis and we should gather the best_a and best_b per axis
            # -> just go through all constants and detect which shapes are not 1
            # -> if a value in the shape is not 1, check if the value is always the same or not
            # -----------------
            expected_shape = variable_input_node.output.shape
            if self.verbose:
                print(f"{variable_input_node.output.shape=}")

            # Gather the shapes of the different constants in the subgraph
            # TODO: also check for uniqueness in the constant
            # if the all values in the tensor are the same we don't need to take it into account
            constant_shapes = list()
            for elt in tlu_subgraph.graph.nodes:
                assert isinstance(elt, Node)
                if isinstance(elt.evaluator, ConstantEvaluator):
                    constant_shape = elt.output.shape
                    if constant_shape:
                        constant_shapes.append(constant_shape)

            # For each axis take the max value for the constant
            if constant_shapes:
                shape_ = tuple(
                    [
                        max(
                            constant_shape[idx]
                            for constant_shape in constant_shapes
                            if len(constant_shape) > idx
                        )
                        for idx in range(max(len(elt) for elt in constant_shapes))
                    ]
                )
                reduce_axes = tuple([idx for idx, elt in enumerate(shape_) if elt == 1])
            else:
                shape_ = tuple()
                reduce_axes = tuple([idx for idx in range(len(expected_shape))])

            if all_node_inputs is None:
                subgraph_inputs = np.array(
                    list(range(int(min_bound), int(max_bound) + 1))
                )
                subgraph_input_shape = tuple([len(subgraph_inputs), *shape_[1:]])

                if len(expected_shape) > 1:
                    assert expected_shape[0] == 1
                    subgraph_inputs = np.tile(
                        subgraph_inputs[
                            tuple(
                                [
                                    slice(0, len(subgraph_inputs), 1),
                                    *[
                                        np.newaxis
                                        for _ in range(len(expected_shape) - 1)
                                    ],
                                ]
                            )
                        ],
                        expected_shape,
                    )
                    subgraph_inputs = subgraph_inputs[
                        tuple([slice(0, elt, 1) for elt in subgraph_input_shape])
                    ]
            else:
                subgraph_inputs = all_node_inputs[tlu_node]

            if self.verbose:
                print(f"{subgraph_inputs.shape=} used for optimization")

            # Compute TLU output on bounds without rounding or calibration
            sorted_nodes = list(nx.topological_sort(tlu_subgraph.graph))
            reference = vectorized_graph_eval(
                tlu_subgraph, subgraph_inputs, sorted_nodes=sorted_nodes
            )

            # Compute with rounding but without calibration
            accumulator_bit_width = Integer.that_can_represent(
                subgraph_inputs
            ).bit_width
            lsbs_to_remove = accumulator_bit_width - self.rounding_threshold
            assert isinstance(subgraph_inputs, np.ndarray)
            if lsbs_to_remove > 0:
                x = round_bit_pattern(subgraph_inputs, lsbs_to_remove=lsbs_to_remove)
            else:
                x = subgraph_inputs
            assert isinstance(x, np.ndarray)
            approximated_no_calib = vectorized_graph_eval(
                tlu_subgraph,
                x,
                sorted_nodes=sorted_nodes,
            )
            assert isinstance(reference, np.ndarray)
            assert isinstance(approximated_no_calib, np.ndarray)
            mse_no_calib = ((reference - approximated_no_calib) ** 2).mean(
                axis=reduce_axes, keepdims=True
            )

            optimization = "exact"
            if optimization == "full_scale":
                # Map [min-bound, max-bound] -> [-2**n, 2**n] with linear scaling

                # Calibrate
                target_bit_width = 20
                range_size = max_bound - min_bound
                b_prime = np.floor(min_bound + range_size / 2)
                a_prime = np.floor(
                    2 ** (target_bit_width - 1)
                    / (max((min_bound - b_prime), (max_bound - b_prime)) / 2)
                )
                best_a = (
                    np.zeros((1,) + subgraph_inputs.shape[1:], dtype=np.int64) + a_prime
                ).astype(np.int64)
                best_b = (
                    np.zeros((1,) + subgraph_inputs.shape[1:], dtype=np.int64) - b_prime
                ).astype(np.int64)

                # Compute mse with calibration
                x = (subgraph_inputs + best_b) * best_a
                accumulator_bit_width = Integer.that_can_represent(
                    [x.min(), x.max()]
                ).bit_width
                lsbs_to_remove = accumulator_bit_width - self.rounding_threshold
                if lsbs_to_remove > 0:
                    x = round_bit_pattern(x, lsbs_to_remove=lsbs_to_remove)
                assert isinstance(x, np.ndarray)
                x = (x.astype(np.float64) / best_a.astype(np.float64)) - best_b.astype(
                    np.float64
                )
                approximated_calib = vectorized_graph_eval(
                    tlu_subgraph,
                    x,
                    sorted_nodes=sorted_nodes,
                )
                mse_calib = ((reference - approximated_calib) ** 2).mean(
                    axis=reduce_axes, keepdims=True
                )
                print(f"{mse_calib.max()=} {mse_no_calib.max()=}")
                best_mse = mse_calib

            elif optimization == "exact":
                # Match the GCD of the steps k*2**n

                # Initialize a and b such that no changes are done
                best_a = (
                    np.ones((1,) + subgraph_inputs.shape[1:], dtype=np.int64)
                ).astype(np.int64)
                best_b = (
                    np.zeros((1,) + subgraph_inputs.shape[1:], dtype=np.int64)
                ).astype(np.int64)
                assert isinstance(reference, np.ndarray)
                n_elems = reference.shape[0]
                x_min, x_max = (
                    min_bound,
                    max_bound,
                )  # Data-bounds (TODO: should be per-axis)

                # Compute mask of values for which there is a change
                change_mask = np.concatenate(
                    [
                        np.zeros(reference[:1].shape).astype(bool),
                        np.diff(reference, axis=0).astype(bool),
                    ]
                ).astype(bool)

                # Some accumulators
                deltas = np.zeros(reference.shape[1:], dtype=np.int64)
                rounding_thresholds = np.zeros(reference.shape[1:], dtype=np.int64)

                # Apply on all elements
                # TODO: vectorize this
                for indexes in tqdm(
                    product(*[range(elt) for elt in reference.shape[1:]])
                ):
                    selection = tuple([slice(0, n_elems), *indexes])
                    steps_indexes = subgraph_inputs[selection][change_mask[selection]]

                    if len(steps_indexes) == 0:
                        # Nothing to do if the function is constant
                        continue

                    th_0 = steps_indexes[0]  # First x such f(x-1) != f(x)
                    delta_axis = np.diff(steps_indexes, axis=0)  # all step sizes

                    if len(delta_axis) == 0:
                        # Single jump
                        # TODO: handle this case
                        # Should only be a matter of setting b to the proper offset
                        # Some stuff to consider with the bounds probably
                        continue

                    # Get the common delta between all steps
                    delta = np.gcd.reduce(delta_axis, dtype=np.int64)
                    deltas[indexes] = delta
                    rounding_threshold = np.ceil(np.log2((x_max - x_min) / delta)).astype(np.int64)
                    rounding_thresholds[indexes] = rounding_threshold

                    # Find new limits such that we have smallest bounds that include actual bounds and
                    # can be expressed as th_0 + (k * delta)
                    x_delta_min = (
                        th_0
                        - ((th_0 - x_min) // delta) * delta
                        - bool(x_min % delta) * delta
                    )
                    x_delta_max = (
                        th_0
                        + ((x_max - th_0) // delta) * delta
                        + bool(x_max % delta) * delta
                    )

                    # Number of elements in the new range for the given step size
                    n_parts = (x_delta_max - x_delta_min) / delta
                    n_round = np.ceil(np.log2(n_parts)).astype(np.int64)
                    assert n_round <= rounding_threshold, f"{n_round=} > {rounding_threshold=}"
                    # print(f"{n_parts=}, {n_round=} {rounding_threshold=}")

                    exceed = ((2**n_round)) - n_parts
                    left_bound_add = np.ceil(exceed / 2).astype(np.int64)
                    right_bound_add = np.floor(exceed / 2).astype(np.int64)
                    assert left_bound_add + right_bound_add == exceed
                    # print(left_bound_add, right_bound_add)

                    x_delta_min -= left_bound_add * delta
                    x_delta_max += right_bound_add * delta
                    # print("DEBUG: ", np.log2(((x_delta_max - x_delta_min) / delta)+1))

                    middle = (x_delta_max - x_delta_min) / 2
                    middle = np.median(np.arange(x_delta_min, x_delta_max + 1, delta))
                    # print(f"{x_delta_min-middle=}, {x_delta_max-middle=}")
                    # print(f"{x_delta_min=}, {x_delta_max=}")

                    mult_first = False
                    rounding_function = self.rounding_function

                    # Find the proper n
                    n = 20
                    if mult_first:
                        a_prime = ((2**n) - delta) / (x_delta_max - x_delta_min)

                        b_prime = -(2 ** (n - 1)) - (
                            x_delta_min * ((2**n - 1) / (x_delta_max - x_delta_min))
                        )
                        a_prime = np.floor(a_prime).astype(np.int64)
                        b_prime = np.floor(b_prime).astype(np.int64)
                    else:
                        a_prime = (2**n - 1) / (x_delta_max - x_delta_min)
                        a_prime = np.floor(a_prime).astype(np.int64)
                        b_prime = middle
                        if rounding_function == round_bit_pattern:
                            b_prime += delta / 2
                        else:
                            b_prime += 0
                        # print(f"{a_prime=}, {b_prime=}")

                        a_prime = np.floor(a_prime).astype(np.int64)
                        b_prime = np.floor(b_prime).astype(np.int64)

                    # print(f"{a_prime=}, {b_prime=}")
                    best_indexes = tuple([0, *indexes])

                    best_a[best_indexes] = a_prime
                    best_b[best_indexes] = b_prime

                    if False:
                        # Plot TLU
                        import matplotlib.pyplot as plt

                        fig, ax = plt.subplots()
                        ax.step(
                            subgraph_inputs[selection],
                            reference[selection],
                            where="post",
                        )
                        ax.set_title(f"{delta=}, {rounding_threshold=}")
                        fig.savefig(
                            f"./{tlu_index}_{'_'.join(str(elt) for elt in indexes)}.png"
                        )
                        plt.close(fig)

            elif optimization == "exhaustive_search":
                # Exhaustive search on a and b with for-loop
                # Reshape a and b to the size of the input
                assert isinstance(variable_input_node.output, ValueDescription)
                best_a = np.ones((1,) + subgraph_inputs.shape[1:], dtype=np.int64)
                best_b = np.zeros((1,) + subgraph_inputs.shape[1:], dtype=np.int64)
                # We can't use np.inf here because with mult with 0 it results in np.nan
                best_mse = (
                    np.ones((1,) + subgraph_inputs.shape[1:], dtype=np.float64)
                    * np.finfo(np.float64).max
                )

                # log 2 things, the mse with respect to rounding with no calibration and with no rounding no calibration
                x = (subgraph_inputs + best_b) * best_a
                accumulator_bit_width = Integer.that_can_represent(x).bit_width
                lsbs_to_remove = accumulator_bit_width - self.rounding_threshold
                if lsbs_to_remove > 0:
                    x = round_bit_pattern(x, lsbs_to_remove=lsbs_to_remove)
                assert isinstance(x, np.ndarray)
                x = (x.astype(np.float64) / best_a.astype(np.float64)) - best_b.astype(
                    np.float64
                )
                approximated_no_calib = vectorized_graph_eval(
                    tlu_subgraph,
                    x,
                    sorted_nodes=sorted_nodes,
                )

                # Search
                a_values = np.arange(1, 2**self.n_bits_range_search)
                b_values = np.arange(
                    -(2**self.n_bits_range_search) + 1, 2**self.n_bits_range_search
                )
                print(f"Search a in [{a_values.min()}, {a_values.max()}]")
                print(f"Search b in [{b_values.min()}, {b_values.max()}]")
                for a, b in tqdm(
                    product(a_values, b_values), total=len(a_values) * len(b_values)
                ):
                    a_arr = np.ones_like(best_a, dtype=np.int64) * a
                    b_arr = np.ones_like(best_b, dtype=np.int64) * b
                    x = (subgraph_inputs + b_arr) * a_arr
                    accumulator_bit_width = Integer.that_can_represent(x).bit_width
                    lsbs_to_remove = accumulator_bit_width - self.rounding_threshold
                    # if self.verbose:
                    #     print(f"{accumulator_bit_width=}, {lsbs_to_remove=} {self.rounding_threshold=}")
                    if lsbs_to_remove > 0:
                        x = round_bit_pattern(x, lsbs_to_remove=lsbs_to_remove)
                    assert isinstance(x, np.ndarray)
                    x = (
                        x.astype(np.float64) / a_arr.astype(np.float64)
                    ) - b_arr.astype(np.float64)
                    approximated = vectorized_graph_eval(
                        tlu_subgraph,
                        x,
                        sorted_nodes=sorted_nodes,
                    )
                    assert isinstance(reference, np.ndarray)
                    assert isinstance(approximated, np.ndarray)
                    mse = ((reference - approximated) ** 2).mean(
                        axis=reduce_axes, keepdims=True
                    )
                    # print(f"{mse=} , {best_mse=}")
                    mask = mse < best_mse
                    best_mse = mse * mask + (best_mse * (~mask))

                    best_a = a_arr * mask + (best_a * (~mask))
                    best_b = b_arr * mask + (best_b * (~mask))

                    # This is probably more efficient?
                    # best_a[mask] = a
                    # best_b[mask] = b

                    # DEBUG
                    if self.dump_all_figures:
                        import matplotlib.pyplot as plt

                        select_slices = tuple(
                            [
                                slice(0, len(subgraph_inputs)),
                                *(0 for _ in subgraph_inputs.shape[1:]),
                            ]
                        )
                        fig, ax = plt.subplots()
                        ax.step(
                            subgraph_inputs[select_slices],
                            reference[select_slices],
                            label="reference",
                        )
                        ax.step(
                            subgraph_inputs[select_slices],
                            approximated_no_calib[select_slices],
                            label="rounded - no calib",
                            linestyle="dotted",
                        )
                        ax.step(
                            subgraph_inputs[select_slices],
                            approximated[select_slices],
                            label="calibrated",
                            linestyle="dashed",
                        )
                        ax.legend()
                        ax.set_title(f"{a=} {b=} {tlu_index=} {select_slices=}")
                        fig.savefig(f"{a}_{b}_{tlu_index}.png")
                        plt.close(fig)

                    # TODO: We could also take the accumulator size into account as a criterion
                    # for selection, for example if exactness is EXACT
                    if (best_mse == 0.0).all():
                        break

            # We need to make sure that we have the correct shape when adding constants in the graph
            # As Concrete Python doesn't handle broadcasting at the graph level
            assert isinstance(variable_input_node.output, ValueDescription)
            best_a = np.broadcast_to(
                best_a, shape=(1,) + variable_input_node.output.shape[1:]
            )
            best_b = np.broadcast_to(
                best_b, shape=(1,) + variable_input_node.output.shape[1:]
            )
            print(f"DEBUG: {best_a.shape=}, {best_b.shape=}")
            print(
                f"DEBUG: {np.ceil(np.log2(np.abs(best_a).max()))=}, {np.ceil(np.log2(np.abs(best_b).max()))+1=}, "
            )

            # ------------------------------------------
            # Add scaling and rounding to the main graph
            # ------------------------------------------

            # Add b
            b_constant_node = Node.constant(best_b)
            b_constant_node.output = deepcopy(variable_input_node.output)
            b_constant_node.output.is_encrypted = False
            b_constant_node.output.dtype.bit_width = (
                variable_input_node.output.dtype.bit_width
            )  # Node needs to have same bit width as input
            # TODO: validate to see what happens if b bit-width is above the input bit-width
            b_constant_node.properties["bit_width"] = (
                variable_input_node.output.dtype.bit_width
            )
            add_bounds = ((min_bound - best_b).min(), (max_bound - best_b).max())
            result_dtype = Integer.that_can_represent(list(add_bounds))
            add_node = Node.generic(
                name="substract",
                inputs=[
                    deepcopy(variable_input_node.output),
                    deepcopy(b_constant_node.output),
                ],
                output=ValueDescription(
                    dtype=result_dtype,
                    shape=variable_input_node.output.shape,
                    is_encrypted=variable_input_node.output.is_encrypted,
                ),
                operation=subtract,
            )
            add_node.bounds = add_bounds

            # Multiply by a
            a_constant_node = Node.constant(best_a)
            a_constant_node.output = deepcopy(variable_input_node.output)
            a_constant_node.output.is_encrypted = False
            a_constant_node.output.dtype.bit_width = (
                add_node.output.dtype.bit_width
            )  # + 1
            a_constant_node.properties["bit_width"] = add_node.output.dtype.bit_width
            mul_bounds = (
                ((min_bound - best_b) * best_a).min(),
                ((max_bound - best_b) * best_a).max(),
            )
            result_dtype = Integer.that_can_represent(list(mul_bounds))
            mul_node = Node.generic(
                name="multiply",
                inputs=[deepcopy(add_node.output), deepcopy(a_constant_node.output)],
                output=ValueDescription(
                    dtype=result_dtype,
                    shape=variable_input_node.output.shape,
                    is_encrypted=variable_input_node.output.is_encrypted,
                ),
                operation=multiply,
            )
            mul_node.bounds = mul_bounds

            # Add edge between TLU-input variable and rounding node
            nx_graph = graph.graph
            nx_graph.add_edge(variable_input_node, add_node, input_idx=0)
            nx_graph.add_edge(b_constant_node, add_node, input_idx=1)
            nx_graph.add_edge(add_node, mul_node, input_idx=0)
            nx_graph.add_edge(a_constant_node, mul_node, input_idx=1)

            # Add rounding
            assert isinstance(mul_node.output.dtype, Integer)
            lsbs_to_remove = mul_node.output.dtype.bit_width - self.rounding_threshold
            if self.verbose:
                print(
                    f"Removing {lsbs_to_remove} lsbs to {mul_node.output.dtype.bit_width} bits"
                )

            if lsbs_to_remove > 0:
                # Create rounding node
                rounding_node = Node.generic(
                    name=rounding_function.__name__,
                    inputs=[deepcopy(mul_node.output)],
                    output=deepcopy(mul_node.output),
                    operation=rounding_function,
                    kwargs={
                        "lsbs_to_remove": lsbs_to_remove,
                        "exactness": self.exactness,
                        "overflow_protection": self.overflow_protection,
                    },
                    attributes={
                        "overflow_protection": self.overflow_protection,
                    },
                )
                rounding_node.properties["final_lsbs_to_remove"] = lsbs_to_remove
                rounding_node.properties["resulting_bit_width"] = (
                    self.rounding_threshold
                )
                rounding_node.properties["overflow_detected"] = False
                rounding_node.properties["overflow_protection"] = False
                rounding_node.properties["exactness"] = self.exactness
                rounding_node.properties["original_input_bit_width"] = (
                    mul_node.output.dtype.bit_width
                )
                nx_graph.add_edge(mul_node, rounding_node, input_idx=0)

                # Add connexion between rounded value and nodes that use it as input
                edge_data = nx_graph.get_edge_data(
                    variable_input_node, tlu_node
                ).values()
                for data in list(edge_data):
                    input_idx = data["input_idx"]
                    nx_graph.add_edge(rounding_node, tlu_node, input_idx=input_idx)
            else:
                nx_graph.add_edge(mul_node, tlu_node, input_idx=0)

            # Remove edge between TLU-input node and TLU node
            nx_graph.remove_edge(variable_input_node, tlu_node)

            # -----------------------------
            # Sub-graph / TLU modifications
            # -----------------------------
            # Detect the input node
            input_node = None
            for node in tlu_subgraph.graph:
                if (
                    "name" in node.properties
                    and node.properties["name"] == "astype"
                    and node.properties["kwargs"]["dtype"] == np.float64
                ):
                    assert (
                        input_node is None
                    ), "More than one astype float node detected"
                    input_node = node
            assert input_node is not None, "Couldn't detect astype float node"

            # input_node = next(nx.topological_sort(tlu_subgraph.graph))

            # Create constant nodes
            cst_a = Node.constant(best_a.astype(np.float64))
            # cst_a_bit_width = max(int(np.ceil(np.log2(max(1, np.abs(best_a).max())))), 1)
            cst_a.output = ValueDescription(
                # dtype=Integer(bit_width=cst_a_bit_width, is_signed=True),
                dtype=Float(64),
                is_encrypted=False,
                shape=best_a.shape,
            )
            cst_b = Node.constant(best_b.astype(np.float64))
            # cst_b_bit_width = max(int(np.ceil(np.log2(max(1, np.abs(best_b).max())))), 1)
            cst_b.output = ValueDescription(
                # dtype=Integer(bit_width=cst_b_bit_width, is_signed=True),
                dtype=Float(64),
                is_encrypted=False,
                shape=best_b.shape,
            )
            div_node = Node.generic(
                name="divide",
                operation=divide,
                inputs=[input_node.output, cst_a.output],
                output=ValueDescription(
                    # dtype=Integer(bit_width=64, is_signed=True), is_encrypted=True, shape=input_node.output.shape
                    dtype=Float(bit_width=64),
                    is_encrypted=True,
                    shape=input_node.output.shape,
                ),
            )
            sub_node = Node.generic(
                name="add",
                operation=add,
                inputs=[div_node.output, cst_b.output],
                output=ValueDescription(
                    # dtype=Integer(bit_width=64, is_signed=True), is_encrypted=True, shape=input_node.output.shape
                    dtype=Float(bit_width=64),
                    is_encrypted=True,
                    shape=input_node.output.shape,
                ),
            )

            # Replace all edges going from the input node with edges that start from the
            # substraction node
            edges = list(tlu_subgraph.graph.out_edges(input_node))
            for in_node, out_node in edges:
                # We should preserve the input_idx
                edge_data = dict(tlu_subgraph.graph.get_edge_data(in_node, out_node))
                tlu_subgraph.graph.remove_edge(in_node, out_node)
                tlu_subgraph.graph.add_edge(
                    sub_node, out_node, input_idx=edge_data[0]["input_idx"]
                )

            # Add re-scaling edges
            tlu_subgraph.graph.add_edge(input_node, div_node, input_idx=0)
            tlu_subgraph.graph.add_edge(cst_a, div_node, input_idx=1)
            tlu_subgraph.graph.add_edge(div_node, sub_node, input_idx=0)
            tlu_subgraph.graph.add_edge(cst_b, sub_node, input_idx=1)

            print("TLU Subgraph after transform")
            print(tlu_subgraph.format())

            # TODO: DEBUGGING SANITY CHECK WHERE WE VALIDATE THAT THE BEST MSE IS INDEED ACHIEVED
            # WITH BEST_A and BEST_B using the subgraph to do the computation
            print("Doing sanity check")
            sanity_check = False
            if sanity_check:
                x = (subgraph_inputs + best_b) * best_a
                if lsbs_to_remove > 0:
                    x = rounding_function(
                        x=x,
                        lsbs_to_remove=lsbs_to_remove,
                        overflow_protection=self.overflow_protection,
                        exactness=self.exactness,
                    )
                tlu_ouput = vectorized_graph_eval(tlu_subgraph, x)
                assert isinstance(tlu_ouput, np.ndarray)
                mse = ((reference - tlu_ouput) ** 2).mean(
                    axis=reduce_axes, keepdims=True
                )
                print(f"{mse=}")

                if True:
                    print("plotting")
                    import matplotlib.pyplot as plt

                    select_slices = tuple(
                        [
                            slice(0, len(subgraph_inputs)),
                            *(0 for _ in subgraph_inputs.shape[1:]),
                        ]
                    )
                    fig, ax = plt.subplots()
                    ax.step(
                        subgraph_inputs[select_slices],
                        reference[select_slices],
                        label="reference",
                        color="blue",
                    )
                    ax.step(
                        subgraph_inputs[select_slices],
                        approximated_no_calib[select_slices],
                        label="rounded - no calib",
                        linestyle="dotted",
                        color="red",
                    )
                    ax.step(
                        subgraph_inputs[select_slices],
                        tlu_ouput[select_slices],
                        label="calibrated",
                        linestyle="dashed",
                        color="green",
                    )
                    ax.legend()
                    ax.set_title(
                        f"{best_a[select_slices]=} {best_b[select_slices]=} {tlu_index=} {select_slices=}"
                    )
                    fig.savefig(f"choice_{tlu_index}.png")
                    plt.close(fig)

                # assert (mse == best_mse).all()  # breaks for now  # breaks for now
                # print("MAX-ABS ERROR: ", np.abs(mse - best_mse).max())
                cycles = nx.recursive_simple_cycles(tlu_subgraph)
                if cycles:
                    raise ValueError()

            print("done with node")
        print(f"{self.exactness=}")

        # for destination in ["debug.png", "debug.dot", "debug.svg"]:
        #     graph.draw(save_to=destination)

        with open("debug.graph", "w", encoding="utf-8") as file:
            file.write(graph.format(show_assigned_bit_widths=True))

        cycles = nx.recursive_simple_cycles(graph.graph)
        if cycles:
            raise ValueError()
        if self.verbose:
            print("TLU optimization done.")
            print("Resulting graph:")
            print(graph.format())
