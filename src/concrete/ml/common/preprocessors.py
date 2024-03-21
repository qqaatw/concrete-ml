"""
Declaration of `TLUOptimizer` graph processor.
"""

from copy import deepcopy
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from concrete.fhe import Exactness, round_bit_pattern
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
        overflow_protection: bool = True,
    ):
        self.rounding_threshold = threshold
        self.exactness = exactness
        self.overflow_protection = overflow_protection

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
                    "overflow_protection": self.overflow_protection,
                    "exactness": self.exactness,
                },
                attributes={
                    "overflow_protection": self.overflow_protection,
                },
            )
            rounding_node.properties["final_lsbs_to_remove"] = lsbs_to_remove
            rounding_node.properties["resulting_bit_width"] = self.rounding_threshold
            rounding_node.properties["overflow_protection"] = self.overflow_protection
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

        pred_results = [deepcopy(node_results[pred]) for pred in graph.ordered_preds_of(node)]
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

        pred_results = [deepcopy(node_results[pred]) for pred in graph.ordered_preds_of(node)]
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

def merge_tlu_constant_shapes(constant_shapes, expected_shape):
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
        reduce_axes = tuple()
    return shape_, reduce_axes


# TODO: WE SHOULD NEVER DO MULT LAST as a small offset will get much bigger

# TODO: serialize the output of the TLU optimizer to be able to re-use the results without having to re-do the whole search
# i.e. some cache system -> basically if the subgraph is the same and the bounds too
# TODO: figure out why we get fhe bool ops in the test
class TLUExhaustiveSearchOptimizer(GraphProcessor):
    def __init__(
        self,
    ):
        raise NotImplementedError("")


class TLUGradientBasedOptimizer(GraphProcessor):
    def __init__(
        self,
    ):
        raise NotImplementedError("")

    def apply(self, graph: Graph):
        # If dataset we need to compute the input/output of all tlu nodes
        self.dataset = None
        # Andrei: When is dataset set ??
        if self.dataset is not None:
            all_nodes_results, all_node_inputs = vectorized_graph_eval_all(graph, *self.dataset)
        else:
            all_nodes_results, all_node_inputs = None, None
        del all_node_inputs
        del all_nodes_results


def add_rounding_node(
    a_node: Node,
    lsbs_to_remove: int,
    graph: nx.DiGraph,
    rounding_function=round_bit_pattern,
    exactness=Exactness.EXACT,
    overflow_protection: bool = False,
):
    if lsbs_to_remove <= 0:
        return a_node

    assert isinstance(a_node.output.dtype, Integer)
    rounding_kwargs = {
        "lsbs_to_remove": lsbs_to_remove,
    }
    attributes = {}
    if rounding_function.__name__ == "round_bit_pattern":
        # These kwargs are not supported atm
        rounding_kwargs["exactness"] = exactness
        rounding_kwargs["overflow_protection"] = overflow_protection
        attributes = {
            "overflow_protection": overflow_protection,
        }
    rounding_node = Node.generic(
        name=rounding_function.__name__,
        inputs=[deepcopy(a_node.output)],
        output=deepcopy(a_node.output),
        operation=rounding_function,
        kwargs=rounding_kwargs,
        attributes=attributes,
    )
    rounding_node.properties["final_lsbs_to_remove"] = lsbs_to_remove
    rounding_node.properties["resulting_bit_width"] = a_node.output.dtype.bit_width - lsbs_to_remove
    rounding_node.properties["overflow_detected"] = False
    rounding_node.properties["original_input_bit_width"] = a_node.output.dtype.bit_width
    if rounding_function.__name__ == "round_bit_pattern":
        rounding_node.properties["overflow_protection"] = overflow_protection
        rounding_node.properties["exactness"] = exactness

    rounding_node.bounds = a_node.bounds  # Might be over/under-estimated

    # Add edge between node and rounding node
    graph.add_edge(a_node, rounding_node, input_idx=0)

    # Replace a -> o_i by rounding_node -> o_i
    edges = list(graph.out_edges(a_node))
    for in_node, out_node in edges:
        if out_node == rounding_node:
            continue
        # We should preserve the input_idx
        edge_data = dict(graph.get_edge_data(in_node, out_node))
        graph.remove_edge(in_node, out_node)
        input_idx: int = edge_data[0]["input_idx"]
        graph.add_edge(rounding_node, out_node, input_idx=input_idx)
    return rounding_node


def add_leveled_op_with_cst(
    a_node: Node,
    b: np.ndarray,
    function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    graph: nx.DiGraph,
):
    assert isinstance(a_node, Node)
    assert isinstance(b, np.ndarray)
    assert isinstance(a_node.output.dtype, (Float, Integer))

    constant_node = Node.constant(b)

    # Handle dtype
    if b.dtype == np.float64:
        constant_dtype = Float(64)
        result_dtype = Float(64)
        bounds = None
    elif b.dtype == np.int64:
        # Compute bounds
        assert (
            isinstance(a_node.bounds, tuple) and len(a_node.bounds) == 2
        ), f"{a_node.bounds=} from {a_node.properties['name']=} is not a Tuple or doesn't have length 2"
        some_inputs = np.zeros((2,) + a_node.output.shape)
        some_inputs[0] = a_node.bounds[0]  # min
        some_inputs[1] = a_node.bounds[1]  # max
        results = function(np.array(some_inputs), b[np.newaxis, ...])
        bounds = (
            results.min(),
            results.max(),
        )
        constant_dtype = Integer.that_can_represent(b)
        result_dtype = Integer.that_can_represent(results.astype(np.int64))
    else:
        raise ValueError(f"Constant {b} should be of dtype np.int64 or np.float64, not {b.dtype}")

    constant_node.output = ValueDescription(
        dtype=constant_dtype,
        is_encrypted=False,
        shape=b.shape,
    )

    assert isinstance(constant_node.output.dtype, (Float, Integer))

    # Create op node
    new_node = Node.generic(
        name=function.__name__,
        inputs=[
            deepcopy(a_node.output),
            deepcopy(constant_node.output),
        ],
        output=ValueDescription(
            dtype=result_dtype,
            shape=a_node.output.shape,
            is_encrypted=a_node.output.is_encrypted,
        ),
        operation=function,
    )
    new_node.bounds = bounds

    # Create new edges
    graph.add_edge(a_node, new_node, input_idx=0)
    graph.add_edge(constant_node, new_node, input_idx=1)

    # Replace a -> o_i by new_node -> o_i
    edges = list(graph.out_edges(a_node))
    for in_node, out_node in edges:
        if out_node == new_node:
            continue
        # We should preserve the input_idx
        edge_data = dict(graph.get_edge_data(in_node, out_node))
        graph.remove_edge(in_node, out_node)
        input_idx: int = edge_data[0]["input_idx"]
        graph.add_edge(new_node, out_node, input_idx=input_idx)

    return new_node


def delta_optimize(
    subgraph_inputs: np.ndarray,
    reference: np.ndarray,
    shape_: Tuple[int, ...],
    bounds: Tuple[int, int],
    rounding_function=round_bit_pattern,
):
    if rounding_function.__name__ != "round_bit_pattern":
        raise ValueError()

    # TODO: support np.array bounds
    # TODO: implement np.array bounds in CP
    # Match the GCD of the steps k*2**n
    x_min, x_max = bounds

    # Initialize a and b such that no changes are done
    best_a = np.ones((1,) + shape_[1:], dtype=np.int64)
    best_b = np.zeros((1,) + shape_[1:], dtype=np.int64)

    n_elems = reference.shape[0]
    ref_diff = np.diff(reference, axis=0).astype(bool)
    n_jumps = (ref_diff > 0).sum()

    # Compute mask of values for which there is a change
    change_mask = np.concatenate(
        [
            np.zeros(reference[:1].shape).astype(bool),
            ref_diff.astype(bool),
        ]
    ).astype(bool)

    # Some accumulators
    deltas = np.zeros(shape_[1:], dtype=np.int64)
    rounding_thresholds = np.zeros(shape_[1:], dtype=np.int64)
    n_rounds = np.ones(shape_[1:], dtype=np.int64)

    # Apply on all elements
    # TODO: vectorize this
    deltas_per_tlu = []
    for indexes in tqdm(product(*[range(elt) for elt in shape_[1:]])):
        selection = tuple([slice(0, n_elems), *indexes])
        best_indexes = tuple([0, *indexes])
        steps_indexes = subgraph_inputs[selection][change_mask[selection]]

        print(f"{steps_indexes=}")

        if len(steps_indexes) == 0:
            print("Constant TLU")
            # The function is constant so nothing to do here
            # We can just round to one
            n_round = 1
            n_rounds[indexes] = int(n_round)
            best_b[best_indexes] = int(0)
            best_a[best_indexes] = int(1)
            continue

        th_0 = steps_indexes[0]  # First x such f(x-1) != f(x)
        delta_axis = np.diff(steps_indexes, axis=0)  # all step sizes

        if len(delta_axis) == 0:
            # Single jump
            # We can just offset by the threshold and round to 1-bit
            print(f"Single jump: {x_min=}, {th_0}, {x_max=}")
            n_round = 1
            n_rounds[indexes] = int(n_round)
            best_b[best_indexes] = int(th_0)
            best_a[best_indexes] = int(1)
            # Map th_0 to 0 then it's just about extracting the sign
            continue

        # Get the common delta between all steps
        deltas_per_tlu.append(np.unique(delta_axis))
        delta = np.bincount(delta_axis).argmax()
        deltas[indexes] = delta

        if delta <= 1:
            n_round = 1
            n_rounds[indexes] = int(n_round)
            best_b[best_indexes] = int(th_0)
            best_a[best_indexes] = 1
            # Map th_0 to 0 then no rounding is needed
            continue

        BIT_WIDTH_ESTIM_FUNC = np.ceil

        rounding_threshold = BIT_WIDTH_ESTIM_FUNC(np.log2((x_max - x_min) / delta)).astype(np.int64)
        rounding_thresholds[indexes] = rounding_threshold

        # Find new limits such that we have smallest bounds that include actual bounds and
        # can be expressed as th_0 + (k * delta)
        x_delta_min = int(th_0 - np.ceil((th_0 - x_min) / delta) * delta)
        x_delta_max = int(th_0 + np.ceil((x_max - th_0) / delta) * delta)
        print(f"{x_delta_min=}, {x_min=}, {x_max=}, {x_delta_max=} {delta=} {th_0=}")
        assert (x_delta_max - x_delta_min) % delta == 0

        # Number of elements in the new range for the given step size
        n_parts = ((x_delta_max - x_delta_min) / delta) + 1
        n_round = int(BIT_WIDTH_ESTIM_FUNC(np.log2(n_parts)).astype(np.int64))

        exceed = ((2**n_round)) - n_parts

        left_bound_add = np.ceil(exceed / 2).astype(np.int64)
        right_bound_add = np.floor(exceed / 2).astype(np.int64)
        assert left_bound_add + right_bound_add == exceed

        # Update bounds to have exactly 2**n_round values in the range
        x_delta_min -= left_bound_add * delta
        x_delta_max += right_bound_add * delta

        assert (x_delta_max - x_delta_min) % delta == 0

        n_parts = ((x_delta_max - x_delta_min) / delta) + 1
        n_bits_before = np.log2(n_parts)
        assert n_bits_before % 1 == 0

        # Arbitrarily high number
        n_bits_after = 23

        # b_prime = ((x_delta_max - x_delta_min) / 2) + (delta / 2) + 1
        # a_prime = (2 ** (n_bits_after)) / (delta * ((2**n_bits_before - 1)))

        a_prime = (2 ** (n_bits_after) - 1) / (x_delta_max - x_delta_min)
        b_prime = (x_delta_min * a_prime) + (2**(n_bits_after - 1))

        # Notebook implementation
        n_round = int(np.around(np.log2((x_max - x_min) / delta)))

        # Find new limits such that we have smallest bounds that include actual bounds as t_0 + (k * step_size)
        x_delta_min = th_0 - ((th_0 - x_min) // delta) * delta - bool(x_min % delta) * delta
        x_delta_max = th_0 + ((x_max - th_0) // delta) * delta + bool(x_max % delta) * delta

        # Number of elements in the new range for the given step size
        n_parts = (x_delta_max - x_delta_min) / delta
        n_round = BIT_WIDTH_ESTIM_FUNC(np.log2(n_parts)).astype(np.int64)
        # assert n_round <= rounding_threshold, f"{n_round=} > {rounding_threshold=}"

        exceed = ((2**n_round)) - n_parts
        left_bound_add = np.ceil(exceed / 2).astype(np.int64)
        right_bound_add = np.floor(exceed / 2).astype(np.int64)
        assert left_bound_add + right_bound_add == exceed
        x_delta_min -= left_bound_add * delta
        x_delta_max += right_bound_add * delta

        n = 23
        a_prime = np.floor(((2**n) - 1) / (x_delta_max - x_delta_min)).astype(np.int64)
        b_prime = ((x_delta_min * a_prime) + (2 ** (n - 1))).astype(np.int64)
        new_min, new_max = (x_delta_min * a_prime) - b_prime, (x_delta_max * a_prime) - b_prime
        assert new_min == -(2 ** (n - 1))
        assert new_max <= ((2 ** (n - 1)) - 1)

        best_a[best_indexes] = int(a_prime)
        best_b[best_indexes] = int(b_prime)
        n_rounds[indexes] = int(n_round)

    # TODO: compute lsbs to remove based on bounds and best_a, best_b
    # The issue is that removing this half will add 1-bit (i.e. if n=23 we'll be on 24 bits)
    acc_bit_with = Integer.that_can_represent(
        (np.array([x_min, x_max]) * best_a) - best_b
    ).bit_width
    n_round = int(n_rounds.max())
    lsbs_to_remove = int(acc_bit_with - n_round)

    # DEBUG: HALF TRICK
    # if lsbs_to_remove > 0:
    #     half = 1 << lsbs_to_remove
    #     best_b += half

    # TODO: This half should probably also be taken into account in the compuation above
    # Because it will add a bit to the accumulator

    # breakpoint() # Check that we are on the correct bit with and equal/close to the bounds
    return n_round, best_a, best_b, deltas_per_tlu, n_jumps


# TODO: extract optimization into another
class TLUDeltaBasedOptimizer(GraphProcessor):
    """
    TLUDeltaBasedOptimizer graph processor, to add approximate rounding and scaling before/in TLUs if desired.
    """
    def __init__(
        self,
        verbose: bool = True,
        exactness: Exactness = Exactness.APPROXIMATE,
        overflow_protection: bool = True,
        internal_bit_width_target=23,
    ):
        self.verbose = verbose
        self.exactness = exactness
        self.overflow_protection = overflow_protection
        self.rounding_function = round_bit_pattern
        self.internal_bit_width_target = internal_bit_width_target
        # Store per PBS statistics
        self._statistics: Dict[int, Dict[str, Union[int, np.ndarray]]] = {}


    @staticmethod
    def extract_tlu_input_bounds(variable_input_node):
        # Extract TLU input bounds
        if variable_input_node.bounds is not None:
            min_bound, max_bound = variable_input_node.bounds
            # For some reason sometimes the bounds booleans
            if isinstance(min_bound, bool):
                min_bound = int(min_bound)
            if isinstance(max_bound, bool):
                max_bound = int(max_bound)
            if isinstance(min_bound, np.bool_):
                min_bound = min_bound.astype(np.int64)
            if isinstance(max_bound, np.bool_):
                max_bound = max_bound.astype(np.int64)
            assert isinstance(min_bound, int) or (
                isinstance(min_bound, np.ScalarType) and min_bound.dtype == np.int64
            ), f"{type(min_bound)=}"
            assert isinstance(max_bound, int) or (
                isinstance(max_bound, np.ScalarType) and max_bound.dtype == np.int64
            ), f"{type(max_bound)=}"
        else:
            raise ValueError("Bounds not found")

        return min_bound, max_bound

    @staticmethod
    def compute_tlu_output_shapes(tlu_subgraph, expected_shape):
        constant_shapes = list()
        orig_constant_shapes = list()
        for elt in tlu_subgraph.graph.nodes:
            assert isinstance(elt, Node)
            if isinstance(elt.evaluator, ConstantEvaluator):
                constant_shape = list(elt.output.shape)
                value = elt.evaluator.properties["constant"]
                if constant_shape:
                    orig_constant_shapes.append(tuple(constant_shape))
                    for axis in range(len(value.shape)):
                        unique_values_per_axis = np.unique(value, axis=axis)
                        if unique_values_per_axis.shape[axis] == 1:
                            constant_shape[axis] = 1

                    constant_shapes.append(tuple(constant_shape))

        # This shape includes constant axes folding and only reduces broadcasted axes
        shape_, reduce_axes = merge_tlu_constant_shapes(constant_shapes, expected_shape)

        # This shape excludes constant axes
        orig_shape_, _ = merge_tlu_constant_shapes(orig_constant_shapes, expected_shape)

        return shape_, reduce_axes, orig_shape_

    @staticmethod
    def get_subgraph_input(subgraph: Graph) -> Node:
        input_node = None
        for node in subgraph.graph:
            if "name" in node.properties and node.properties["name"] == "input":
                assert input_node is None, "More than one astype float node detected"
                input_node = node
        if input_node is None:
            # Try falling back to the first node if it's astype
            # TODO: figure out how to insert right in the begining of the graph
            for first_node in nx.topological_sort(subgraph.graph):
                # Only constants allowed
                if "constant" in first_node.properties:
                    continue
                elif first_node.properties["name"] == "astype":
                    return first_node
            breakpoint()
            raise ValueError(f"Couldn't detect input node in:\n{subgraph.format()}")
        return input_node

    @staticmethod
    def get_tlu_node_subgraph_input_node(graph, tlu_node):
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
        if not isinstance(variable_input_node.output.dtype, Integer):
            raise ValueError("TLU node got input dtype that isn't integer")
        return variable_input_node

    @staticmethod
    def make_subgraph_input_tensor(min_bound, max_bound, orig_shape_, expected_shape):
        subgraph_inputs = np.array(list(range(int(min_bound), int(max_bound) + 1)))
        subgraph_input_shape = tuple([len(subgraph_inputs), *orig_shape_[1:]])

        if len(expected_shape) > 1:
            assert expected_shape[0] == 1
            subgraph_inputs = np.tile(
                subgraph_inputs[
                    tuple(
                        [
                            slice(0, len(subgraph_inputs), 1),
                            *[np.newaxis for _ in range(len(expected_shape) - 1)],
                        ]
                    )
                ],
                expected_shape,
            )

            subgraph_inputs = subgraph_inputs[
                tuple([slice(0, elt, 1) for elt in subgraph_input_shape])
            ]

        return subgraph_inputs

    def apply(self, graph: Graph):
        tlu_nodes = graph.query_nodes(
            custom_filter=is_node_tlu,  # TLU filter function
            ordered=True,  # Not striclty necessary but easier to debug
        )

        for tlu_index, tlu_node in enumerate(tlu_nodes):
            # On each tlu we do:
            # 1. Optimize a and b for the subgraph
            # 2. Insert a and b in the graph
            # 3. Insert rounding to the graph
            # 4. Insert a and b in the subgraph

            # Get TLU sub-graph
            if "subgraph" not in tlu_node.evaluator.properties["kwargs"]:
                continue
            tlu_subgraph: Graph = tlu_node.evaluator.properties["kwargs"]["subgraph"]

            if self.verbose:
                print(f"TLU-{tlu_index} before optimization")
                print("#" * 20)
                print(tlu_subgraph.format())
                print("#" * 20)

            # TLU input node (multiple inputs using the same subgraph is not supported)
            variable_input_node = self.get_tlu_node_subgraph_input_node(graph, tlu_node)

            min_bound, max_bound = self.extract_tlu_input_bounds(variable_input_node)

            # Create input with proper shape on the bounds for optimization
            expected_shape = variable_input_node.output.shape

            shape_, reduce_axes, orig_shape_ = self.compute_tlu_output_shapes(tlu_subgraph, expected_shape)

            # Create an input which the full input range
            subgraph_inputs = self.make_subgraph_input_tensor(
                min_bound, max_bound, orig_shape_, expected_shape
            )

            # Compute TLU output on bounds without rounding or calibration for reference
            sorted_nodes = list(nx.topological_sort(tlu_subgraph.graph))
            reference = vectorized_graph_eval(
                tlu_subgraph, subgraph_inputs, sorted_nodes=sorted_nodes
            )
            assert isinstance(reference, np.ndarray)
            reference = reference.astype(np.int64)

            n_round, best_a, best_b, deltas_per_tlu, n_jumps = delta_optimize(
                subgraph_inputs,
                reference,
                shape_,
                (int(min_bound), int(max_bound)),
            )

            # For testing purposes we had some properties
            tlu_node.properties["attributes"]["deltas_per_tlu"] = deltas_per_tlu
            tlu_node.properties["attributes"]["opt_round_a"] = best_a
            tlu_node.properties["attributes"]["opt_round_b"] = best_b

            if all([np.all(deltas == 1) for deltas in deltas_per_tlu]):
                continue

            # We need to make sure that we have the correct shape when adding constants in the graph
            # As Concrete Python doesn't handle broadcasting at the graph level
            assert isinstance(variable_input_node.output, ValueDescription)
            best_a = np.broadcast_to(best_a, shape=(1,) + variable_input_node.output.shape[1:])
            best_b = np.broadcast_to(best_b, shape=(1,) + variable_input_node.output.shape[1:])

            # Main graph modification
            # Add scaling and rounding to the main graph
            previous_node = variable_input_node
            # Multiply by a
            previous_node = add_leveled_op_with_cst(
                previous_node, best_a.astype(np.int64), multiply, graph.graph
            )
            # Subtract b
            previous_node = add_leveled_op_with_cst(
                previous_node, best_b.astype(np.int64), subtract, graph.graph
            )
            # Round by n_round
            assert isinstance(previous_node.output.dtype, Integer)
            lsbs_to_remove = int(previous_node.output.dtype.bit_width - n_round)
            previous_node = add_rounding_node(
                previous_node,
                lsbs_to_remove,
                graph.graph,
                rounding_function=self.rounding_function,
                exactness=self.exactness,
                overflow_protection=self.overflow_protection,
            )

            # DEBUG: sanity check
            approx_subgraph_inputs = (
                self.rounding_function(
                    (subgraph_inputs * best_a) - best_b, lsbs_to_remove=lsbs_to_remove
                ).astype(np.float64)
                + best_b.astype(np.float64)
            ) / best_a.astype(np.float64)
            approx_reference = vectorized_graph_eval(
                tlu_subgraph, approx_subgraph_inputs, sorted_nodes=sorted_nodes
            )
            if (((reference - approx_reference) > 0).sum(axis=0) >= n_jumps).any():
                pass
                # breakpoint()

            # Store some statistics for testing/debugging in the object itself
            self._statistics[tlu_index] = {}
            self._statistics[tlu_index]["msbs_to_keep"] = n_round
            self._statistics[tlu_index]["lsbs_to_remove"] = lsbs_to_remove
            self._statistics[tlu_index]["a"] = best_a
            self._statistics[tlu_index]["b"] = best_b

            # Sub-graph modification
            previous_node = self.get_subgraph_input(tlu_subgraph)
            # Add b
            previous_node = add_leveled_op_with_cst(
                previous_node, best_b.astype(np.float64), add, graph=tlu_subgraph.graph
            )
            # Divide by a
            previous_node = add_leveled_op_with_cst(
                previous_node, best_a.astype(np.float64), divide, graph=tlu_subgraph.graph
            )

            # ##################################################
            # ################## DEBUG #########################
            # ##################################################
            # TODO: DEBUGGING SANITY CHECK WHERE WE VALIDATE THAT THE BEST MSE IS INDEED ACHIEVED
            # WITH BEST_A and BEST_B using the subgraph to do the computation
            print("Doing sanity check")
            sanity_check = False
            if sanity_check:
                x = (subgraph_inputs + best_b) * best_a
                if lsbs_to_remove > 0:
                    x = self.rounding_function(
                        x=x,
                        lsbs_to_remove=lsbs_to_remove,
                        overflow_protection=self.overflow_protection,
                        exactness=self.exactness,
                    )
                tlu_ouput = vectorized_graph_eval(tlu_subgraph, x)
                assert isinstance(tlu_ouput, np.ndarray)
                mse = ((reference - tlu_ouput) ** 2).mean(axis=reduce_axes, keepdims=True)
                print(f"{mse=}")

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
            # ##################################################
            # ################## DEBUG #########################
            # ##################################################

        if self.verbose:
            print("TLU optimization done.")
            print("Resulting graph:")
            print(graph.format())
