"""
Declaration of `TLUOptimizer` graph processor.
"""

import warnings
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
                tlu_subgraph: Graph = tlu_node.evaluator.properties["kwargs"][
                    "subgraph"
                ]
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

def merge_tlu_constant_shapes(constant_shapes):
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
    return shape_, reduce_axes

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
        self._rounding_bits: Dict[int, Dict[str, int]] = {}


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
    def compute_tlu_output_shapes(tlu_subgraph):
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
        shape_, reduce_axes = merge_tlu_constant_shapes(constant_shapes)

        # This shape excludes constant axes
        orig_shape_, _ = merge_tlu_constant_shapes(orig_constant_shapes)

        return shape_, reduce_axes, orig_shape_

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
        return variable_input_node

    @staticmethod
    def make_subgraph_input_tensor(min_bound, max_bound, orig_shape_, expected_shape):
        subgraph_inputs = np.array(
            list(range(int(min_bound), int(max_bound) + 1))
        )
        subgraph_input_shape = tuple([len(subgraph_inputs), *orig_shape_[1:]])

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

        return subgraph_inputs

    def apply(self, graph: Graph):
        if self.rounding_threshold is None:
            return

        # If dataset we need to compute the input/output of all tlu nodes
        # Andrei: When is dataset set ?? 
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

            if not "subgraph" in tlu_node.evaluator.properties["kwargs"]:
                continue
            
            tlu_subgraph: Graph = tlu_node.evaluator.properties["kwargs"]["subgraph"]

            if self.verbose:
                print(tlu_subgraph.format())

            variable_input_node = self.get_tlu_node_subgraph_input_node(graph, tlu_node)

            # Check subgraph input datatype, must be Integer
            variable_input_dtype = variable_input_node.output.dtype
            if not isinstance(variable_input_dtype, Integer):
                raise ValueError("TLU node got input dtype that isn't integer")

            # ---------------------------------------------
            # Search a and b that reduce the MSE of the TLU
            # ---------------------------------------------

            min_bound, max_bound = self.extract_tlu_input_bounds(variable_input_node)

            # Create input with proper shape on the bounds for optimization
            expected_shape = variable_input_node.output.shape

            shape_, reduce_axes, orig_shape_ = self.compute_tlu_output_shapes(tlu_subgraph)

            # Create an input which the full input range 
            subgraph_inputs = self.make_subgraph_input_tensor(min_bound, max_bound, orig_shape_, expected_shape)

            # Compute TLU output on bounds without rounding or calibration for reference
            sorted_nodes = list(nx.topological_sort(tlu_subgraph.graph))
            reference = vectorized_graph_eval(
                tlu_subgraph, subgraph_inputs, sorted_nodes=sorted_nodes
            )
            assert isinstance(reference, np.ndarray)            
            reference = reference.astype(np.int64)

            # # Compute with rounding but without calibration
            # accumulator_bit_width = Integer.that_can_represent(
            #     subgraph_inputs
            # ).bit_width
            # lsbs_to_remove = accumulator_bit_width - n_round
            # assert isinstance(subgraph_inputs, np.ndarray)
            # if lsbs_to_remove > 0:
            #     x = round_bit_pattern(subgraph_inputs, lsbs_to_remove=lsbs_to_remove)
            # else:
            #     x = subgraph_inputs
            # assert isinstance(x, np.ndarray)
            # approximated_no_calib = vectorized_graph_eval(
            #     tlu_subgraph,
            #     x,
            #     sorted_nodes=sorted_nodes,
            # )
            # assert isinstance(approximated_no_calib, np.ndarray)
            # approximated_no_calib = approximated_no_calib.astype(np.int64)
            # assert isinstance(reference, np.ndarray)
            # assert isinstance(approximated_no_calib, np.ndarray)
            # mse_no_calib = ((reference - approximated_no_calib) ** 2).mean(
            #     axis=reduce_axes, keepdims=True
            # )

            # optimization = "exact"
            # if optimization == "exact":
            # Match the GCD of the steps k*2**n

            # Initialize a and b such that no changes are done
            best_a = (
                np.ones((1,) + shape_[1:], dtype=np.int64)
            ).astype(np.int64)

            best_b = (
                np.zeros((1,) + shape_[1:], dtype=np.int64)
            ).astype(np.int64)

            n_elems = reference.shape[0]
            # Data-bounds (TODO: should be per-axis)
            x_min, x_max = (
                min_bound,
                max_bound,
            )  

            # Compute mask of values for which there is a change
            change_mask = np.concatenate(
                [
                    np.zeros(reference[:1].shape).astype(bool),
                    np.diff(reference, axis=0).astype(bool),
                ]
            ).astype(bool)

            # Some accumulators
            deltas = np.zeros(shape_[1:], dtype=np.int64)
            rounding_thresholds = np.zeros(shape_[1:], dtype=np.int64)
            n_rounds = np.ones(reference.shape[1:], dtype=np.int64)

            # Apply on all elements
            # TODO: vectorize this
            deltas_per_tlu = []

            for indexes in tqdm(
                product(*[range(elt) for elt in shape_[1:]])
            ):
                selection = tuple([slice(0, n_elems), *indexes])
                best_indexes = tuple([0, *indexes])
                steps_indexes = subgraph_inputs[selection][change_mask[selection]]

                if len(steps_indexes) == 0:
                    # The function is constant
                    continue

                th_0 = steps_indexes[0]  # First x such f(x-1) != f(x)
                delta_axis = np.diff(steps_indexes, axis=0)  # all step sizes

                if len(delta_axis) == 0:
                    # Single jump
                    # We can just offset by the threshold and round to 1-bit
                    print(f"{x_min=}, {th_0}, {x_max=}")
                    best_b[best_indexes] = th_0
                    best_a[best_indexes] = 1.0
                    # Map th_0 to 0 then it's just about extracting the sign
                    continue

                # Get the common delta between all steps
                deltas_per_tlu.append(np.unique(delta_axis))
                delta = np.bincount(delta_axis).argmax()
                deltas[indexes] = delta

                BIT_WIDTH_ESTIM_FUNC = np.ceil # np.around

                rounding_threshold = BIT_WIDTH_ESTIM_FUNC(
                    np.log2((x_max - x_min) / delta)
                ).astype(np.int64)
                rounding_thresholds[indexes] = rounding_threshold

                # Find new limits such that we have smallest bounds that include actual bounds and
                # can be expressed as th_0 + (k * delta)
                x_delta_min = int(
                    th_0
                    - np.ceil((th_0 - x_min) / delta) * delta
                )
                x_delta_max = int(
                    th_0
                    + np.ceil((x_max - th_0) / delta) * delta
                )
                print(f"{x_delta_min=}, {x_min=}, {x_max=}, {x_delta_max=} {delta=} {th_0=}")
                assert (x_delta_max - x_delta_min) % delta == 0

                # TODO: FIX THIS n-round can be different for each axis
                # Number of elements in the new range for the given step size
                n_parts = ((x_delta_max - x_delta_min) / delta) + 1
                n_round = int(BIT_WIDTH_ESTIM_FUNC(np.log2(n_parts)).astype(np.int64))

                # TODO: FIX THIS
                # if n_round > rounding_threshold:
                #     warnings.warn(f"{n_round=} > {rounding_threshold=}")
                # print(f"{n_parts=}, {n_round=} {rounding_threshold=}")

                # TODO: DEBUG: make sure there isn't a -1 around here
                exceed = ((2**n_round)) - n_parts

                left_bound_add = np.ceil(exceed / 2).astype(np.int64)
                right_bound_add = np.floor(exceed / 2).astype(np.int64)
                assert left_bound_add + right_bound_add == exceed

                # Update bounds to have exactly 2**n_round values in the range
                x_delta_min -= left_bound_add * delta
                x_delta_max += right_bound_add * delta

                print(f"{x_delta_min=}, {x_min=}, {x_max=}, {x_delta_max=}, {delta=} {th_0=}")
                assert (x_delta_max - x_delta_min) % delta == 0

                n_parts = ((x_delta_max - x_delta_min) / delta) + 1
                n_bits_before = np.log2(n_parts)
                assert n_bits_before % 1 == 0

                # Arbitrarily high number
                n_bits_after = 23

                # TODO: DEBUG
                # b_prime = x_delta_min - (-(2 ** (n_bits_before - 1)))
                b_prime = ((x_delta_max - x_delta_min) / 2) + (delta / 2) + 1
                a_prime = (2 ** (n_bits_after)) / (delta * ((2**n_bits_before - 1)))

                # NOTEBOOK IMPLEMENTATION
                n_round = int(np.around(np.log2((x_max - x_min)/delta)))
                # Find new limits such that we have smallest bounds that include actual bounds as t_0 + (k * step_size)
                x_delta_min = th_0 - ((th_0 - x_min) // delta ) * delta - bool(x_min % delta)*delta
                x_delta_max = th_0 + ((x_max - th_0) // delta ) * delta + bool(x_max % delta)*delta
                 # Number of elements in the new range for the given step size
                n_parts = ((x_delta_max - x_delta_min) / delta)
                n_round = BIT_WIDTH_ESTIM_FUNC(np.log2(n_parts)).astype(np.int64)
                assert n_round <= rounding_threshold, f"{n_round=} > {rounding_threshold=}"
                print(f"{n_parts=}, {n_round=}")
                
                exceed = ((2**n_round)) - n_parts
                left_bound_add = np.ceil(exceed/2).astype(np.int64)
                right_bound_add = np.floor(exceed/2).astype(np.int64)
                assert left_bound_add + right_bound_add == exceed
                x_delta_min -= left_bound_add * delta
                x_delta_max += right_bound_add * delta
                middle = (x_delta_max - x_delta_min)/2
                middle = np.median(np.arange(x_delta_min, x_delta_max+1, delta))
                # Find the proper n
                n = 23
                mult_first = False
                if mult_first:
                    a_prime = ((2**n) - delta)/(x_delta_max - x_delta_min)

                    b_prime = -2**(n-1) - (x_delta_min * ((2**n - 1)/(x_delta_max - x_delta_min)))
                    
                    print(f"{a_prime=}, {b_prime=}")
                    
                    a_prime = np.floor(a_prime).astype(np.int64)
                    b_prime = np.floor(b_prime).astype(np.int64)
                else:
                    a_prime = (2**n - 1) / (x_delta_max - x_delta_min)
                    a_prime = np.floor(a_prime).astype(np.int64)

                    # b_prime = (2**(n-1)) + (x_delta_min * a_prime)
                    b_prime = middle
                    if self.rounding_function == round_bit_pattern:
                        b_prime += (delta/2)
                    else:
                        b_prime += 0
                    print(f"{a_prime=}, {b_prime=}")

                    a_prime = np.floor(a_prime).astype(np.int64)
                    b_prime = np.floor(b_prime).astype(np.int64)

                print((x_delta_min - b_prime) * a_prime, (x_delta_max - b_prime) * a_prime)

                # breakpoint()

                best_a[best_indexes] = a_prime
                best_b[best_indexes] = b_prime
                n_rounds[indexes] = n_round

            n_round = int(n_rounds.max())
            print(f"ROUNDING TO {n_round}")

            tlu_node.properties["attributes"]["deltas_per_tlu"] = deltas_per_tlu
            tlu_node.properties['attributes']['opt_round_a'] = best_a
            tlu_node.properties['attributes']['opt_round_b'] = best_b

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
            print(f"DEBUG: {best_a=}, {best_b=}")
            assert (best_a >= 1.0).all()
            print(
                f"DEBUG: {np.ceil(np.log2(np.abs(best_a).max()))=}, {np.ceil(np.log2(max(np.abs(best_b).max(), 1)))=}, "
            )

            # ------------------------------------------
            # Add scaling and rounding to the main graph
            # ------------------------------------------

            # Subtract b
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
            substract_bounds = (min_bound - best_b).min(), (max_bound- best_b).max()
            result_dtype = Integer.that_can_represent(list(substract_bounds))
            # X - b
            substract_node = Node.generic(
                name="subtract",
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
            substract_node.bounds = substract_bounds

            # Multiply by a
            previous_node = substract_node
            a_constant_node = Node.constant(best_a)
            a_constant_node.output = deepcopy(previous_node.output)
            a_constant_node.output.is_encrypted = False
            a_constant_node.output.dtype.bit_width = (
                previous_node.output.dtype.bit_width
            )
            a_constant_node.properties["bit_width"] = previous_node.output.dtype.bit_width
            mul_bounds = (
                ((min_bound - best_b) * best_a).min(),
                ((max_bound - best_b) * best_a).max(),
            )
            result_dtype = Integer.that_can_represent(list(mul_bounds))
            # (X - b) * a
            mul_node = Node.generic(
                name="multiply",
                inputs=[deepcopy(substract_node.output), deepcopy(a_constant_node.output)],
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
            nx_graph.add_edge(variable_input_node, substract_node, input_idx=0)
            nx_graph.add_edge(b_constant_node, substract_node, input_idx=1)
            nx_graph.add_edge(substract_node, mul_node, input_idx=0)
            nx_graph.add_edge(a_constant_node, mul_node, input_idx=1)

            # Add rounding
            assert isinstance(mul_node.output.dtype, Integer)
            lsbs_to_remove = int(mul_node.output.dtype.bit_width - n_round)
            if self.verbose:
                print(
                    f"Removing {lsbs_to_remove} lsbs to {mul_node.output.dtype.bit_width} bits"
                )

            if lsbs_to_remove > 0:
                # Create rounding node
                self._rounding_bits[tlu_index] = {
                    "accumulator_bit_width": mul_node.output.dtype.bit_width,
                    "msbs_to_keep": n_round,
                    "lsbs_to_remove": lsbs_to_remove,
                }
                rounding_node = Node.generic(
                    name=self.rounding_function.__name__,
                    inputs=[deepcopy(mul_node.output)],
                    output=deepcopy(mul_node.output),
                    operation=self.rounding_function,
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
                    n_round
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
                    and node.properties["name"] == "input"
                ):
                    assert (
                        input_node is None
                    ), "More than one astype float node detected"
                    input_node = node

            if input_node is None:
                breakpoint()
                raise ValueError("Couldn't detect astype float node")

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
                    x = self.rounding_function(
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
