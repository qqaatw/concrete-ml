from typing import List

import numpy
import pytest
from concrete.fhe import Compiler, Configuration, Exactness, round_bit_pattern, univariate, DebugArtifacts

from concrete.ml.common.preprocessors import (
    CycleDetector,
    GraphProcessor,
    InsertRounding,
    TLUDeltaBasedOptimizer,
)


def make_step_function(n_thresholds, delta, x_min, x_max):
    thresholds_ = []  # First threshold
    th0 = numpy.random.randint(x_min, x_max)
    for index in range(n_thresholds):
        thresholds_.append(th0 + index * delta)

    thresholds = tuple(thresholds_)

    # Step size function to optimize
    def util(x):
        return sum([numpy.where(x >= float(threshold), 1., 0.) for threshold in thresholds])

    def step_function(x):
        # res = numpy.zeros_like(x, dtype=numpy.float64)
        return univariate(util)(x).astype(numpy.int64)

    def f(x): # (1,)
        return step_function(x.astype(numpy.float64))

    return f

def make_identity_function():
    def f(x):
        return (x * 1.0).astype(numpy.int64)
    return f

def make_random_function(x_min, x_max):
    freq = numpy.random.random(size=(3,))

    def f(x):
        x = x.astype(numpy.float64)
        y = numpy.sin(x * freq[0]) + numpy.sin(x * freq[1]) + numpy.sin(x * freq[2])
        y = y * (x_max - x_min) + x_max
        y = y.astype(numpy.int64)
        return y

    return f

@pytest.mark.parametrize("execution_number", range(7))
@pytest.mark.parametrize("function_name", ["staircase", "identity", "random"])
def test_tlu_optimizer(execution_number: int, function_name: str):
    curr_seed = numpy.random.randint(0, 2**32)
    numpy.random.seed(curr_seed + execution_number)

    # Create function
    if function_name == "staircase":   
        n_bits_from = numpy.random.choice(range(2, 9))
        # 2**n range
        x_min, x_max = -(2**n_bits_from), (2**n_bits_from) - 1

        # Real sub-range
        x_min = numpy.random.randint(x_min, x_max)
        x_max = numpy.random.randint(x_max, x_max + 1)

        # Function thresholds
        delta = 2 ** (n_bits_from // 2) - 1  # Constant step size assumption

        f = make_step_function(execution_number, delta, x_min, x_max)
    elif function_name == "identity":
        n_bits_from = execution_number + 1
        x_min, x_max = -(2**n_bits_from), (2**n_bits_from) - 1

        f = make_identity_function()
    elif function_name == "random":
        n_bits_from = execution_number + 1
        x_min, x_max = -(2**n_bits_from), (2**n_bits_from) - 1
        f = make_random_function(x_min, x_max)
    else:
        assert False, f"Invalid function to test for TLU optimization {function_name}"

    # Function definition bounds
    input_set = numpy.arange(x_min, x_max + 1, 1, dtype=numpy.int64)
    input_set_as_list_of_array = [numpy.array([elt]) for elt in input_set]

    # Optim, Rounding
    tlu_optimizer = TLUDeltaBasedOptimizer(verbose=False, exactness=Exactness.EXACT)
    cycle_detector = CycleDetector()
    additional_pre_processors: List[GraphProcessor] = [tlu_optimizer]
    additional_post_processors: List[GraphProcessor] = [cycle_detector]
    compilation_configuration = Configuration(
        additional_pre_processors=additional_pre_processors,
        additional_post_processors=additional_post_processors,
        
    )
    compiler = Compiler(
        f,
        parameter_encryption_statuses={"x": "encrypted"},
    )
    artifacts = DebugArtifacts(f".debug_artifacts_{execution_number}")
    circuit = compiler.compile(
        input_set_as_list_of_array,
        configuration=compilation_configuration,
        artifacts=artifacts,
    )
    artifacts.export()


    # Find optimized TLUs
    tlu_nodes = circuit.graph.query_nodes(
        custom_filter=lambda node: node.converted_to_table_lookup
        and "opt_round_a" in node.properties["attributes"],
        ordered=True,
    )

    # A single TLU should be present
    assert len(tlu_nodes) == 1

    tlu_node = tlu_nodes[0]
    deltas_tlu = tlu_node.properties["attributes"]["deltas_per_tlu"]

    print(circuit.mlir)

    if tlu_optimizer._statistics:
        rounding_threshold = tlu_optimizer._statistics[0]["msbs_to_keep"]
        lsbs_to_remove = tlu_optimizer._statistics[0]["lsbs_to_remove"]
    else:
        rounding_threshold = None
        lsbs_to_remove = None

    # No-optim, rounding
    rounding = InsertRounding(threshold=rounding_threshold)
    additional_pre_processors: List[GraphProcessor] = [rounding]
    additional_post_processors: List[GraphProcessor] = [cycle_detector]
    compilation_configuration = Configuration(
        additional_pre_processors=additional_pre_processors,
        additional_post_processors=additional_post_processors,
    )
    compiler = Compiler(
        f,
        parameter_encryption_statuses={"x": "encrypted"},
    )
    circuit_no_optim_rounding = compiler.compile(
        input_set_as_list_of_array,
        configuration=compilation_configuration,
    )
    print(circuit_no_optim_rounding.mlir)

    # No-optim, no-rounding
    compiler = Compiler(
        f,
        parameter_encryption_statuses={"x": "encrypted"},
    )
    circuit_no_optim_no_rounding = compiler.compile(
        input_set_as_list_of_array,
    )

    reference = f(input_set)

    if function_name == "identity":
        assert circuit.mlir == circuit_no_optim_no_rounding.mlir
        return

    transformed_input_set = input_set.copy()
    if tlu_optimizer._statistics:
        transformed_input_set = transformed_input_set * tlu_optimizer._statistics[0]["a"]
        transformed_input_set = transformed_input_set - tlu_optimizer._statistics[0]["b"]
        if lsbs_to_remove is not None:
            assert isinstance(lsbs_to_remove, int)
            tlu_optimizer.rounding_function(transformed_input_set, lsbs_to_remove=lsbs_to_remove)
        transformed_input_set = transformed_input_set + tlu_optimizer._statistics[0]["b"]
        transformed_input_set = transformed_input_set / tlu_optimizer._statistics[0]["a"]
    # Apply function
    approx_reference = f(transformed_input_set)

    oldsim_result = numpy.array([circuit.graph(numpy.array([elt])) for elt in input_set])[..., 0]

    simulated = numpy.array([circuit.simulate(numpy.array([elt])) for elt in input_set])[..., 0]

    simulated_no_optim_no_rounding = numpy.array(
        [circuit_no_optim_no_rounding.simulate(numpy.array([elt])) for elt in input_set]
    )[..., 0]
    simulated_no_optim_rounding = numpy.array(
        [circuit_no_optim_rounding.simulate(numpy.array([elt])) for elt in input_set]
    )[..., 0]

    graph_res = numpy.array([circuit.graph(numpy.array([elt])) for elt in input_set])[..., 0]
    graph_res_no_optim_no_rounding = numpy.array(
        [circuit_no_optim_no_rounding.graph(numpy.array([elt])) for elt in input_set]
    )[..., 0]
    graph_res_no_optim_rounding = numpy.array(
        [circuit_no_optim_rounding.graph(numpy.array([elt])) for elt in input_set]
    )[..., 0]

    debug = True
    if debug:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        # Numpy calls
        ax.step(input_set, reference, label="reference", color="blue", alpha=.5)
        ax.step(input_set, approx_reference, label="approx reference", color="purple", linestyle="dotted", alpha=.5)

        # Simulation
        ax.step(input_set, simulated, label="optimized", color="red", linestyle=(0, (5, 5)), alpha=.5)
        ax.step(
            input_set,
            simulated_no_optim_no_rounding,
            label="not optimized, not rounded",
            color="green",
            linestyle=(0, (5, 5)), 
            alpha=.5,
        )
        ax.step(
            input_set,
            simulated_no_optim_rounding,
            label="not optimized, rounded",
            color="yellow",
            linestyle=(0, (5, 5)),
            alpha=.5,
        )

        # Graph call
        ax.step(input_set, graph_res, label="optimized (graph)", color="red", linestyle=(0, (5, 5)), alpha=.5)
        ax.step(
            input_set,
            graph_res_no_optim_no_rounding,
            label="not optimized, not rounded (graph)",
            color="green",
            linestyle=(0, (5, 5)), 
            alpha=.5,
        )

        ax.step(
            input_set,
            graph_res_no_optim_rounding,
            label="not optimized, rounded (graph)",
            color="yellow",
            linestyle=(0, (5, 5)),
            alpha=.5,
        )

        # Some lines for reference
        # x_min
        ax.vlines(
            x=input_set[0],
            ymin=reference.min(),
            ymax=reference.max(),
            linestyle=(0, (1, 1)),
            color="grey",
            label="x_min",
            alpha=.5,
        )
        # x_max
        ax.vlines(
            x=input_set[-1],
            ymin=reference.min(),
            ymax=reference.max(),
            linestyle=(0, (1, 1)),
            color="grey",
            label="x_max",
            alpha=.5,
        )

        # Dump figure
        plt.legend()
        fig.savefig(f"debug_{execution_number}.png")
        plt.close(fig)

        with open(f"debug_{execution_number}.graph", "w", encoding="utf-8") as file:
            file.write(circuit.graph.format())

        with open(f"debug_{execution_number}.mlir", "w", encoding="utf-8") as file:
            file.write(circuit.mlir)

    not_equal = reference != simulated
    if (not_equal.sum() > execution_number) or (approx_reference != simulated).any():
        raise Exception(
            f"TLU Optimizer is not exact: {not_equal.mean()=} = {not_equal.sum()}/{not_equal.size}\n"
            f"{tlu_optimizer._statistics=}\n"
            f"{execution_number=}, {lsbs_to_remove=} {(reference == approx_reference).mean()=}\n{'#'*20}\n"
            f"{(simulated == graph_res).mean()=}, {(simulated_no_optim_no_rounding == graph_res_no_optim_no_rounding).mean()=}, {(simulated_no_optim_rounding == graph_res_no_optim_rounding).mean()=}\n"
            f"{(approx_reference == simulated).mean()=}\n"
            f"{circuit.graph.format()}\n{'#'*20}\n"
            f"{circuit_no_optim_no_rounding.graph.format()}\n"
        )
