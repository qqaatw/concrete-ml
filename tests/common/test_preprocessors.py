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


@pytest.mark.parametrize("execution_number", range(10))
def test_tlu_optimizer(execution_number: int):
    curr_seed = numpy.random.randint(0, 2**32)
    numpy.random.seed(execution_number*100)

    # Create function
    n_bits_from = numpy.random.choice(range(2, 9))
    # 2**n range
    x_min, x_max = -(2**n_bits_from), (2**n_bits_from) - 1

    # Real sub-range
    x_min = numpy.random.randint(x_min, x_max)
    x_max = numpy.random.randint(x_max, x_max + 1)

    # Function thresholds
    delta = 2 ** (n_bits_from // 2) - 1  # Constant step size assumption

    n_thresholds = execution_number
    thresholds_ = []  # First threshold
    th0 = numpy.random.randint(x_min, x_max)
    for index in range(n_thresholds):
        thresholds_.append(th0 + index * delta)

    thresholds = tuple(thresholds_)

    print(thresholds)

    # Function definition bounds
    input_set = numpy.arange(x_min, x_max + 1, 1, dtype=numpy.int64)
    input_set_as_list_of_array = [numpy.array([elt]) for elt in input_set]

    # Step size function to optimize
    def step_function(x):
        res = numpy.zeros_like(x, dtype=numpy.float64)
        for threshold in thresholds:
            res = res + x >= float(threshold)
        return res.astype(numpy.int64)

    def f(x):
        return step_function(x.astype(numpy.float64))

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

    # No-optim, no-rounding
    compiler = Compiler(
        f,
        parameter_encryption_statuses={"x": "encrypted"},
    )
    circuit_no_optim_no_rounding = compiler.compile(
        input_set_as_list_of_array,
    )

    reference = f(input_set)

    transformed_input_set = input_set.copy()
    if tlu_optimizer._statistics:
        transformed_input_set = transformed_input_set * tlu_optimizer._statistics[0]["a"]
        transformed_input_set = transformed_input_set - tlu_optimizer._statistics[0]["b"]
        if lsbs_to_remove is not None:
            assert isinstance(lsbs_to_remove, int)
            tlu_optimizer.rounding_function(transformed_input_set, lsbs_to_remove=lsbs_to_remove)
        transformed_input_set = transformed_input_set + tlu_optimizer._statistics[0]["b"]
        transformed_input_set = transformed_input_set / tlu_optimizer._statistics[0]["a"]
    approx_reference = f(transformed_input_set)

    simulated = numpy.array([circuit.simulate(numpy.array([elt])) for elt in input_set])[..., 0]
    simulated_no_optim_no_rounding = numpy.array(
        [circuit_no_optim_no_rounding.simulate(numpy.array([elt])) for elt in input_set]
    )[..., 0]
    simulated_no_optim_rounding = numpy.array(
        [circuit_no_optim_rounding.simulate(numpy.array([elt])) for elt in input_set]
    )[..., 0]

    debug = True
    if debug:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.step(input_set, reference, label="reference", color="blue")
        ax.step(input_set, approx_reference, label="approx reference", color="purple")
        ax.step(input_set, simulated, label="optimized", color="red", linestyle=(0, (5, 5)))
        ax.step(
            input_set,
            simulated_no_optim_no_rounding,
            label="not optimized, not rounded",
            color="green",
            linestyle=(0, (5, 5)),
        )
        ax.step(
            input_set,
            simulated_no_optim_rounding,
            label="not optimized, rounded",
            color="yellow",
            linestyle=(0, (5, 5)),
        )
        ax.vlines(
            x=input_set[0],
            ymin=reference.min(),
            ymax=reference.max(),
            linestyle=(0, (1, 1)),
            color="grey",
            label="x_min",
        )
        ax.vlines(
            x=input_set[-1],
            ymin=reference.min(),
            ymax=reference.max(),
            linestyle=(0, (1, 1)),
            color="grey",
            label="x_max",
        )
        plt.legend()
        fig.savefig(f"debug_{execution_number}.png")
        plt.close(fig)

        with open(f"debug_{execution_number}.graph", "w", encoding="utf-8") as file:
            file.write(circuit.graph.format())

        with open(f"debug_{execution_number}.mlir", "w", encoding="utf-8") as file:
            file.write(circuit.mlir)

    not_equal = reference != simulated
    if not_equal.sum() > len(thresholds):
        raise Exception(
            f"TLU Optimizer is not exact: {not_equal.mean()=} = {not_equal.sum()}/{not_equal.size}\n"
            f"{tlu_optimizer._statistics=}"
            f"{execution_number=}, {lsbs_to_remove=} {(reference == approx_reference).mean()=}\n"
            f"{circuit.graph.format()}\n\n"
            f"{circuit_no_optim_no_rounding.graph.format()}"
        )
