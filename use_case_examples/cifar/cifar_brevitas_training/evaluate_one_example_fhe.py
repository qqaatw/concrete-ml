import json
import os
import time
from functools import partial
from importlib.metadata import version
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from concrete.fhe import Exactness, round_bit_pattern, truncate_bit_pattern
from concrete.fhe.compilation.configuration import Configuration
from models import cnv_2w2a
from torch.utils.data import DataLoader
from trainer import get_test_set

from concrete import fhe
from concrete.ml.common.preprocessors import InsertRounding, TLU1bitDecomposition
from concrete.ml.quantization import QuantizedModule
from concrete.ml.torch.compile import compile_brevitas_qat_model

SIMULATE_ONLY = False
CURRENT_DIR = Path(__file__).resolve().parent
KEYGEN_CACHE_DIR = CURRENT_DIR.joinpath(".keycache")

# Add MPS (for macOS with Apple Silicon or AMD GPUs) support when error is fixed. For now, we
# observe a decrease in torch's top1 accuracy when using MPS devices
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3953
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", 1000 if SIMULATE_ONLY else 1))
P_ERROR = float(os.environ.get("P_ERROR", 0.01))


def measure_execution_time(func):
    """Run a function and return execution time and outputs.

    Usage:
        def f(x, y):
            return x + y
        output, execution_time = measure_execution_time(f)(x,y)
    """

    def wrapper(*args, **kwargs):
        # Get the current time
        start = time.time()

        # Call the function
        result = func(*args, **kwargs)

        # Get the current time again
        end = time.time()

        # Calculate the execution time
        execution_time = end - start

        # Return the result and the execution time
        return result, execution_time

    return wrapper


# Instantiate the model
torch_model = cnv_2w2a(pre_trained=False)
torch_model.eval()


# Load the saved parameters using the available checkpoint
checkpoint = torch.load(
    CURRENT_DIR.joinpath("experiments/CNV_2W2A_2W2A_20221114_131345/checkpoints/best.tar"),
    map_location=DEVICE,
)
torch_model.load_state_dict(checkpoint["state_dict"], strict=False)

# Import and load the CIFAR test dataset
test_set = get_test_set(dataset="CIFAR10", datadir=CURRENT_DIR.joinpath(".datasets/"))
test_loader = DataLoader(test_set, batch_size=NUM_SAMPLES, shuffle=False)

# Get the first sample
x, labels = next(iter(test_loader))

# Parameter `enable_unsafe_features` and `use_insecure_key_cache` are needed in order to be able to
# cache generated keys through `insecure_key_cache_location`. As the name suggests, these
# parameters are unsafe and should only be used for debugging in development
# Multi-parameter strategy is used in order to speed-up the FHE executions
base_configuration = Configuration()

exactness = Exactness.APPROXIMATE
msbs_to_keep = 1
rounding_function = round_bit_pattern

tlu_optimizer = TLU1bitDecomposition(
    n_jumps_limit=2,
    exactness=exactness,
    msbs_to_keep=msbs_to_keep,
    rounding_function=rounding_function,
)
rounding = InsertRounding(6, exactness=exactness)

configuration = Configuration(
    dump_artifacts_on_unexpected_failures=False,
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location=KEYGEN_CACHE_DIR,
    additional_pre_processors=[
        tlu_optimizer,
        rounding,
    ],
    fhe_simulation=SIMULATE_ONLY,
    fhe_execution=not SIMULATE_ONLY,
)


print("Compiling the model.")
quantized_numpy_module, compilation_execution_time = measure_execution_time(
    compile_brevitas_qat_model
)(
    torch_model,
    x,
    configuration=configuration,
    p_error=P_ERROR,
)
assert isinstance(quantized_numpy_module, QuantizedModule)
assert quantized_numpy_module.fhe_circuit is not None

print(tlu_optimizer.statistics)

print(f"Compilation time took {compilation_execution_time} seconds")

# Display the max bit-width in the model
print(
    "Max bit-width used in the circuit: ",
    f"{quantized_numpy_module.fhe_circuit.graph.maximum_integer_bit_width()} bits",
)

# Save the graph and mlir
print("Saving graph and mlir to disk.")
open("cifar10.graph", "w").write(str(quantized_numpy_module.fhe_circuit))
open("cifar10.mlir", "w").write(quantized_numpy_module.fhe_circuit.mlir)

pprint(quantized_numpy_module.fhe_circuit.statistics)

# import sys
#
# if sys.platform == "darwin":
#     print("skipping fhe evaluation on darwin platform")
#     sys.exit(0)

if not SIMULATE_ONLY:
    # Key generation
    print("Creation of the private and evaluation keys.")
    _, keygen_execution_time = measure_execution_time(quantized_numpy_module.fhe_circuit.keygen)(
        force=True
    )
    print(f"Keygen took {keygen_execution_time} seconds")

# Data torch to numpy
x_numpy = x.numpy()

# Initialize a list to store all the results
all_results = []

# Iterate through the NUM_SAMPLES
for image_index in range(NUM_SAMPLES):
    # Take one example
    test_x_numpy = x_numpy[image_index : image_index + 1]

    # Get the torch prediction
    torch_output = torch_model(x[image_index : image_index + 1])

    # Quantize the input
    q_x_numpy, quantization_execution_time = measure_execution_time(
        quantized_numpy_module.quantize_input
    )(test_x_numpy)

    print(f"Quantization of a single input (image) took {quantization_execution_time} seconds")
    print(f"Size of CLEAR input is {q_x_numpy.nbytes} bytes\n")

    expected_quantized_prediction, clear_inference_time = measure_execution_time(
        partial(quantized_numpy_module.fhe_circuit.simulate)
    )(q_x_numpy)

    if not SIMULATE_ONLY:
        # Encrypt the input
        encrypted_q_x_numpy, encryption_execution_time = measure_execution_time(
            quantized_numpy_module.fhe_circuit.encrypt
        )(q_x_numpy)
        print(f"Encryption of a single input (image) took {encryption_execution_time} seconds\n")

        print(
            f"Size of ENCRYPTED input is {quantized_numpy_module.fhe_circuit.size_of_inputs} bytes"
        )
        print(
            f"Size of ENCRYPTED output is {quantized_numpy_module.fhe_circuit.size_of_outputs} bytes"
        )
        print(
            f"Size of keyswitch key is {quantized_numpy_module.fhe_circuit.size_of_keyswitch_keys} bytes"
        )
        print(
            f"Size of bootstrap key is {quantized_numpy_module.fhe_circuit.size_of_bootstrap_keys} bytes"
        )
        print(
            f"Size of secret key is {quantized_numpy_module.fhe_circuit.size_of_secret_keys} bytes"
        )
        print(f"Complexity is {quantized_numpy_module.fhe_circuit.complexity}\n")

        print("Running FHE inference")
        fhe_output, fhe_execution_time = measure_execution_time(
            quantized_numpy_module.fhe_circuit.run
        )(encrypted_q_x_numpy)
        print(f"FHE inference over a single image took {fhe_execution_time}")

        # Decrypt print the result
        decrypted_fhe_output, decryption_execution_time = measure_execution_time(
            quantized_numpy_module.fhe_circuit.decrypt
        )(fhe_output)

    else:
        decrypted_fhe_output, _ = measure_execution_time(
            quantized_numpy_module.fhe_circuit.simulate
        )(q_x_numpy)

    print(
        f"Expected prediction. Class={np.argmax(expected_quantized_prediction)} Logits={expected_quantized_prediction}"
    )
    print(
        f"Circuit prediction with {'simulation' if not SIMULATE_ONLY else 'FHE'}: Class={np.argmax(decrypted_fhe_output)} Logits={decrypted_fhe_output}"
    )

    result = {
        "image_index": image_index,
        # Timings
        "quantization_time": quantization_execution_time,
        "inference_time": clear_inference_time,
        "label": labels[image_index].item(),
        "p_error": P_ERROR,
    }

    if not SIMULATE_ONLY:
        result = {
            **result,
            **{
                "encryption_time": encryption_execution_time,
                "fhe_time": fhe_execution_time,
                "decryption_time": decryption_execution_time,
            },
        }

    for prediction_index, prediction in enumerate(expected_quantized_prediction[0]):
        result[f"quantized_prediction_{prediction_index}"] = prediction
    for prediction_index, prediction in enumerate(decrypted_fhe_output[0]):
        result[f"prediction_{prediction_index}"] = prediction
    for prediction_index, prediction in enumerate(torch_output[0]):
        result[f"torch_prediction_{prediction_index}"] = prediction.item()

    all_results.append(result)


# Write the results to a CSV file
with open("inference_results.csv", "w", encoding="utf-8") as file:
    # Write the header row
    columns = list(all_results[0].keys())
    file.write(",".join(columns) + "\n")

    # Write the data rows
    for result in all_results:
        file.write(",".join(str(result[column]) for column in columns) + "\n")

metadata = {
    "p_error": P_ERROR,
    "cml_version": version("concrete-ml"),
    "cnp_version": version("concrete-python"),
}
with open("metadata.json", "w") as file:
    json.dump(metadata, file)
