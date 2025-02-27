# Using Torch

This document explains how to implement machine learning models with Torch in Concrete ML, leveraging Fully Homomorphic Encryption (FHE).

## Introduction

There are two approaches to build [FHE-compatible deep networks](../getting-started/concepts.md#model-accuracy-considerations-under-fhe-constraints):

- [**Quantization Aware Training (QAT)**](../getting-started/concepts.md#i-model-development): This method requires using custom layers to quantize weights and activations to low bit-widths. Concrete ML works with [Brevitas](../explanations/inner-workings/external_libraries.md#brevitas), a library that provides QAT support for PyTorch.

  - Use `compile_brevitas_qat_model` to compile models in this mode.

- [**Post Training Quantization (PTQ)**](../getting-started/concepts.md#i-model-development): This method allows to compile a vanilla PyTorch model. However, accuracy may decrease significantly when quantizing weights and activations to fewer than 7 bits. On the other hand, depending on the model size, quantizing with 6-8 bits can be incompatible with FHE constraints. Thus you need to determine the trade-off between model accuracy and FHE compatibility.

  - Use `compile_torch_model` to compile models in this mode.

Both approaches require setting `rounding_threshold_bits` parameter accordingly. You should experiment to find the best values, starting with an initial value of `6`. See [here](../explanations/advanced_features.md#rounded-activations-and-quantizers) for more details.

{% hint style="info" %}
See the [common compilation errors page](./fhe_assistant.md#common-compilation-errors) for explanations and solutions to some common errors raised by the compilation function.
{% endhint %}

## Quantization Aware training (QAT)

The following example uses a simple QAT PyTorch model that implements a fully connected neural network with two hidden layers. Due to its small size, making this model respect FHE constraints is relatively easy. To use QAT, Brevitas `QuantIdentity` nodes must be inserted in the PyTorch model, including one that quantizes the input of the `forward` function.

```python
import brevitas.nn as qnn
import torch.nn as nn
import torch

N_FEAT = 12
n_bits = 3

class QATSimpleNet(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()

        self.quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.fc1 = qnn.QuantLinear(N_FEAT, n_hidden, True, weight_bit_width=n_bits, bias_quant=None)
        self.quant2 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.fc2 = qnn.QuantLinear(n_hidden, n_hidden, True, weight_bit_width=n_bits, bias_quant=None)
        self.quant3 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.fc3 = qnn.QuantLinear(n_hidden, 2, True, weight_bit_width=n_bits, bias_quant=None)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.quant2(torch.relu(self.fc1(x)))
        x = self.quant3(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

```

Once the model is trained, use [`compile_brevitas_qat_model`](../references/api/concrete.ml.torch.compile.md#function-compile_brevitas_qat_model) from Concrete ML to perform conversion and compilation of the QAT network. Here, 3-bit quantization is used for both the weights and activations. This function automatically identifies the number of quantization bits used in the Brevitas model.

<!--pytest-codeblocks:cont-->

```python
from concrete.ml.torch.compile import compile_brevitas_qat_model
import numpy

torch_input = torch.randn(100, N_FEAT)
torch_model = QATSimpleNet(30)
quantized_module = compile_brevitas_qat_model(
    torch_model, # our model
    torch_input, # a representative input-set to be used for both quantization and compilation
    rounding_threshold_bits={"n_bits": 6, "method": "approximate"}
)

```

{% hint style="warning" %}
If `QuantIdentity` layers are missing for any input or intermediate value, the compile function will raise an error. See the [common compilation errors page](./fhe_assistant.md#common-compilation-errors) for an explanation.
{% endhint %}

## Post Training quantization (PTQ)

The following example demonstrates a simple PyTorch model that implements a fully connected neural network with two hidden layers. The model is compiled with `compile_torch_model` to use FHE.

```python
import torch.nn as nn
import torch

N_FEAT = 12
n_bits = 6

class PTQSimpleNet(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()

        self.fc1 = nn.Linear(N_FEAT, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

from concrete.ml.torch.compile import compile_torch_model
import numpy

torch_input = torch.randn(100, N_FEAT)
torch_model = PTQSimpleNet(5)
quantized_module = compile_torch_model(
    torch_model, # our model
    torch_input, # a representative input-set to be used for both quantization and compilation
    n_bits=6,
    rounding_threshold_bits={"n_bits": 6, "method": "approximate"}
)
```

## Configuring quantization parameters

The quantization parameters, along with the number of neurons in each layer, determine the accumulator bit-width of the network. Larger accumulator bit-widths result in higher accuracy but slower FHE inference time.

**QAT**: Configure parameters such as `bit_width` and `weight_bit_width`. Set `n_bits=None` in the `compile_brevitas_qat_model`.

**PTQ**: Set the `n_bits` value in the `compile_torch_model` function. Manually determine the trade-off between accuracy, FHE compatibility, and latency.

## Running encrypted inference

The model can now perform encrypted inference.

<!--pytest-codeblocks:cont-->

```python
x_test = numpy.array([numpy.random.randn(N_FEAT)])

y_pred = quantized_module.forward(x_test, fhe="execute")
```

In this example, the input values `x_test` and the predicted values `y_pred` are floating points. The quantization (respectively de-quantization) step is done in the clear within the `forward` method, before (respectively after) any FHE computations.

## Simulated FHE Inference in the clear

You can perform the inference on clear data in order to evaluate the impact of quantization and of FHE computation on the accuracy of their model. See [this section](../deep-learning/fhe_assistant.md#simulation) for more details.

There are two approaches:

- `quantized_module.forward(quantized_x, fhe="simulate")`: This method simulates FHE execution taking into account Table Lookup errors. De-quantization must be done in a second step as for actual FHE execution. Simulation takes into account the `p_error`/`global_p_error` parameters
- `quantized_module.forward(quantized_x, fhe="disable")`: This method computes predictions in the clear on quantized data, and then de-quantize the result. The return value of this function contains the de-quantized (float) output of running the model in the clear. Calling this function on clear data is useful when debugging, but this does not perform actual FHE simulation.

{% hint style="info" %}
FHE simulation allows to measure the impact of the Table Lookup error on the model accuracy. You can adjust the Table Lookup error using `p_error`/`global_p_error`, as described in the [approximate computation ](../explanations/advanced_features.md#approximate-computations)section.
{% endhint %}

## Supported operators and activations

Concrete ML supports a variety of PyTorch operators that can be used to build fully connected or convolutional neural networks, with normalization and activation layers. Moreover, many element-wise operators are supported.

### Operators

#### Univariate operators

- [`torch.nn.identity`](https://pytorch.org/docs/stable/generated/torch.nn.Identity.html)
- [`torch.clip`](https://pytorch.org/docs/stable/generated/torch.clip.html)
- [`torch.clamp`](https://pytorch.org/docs/stable/generated/torch.clamp.html)
- [`torch.round`](https://pytorch.org/docs/stable/generated/torch.round.html)
- [`torch.floor`](https://pytorch.org/docs/stable/generated/torch.floor.html)
- [`torch.min`](https://pytorch.org/docs/stable/generated/torch.min.html)
- [`torch.max`](https://pytorch.org/docs/stable/generated/torch.max.html)
- [`torch.abs`](https://pytorch.org/docs/stable/generated/torch.abs.html)
- [`torch.neg`](https://pytorch.org/docs/stable/generated/torch.neg.html)
- [`torch.sign`](https://pytorch.org/docs/stable/generated/torch.sign.html)
- [`torch.logical_or, torch.Tensor operator ||`](https://pytorch.org/docs/stable/generated/torch.logical_or.html)
- [`torch.logical_not`](https://pytorch.org/docs/stable/generated/torch.logical_not.html)
- [`torch.gt, torch.greater`](https://pytorch.org/docs/stable/generated/torch.gt.html)
- [`torch.ge, torch.greater_equal`](https://pytorch.org/docs/stable/generated/torch.ge.html)
- [`torch.lt, torch.less`](https://pytorch.org/docs/stable/generated/torch.lt.html)
- [`torch.le, torch.less_equal`](https://pytorch.org/docs/stable/generated/torch.le.html)
- [`torch.eq`](https://pytorch.org/docs/stable/generated/torch.eq.html)
- [`torch.where`](https://pytorch.org/docs/stable/generated/torch.where.html)
- [`torch.exp`](https://pytorch.org/docs/stable/generated/torch.exp.html)
- [`torch.log`](https://pytorch.org/docs/stable/generated/torch.log.html)
- [`torch.pow`](https://pytorch.org/docs/stable/generated/torch.pow.html)
- [`torch.sum`](https://pytorch.org/docs/stable/generated/torch.sum.html)
- [`torch.mul, torch.Tensor operator *`](https://pytorch.org/docs/stable/generated/torch.mul.html)
- [`torch.div, torch.Tensor operator /`](https://pytorch.org/docs/stable/generated/torch.div.html)
- [`torch.nn.BatchNorm2d`](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
- [`torch.nn.BatchNorm3d`](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm3d.html)
- [`torch.erf, torch.special.erf`](https://pytorch.org/docs/stable/special.html#torch.special.erf)
- [`torch.nn.functional.pad`](https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html)

#### Shape modifying operators

- [`torch.reshape`](https://pytorch.org/docs/stable/generated/torch.reshape.html)
- [`torch.Tensor.view`](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view)
- [`torch.flatten`](https://pytorch.org/docs/stable/generated/torch.flatten.html)
- [`torch.unsqueeze`](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html)
- [`torch.squeeze`](https://pytorch.org/docs/stable/generated/torch.squeeze.html)
- [`torch.transpose`](https://pytorch.org/docs/stable/generated/torch.transpose.html)
- [`torch.concat, torch.cat`](https://pytorch.org/docs/stable/generated/torch.cat.html)
- [`torch.nn.Unfold`](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html)

#### Tensor operators

- [`torch.Tensor.expand`](https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html)
- [`torch.Tensor.to`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html) -- for casting to dtype

#### Multi-variate operators: encrypted input and unencrypted constants

- [`torch.nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [`torch.conv1d`, `torch.nn.Conv1D`](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
- [`torch.conv2d`, `torch.nn.Conv2D`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [`torch.nn.AvgPool2d`](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html)
- [`torch.nn.MaxPool2d`](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)

Concrete ML also supports some of their QAT equivalents from Brevitas.

- `brevitas.nn.QuantLinear`
- `brevitas.nn.QuantConv1d`
- `brevitas.nn.QuantConv2d`

#### Multi-variate operators: encrypted+unencrypted or encrypted+encrypted inputs

- [`torch.add, torch.Tensor operator +`](https://pytorch.org/docs/stable/generated/torch.Tensor.add.html)
- [`torch.sub, torch.Tensor operator -`](https://pytorch.org/docs/stable/generated/torch.Tensor.sub.html)
- [`torch.matmul`](https://pytorch.org/docs/stable/generated/torch.matmul.html)

### Quantizers

- `brevitas.nn.QuantIdentity`

### Activation functions

- [`torch.nn.CELU`](https://pytorch.org/docs/stable/generated/torch.nn.CELU.html)
- [`torch.nn.ELU`](https://pytorch.org/docs/stable/generated/torch.nn.ELU.html)
- [`torch.nn.GELU`](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)
- [`torch.nn.HardSigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html)
- [`torch.nn.Hardswish`](https://pytorch.org/docs/stable/generated/torch.nn.Hardswish)
- [`torch.nn.HardTanh`](https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html)
- [`torch.nn.LeakyReLU`](https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html)
- [`torch.nn.LogSigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.LogSigmoid.html)
- [`torch.nn.Mish`](https://pytorch.org/docs/stable/generated/torch.nn.Mish.html)
- [`torch.nn.PReLU`](https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html)
- [`torch.nn.ReLU6`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html)
- [`torch.nn.ReLU`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)
- [`torch.nn.SELU`](https://pytorch.org/docs/stable/generated/torch.nn.SELU.html)
- [`torch.nn.Sigmoid`](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)
- [`torch.nn.SiLU`](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html)
- [`torch.nn.Softplus`](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html)
- [`torch.nn.Softshrink`](https://pytorch.org/docs/stable/generated/torch.nn.Softshrink.html)
- [`torch.nn.Softsign`](https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html)
- [`torch.nn.Tanh`](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html)
- [`torch.nn.Tanhshrink`](https://pytorch.org/docs/stable/generated/torch.nn.Tanhshrink.html)
- [`torch.nn.Threshold`](https://pytorch.org/docs/stable/generated/torch.nn.Threshold.html) -- partial support

{% hint style="info" %}
The equivalent versions from `torch.functional` are also supported.
{% endhint %}
