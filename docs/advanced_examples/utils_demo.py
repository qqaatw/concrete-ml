import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.metrics import accuracy_score
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat
import brevitas.nn as qnn
import torch
from tqdm import tqdm
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat
from torch import nn
from concrete.ml.torch.compile import compile_brevitas_qat_model

from torch.utils.data import Dataset, DataLoader

import torch
from sklearn.datasets import make_classification, make_moons, make_circles


# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def plot_datasets():
    # Generate make_classification dataset
    X_clf, y_clf = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    rng = np.random.RandomState(2)
    X_clf += 2 * rng.uniform(size=X_clf.shape)
    
    # Generate make_moons dataset
    X_moons, y_moons = make_moons(n_samples=500, noise=0.2, random_state=42)
    
    # Generate make_circles dataset
    X_circles, y_circles = make_circles(n_samples=500, noise=0.2, factor=0.5, random_state=42)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot make_classification dataset
    scatter = axes[0].scatter(X_clf[:, 0], X_clf[:, 1], c=y_clf, cmap='bwr', edgecolors='k')
    axes[0].set_title('make_classification')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].legend(handles=scatter.legend_elements()[0], labels=['Class 0', 'Class 1'])
    
    # Plot make_moons dataset
    scatter = axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='bwr', edgecolors='k')
    axes[1].set_title('make_moons')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].legend(handles=scatter.legend_elements()[0], labels=['Class 0', 'Class 1'])
    
    # Plot make_circles dataset
    scatter = axes[2].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='bwr', edgecolors='k')
    axes[2].set_title('make_circles')
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].legend(handles=scatter.legend_elements()[0], labels=['Class 0', 'Class 1'])
    
    # Adjust subplot spacing
    plt.tight_layout()
    
    # Show the plots
    plt.show()
    return (X_clf, y_clf), (X_moons, y_moons), (X_circles, y_circles)


def plot_data(X_train, X_test, y_train, y_test, title):
    _, ax = plt.subplots(1, figsize=(7, 4))
    ax.set_title(title)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,  marker="o", cmap="jet", label="Train data", alpha=0.6)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker="x", cmap="jet", label="Test data")
    ax.legend(loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show();


# def plot_counter_plot(clf, X_train, X_test, y_train, y_test):


#     if hasattr(clf, "n_bits"):
#         clf.compile(X_train)
#         y_pred = clf.predict(X_test, fhe="simulate")
#         accuracy = accuracy_score(y_test, y_pred)
#         title = f"Concrete ML decision boundaries with {clf.n_bits}-bits of quantization"
#     else:
#         print("sklearn")
#         y_pred = clf.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         title = f"Scikit-learn decision boundaries"
    
#     plt.ioff()
#     plt.clf()
#     fig, ax = plt.subplots(1, figsize=(12, 8))
#     fig.patch.set_facecolor("white")

#     X = np.concatenate((X_train, X_test))
#     b_min = np.min(X, axis=0)
#     b_max = np.max(X, axis=0)

#     x_test_grid, y_test_grid = np.meshgrid(
#         np.linspace(b_min[0], b_max[0], 50), np.linspace(b_min[1], b_max[1], 50)
#     )
#     x_grid_test = np.vstack([x_test_grid.ravel(), y_test_grid.ravel()]).transpose()
#     y_score_grid = clf.predict_proba(x_grid_test)[:, 1]

#     ax.contourf(x_test_grid, y_test_grid, y_score_grid.reshape(x_test_grid.shape), cmap="coolwarm", alpha=0.7)
#     CS1 = ax.contour(
#         x_test_grid,
#         y_test_grid,
#         y_score_grid.reshape(x_test_grid.shape),
#         levels=[0.5],
#         linewidths=2,
#     )

#     CS1.collections[0].set_label(f"title | Accuracy = {accuracy:.2%}")
#     if hasattr(clf, "n_bits"):
#         CS1.collections[0].set_label(f"{clf.fhe_circuit.graph.maximum_integer_bit_width()} max bit-width in the circuit")
#     ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker="o", cmap="jet", label="Train data")
#     ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker="x", cmap="jet", label="Test data")
#     ax.legend(loc="upper right")
#     ax.set_title(title)
#     plt.show()


class QuantCustomModel(nn.Module):
    """A small quantized network with Brevitas, trained on make_classification."""

    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        hidden_shape: int = 100,
        n_bits: int = 5,
        act_quant=Int8ActPerTensorFloat,
        weight_quant=Int8WeightPerTensorFloat,
    ):
        """Quantized Torch Model with Brevitas.

        Args:
            input_shape (int): Input size
            output_shape (int): Output size
            hidden_shape (int): Hidden size
            n_bits (int): Bit of quantization
            weight_quant (brevitas.quant): Quantization protocol of weights
            act_quant (brevitas.quant): Quantization protocol of activations.

        """
        super().__init__()

        self.quant_input = qnn.QuantIdentity(
            bit_width=n_bits, act_quant=act_quant, return_quant_tensor=True
        )
        self.linear1 = qnn.QuantLinear(
            in_features=input_shape,
            out_features=hidden_shape,
            weight_bit_width=n_bits,
            weight_quant=weight_quant,
            bias=True,
            return_quant_tensor=True,
        )

        self.relu1 = qnn.QuantReLU(return_quant_tensor=True, bit_width=n_bits, act_quant=act_quant)
        self.linear2 = qnn.QuantLinear(
            in_features=hidden_shape,
            out_features=output_shape,
            weight_bit_width=n_bits,
            weight_quant=weight_quant,
            bias=True,
            return_quant_tensor=True,
        )


    def forward(self, x):

        x = self.quant_input(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x.value


class CustomNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()


def plot_decision_boundary(model, dataset):
    # Generate a grid of points within the input range
    X, y = dataset[0], dataset[1]
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02  # step size for the meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Convert the grid points to tensors
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    # Make predictions for each point in the grid
    Z = model.predict(grid_tensor)
    Z = Z.reshape(xx.shape)

    # Create a contour plot
    plt.contourf(xx, yy, Z, alpha=0.8)

    # Plot the training samples with labels
    classes = np.unique(y)
    for class_label in classes:
        class_X = X[y == class_label]
        plt.scatter(class_X[:, 0], class_X[:, 1], label=f"Class {class_label}", edgecolors='k', alpha=0.7)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Decision Boundaries')
    plt.legend()
    
    plt.show()

