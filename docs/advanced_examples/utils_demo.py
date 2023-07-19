import time
import random
import numpy as np
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import brevitas.nn as qnn

from brevitas import config

from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

from brevitas.nn import QuantLinear, QuantReLU

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
    scatter = axes[0].scatter(X_clf[:, 0], X_clf[:, 1], c=y_clf, cmap='coolwarm', edgecolors='k')
    axes[0].set_title('make_classification')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].legend(handles=scatter.legend_elements()[0], labels=['Class 0', 'Class 1'])
    
    # Plot make_moons dataset
    scatter = axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='coolwarm', edgecolors='k')
    axes[1].set_title('make_moons')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].legend(handles=scatter.legend_elements()[0], labels=['Class 0', 'Class 1'])
    
    # Plot make_circles dataset
    scatter = axes[2].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='coolwarm', edgecolors='k')
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

        self.n_bits = n_bits

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
    def __init__(self,
        input_shape: int,
        output_shape: int,
        hidden_shape: int = 100):
        super(CustomNeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_shape, hidden_shape)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_shape, output_shape)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out
    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()


def plot_decision_boundary(model, dataset, title, qmodel=None):
    # Generate a grid of points within the input range
    X, y = dataset[0], dataset[1]
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02  # step size for the meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Convert the grid points to tensors
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    # Make predictions for each point in the grid
    if isinstance(model, QuantCustomModel) and qmodel is not None:
        Z = qmodel.forward(grid_tensor.numpy(), fhe="simulate")
        Z = Z.argmax(1)
    else:
        Z = model.predict(grid_tensor)

    Z = Z.reshape(xx.shape)

    # Create a contour plot
    plt.contourf(xx, yy, Z, alpha=0.6,  cmap="coolwarm")

    # Plot the training samples with labels
    classes = np.unique(y)
    for class_label, color in zip(classes, ["blue", "red"]):
        class_X = X[y == class_label]
        plt.scatter(class_X[:, 0], class_X[:, 1], label=f"Class {class_label}",             
                c=color,
                edgecolors="k", alpha=0.8)
    plt.xlabel('X1')
    plt.ylabel('X2')
    # Remove ticks on both x and y axes
    plt.xticks([])
    plt.yticks([])
    plt.title(f'{title} decision Boundary')
    plt.legend()
    
    plt.show()


def mapping_keys(pre_trained_weights, model: nn.Module, device: str) -> nn.Module:

    """
    Initialize the quantized model with pre-trained fp32 weights.

    Args:
        pre_trained_weights (Dict): The state_dict of the pre-trained fp32 model.
        model (nn.Module): The Brevitas model.
        device (str): Device type.

    Returns:
        Callable: The quantized model with the pre-trained state_dict.
    """

    # Brevitas requirement to ignore missing keys
    config.IGNORE_MISSING_KEYS = True

    old_keys = list(pre_trained_weights.keys())
    new_keys = list(model.state_dict().keys())
    new_state_dict = OrderedDict()

    for old_key, new_key in zip(old_keys, new_keys):
        new_state_dict[new_key] = pre_trained_weights[old_key]

    model.load_state_dict(new_state_dict)
    model = model.to(device)

    return model

def train(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    path="checkpoint.pt",
    step=10,
    learning_rate=0.001,
    epochs=5000,
    verbose=False,
    seed=23,
    gamma=0.1,
    milestones=[5],
):
    # Define the loss function and optimizer
    torch.manual_seed(seed)
    random.seed(seed)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss_test = criterion(model(X_test), y_test)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Apply the learning rate scheduler
        scheduler.step()

        if (epoch + 1) % step == 0 and verbose:
            print(
                f"Epoch [{(epoch + 1):>{len(str(epochs))}} / {epochs}], "
                f"Train Loss: {loss.item():.2f}, Test Loss: {loss_test.item():.2f}, "
                f"Learning Rate: {scheduler.get_last_lr()[0]}"
            )

    # Test Evaluation
    with torch.no_grad():
        # Switch to evaluation mode
        model.eval()

        # Forward pass on the testing set
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)

        # Calculate accuracy
        test_accuracy = accuracy_score(y_test, predicted)

    # Train Evaluation
    with torch.no_grad():
        # Switch to evaluation mode
        model.eval()

        # Forward pass on the testing set
        outputs = model(X_train)
        _, predicted = torch.max(outputs.data, 1)

        # Calculate accuracy
        train_accuracy = accuracy_score(y_train, predicted)

    torch.save(model.state_dict(), path)

    print(f"\nTrain accuracy: {train_accuracy} vs Test accuracy: {test_accuracy}")

def torch_evaluation(model, X_data, y_data, device):
    with torch.no_grad():
        # Switch to evaluation mode
        model.eval()

        # Forward pass on the testing set
        outputs = model(X_data.to(device))
        _, predicted = torch.max(outputs.data, 1)

    # Calculate accuracy
    accuracy = accuracy_score(y_data, predicted.cpu().detach().numpy())
    print("Accuracy:", accuracy)