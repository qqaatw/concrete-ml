import numpy 
import torch 
from torch import nn
from pathlib import Path

from concrete.ml.torch.compile import compile_torch_model

from cifar import Net, load_data

def test(
    quantized_module,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,  # pylint: disable=no-member
    fhe="disable",
):
    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = quantized_module.forward(images, fhe=fhe)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy

def main():
    script_dir = Path(__file__).parent
    weights = numpy.load(script_dir / f"weights/round-10-weights.npz", allow_pickle=False)
    model = Net()
    model.set_weights(weights)
    
    compile_set = numpy.random.randint(0, 255, (200, 32, 32)).astype(float)
    
    quantized_module = compile_torch_model(
        model,
        compile_set,
        rounding_threshold_bits=6,
    )
    
    _, testloader = load_data()
    
    _, accuracy = test(quantized_module, testloader, device="cpu", fe="simulate")
    
    print("Simulated accuracy:", accuracy)
    
    
if __name__ == "__main__":
    main()
