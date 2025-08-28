import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models import SimpleCNN
from utils import get_loaders


def train_model(lr=0.001, optimizer_name="adam", dropout=0.5, epochs=2, batch_size=64, device="mps"):
    trainloader, testloader = get_loaders(batch_size=batch_size)

    net = SimpleCNN(dropout=dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {running_loss/len(trainloader):.4f}")

    # quick eval
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")
    return acc


def main():
    parser = argparse.ArgumentParser(description="Train CNN on CIFAR-10 with hyperparams")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    train_model(
        lr=args.lr,
        optimizer_name=args.optimizer,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
