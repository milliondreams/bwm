"""Plain PyTorch training script. No AzureML imports — runs locally too.

AzureML injects args via the `command(...)` job spec; MLflow auto-logs metrics.
"""
import argparse
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--output-dir", type=str, default="./outputs")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlflow.log_params(vars(args))
    mlflow.log_param("device", str(device))

    tx = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    train_ds = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=tx)
    test_ds = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=tx)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_ds)

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
        acc = correct / len(test_ds)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("test_accuracy", acc, step=epoch)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} test_acc={acc:.4f}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    mlflow.pytorch.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()
