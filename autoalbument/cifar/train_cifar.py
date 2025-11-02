import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import Cifar10TrainDataset
from wrn import wide_resnet_28x10
from transforms import (
    get_base_train_transforms,
    get_policy_train_transforms,
    get_test_transforms,
)

# check if cuda is available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


def train_one(model, train_loader, test_loader, device, epochs, lr, weight_decay, writer, save_best_path):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_batches = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_batches += 1
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        if writer is not None:
            if total_batches > 0:
                writer.add_scalar("train/loss", running_loss / total_batches, epoch)
            writer.add_scalar("val/acc", acc, epoch)
        if acc > best_acc:
            best_acc = acc
            if save_best_path:
                torch.save({"model": model.state_dict(), "best_acc": best_acc}, save_best_path)
    return best_acc


def build_loaders(root, batch_size, workers, train_tfms, test_tfms, download):
    train_set = Cifar10TrainDataset(root=root, train=True, download=download, transform=train_tfms)
    test_set = Cifar10TrainDataset(root=root, train=False, download=download, transform=test_tfms) 
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=str(Path.home() / "data/cifar10"))
    parser.add_argument("--policy_json", type=str, default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--only_policy", action="store_true")
    parser.add_argument("--output_dir", type=str, default=str(Path("./outputs/cifar").resolve()))
    parser.add_argument("--log_dir", type=str, default=str(Path("./runs/cifar").resolve()))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    test_tfms = get_test_transforms()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    results = {}

    if not args.only_policy:
        base_tfms = get_base_train_transforms()
        base_train_loader, base_test_loader = build_loaders(
            args.data_root, args.batch_size, args.workers, base_tfms, test_tfms, args.download
        )
        base_model = wide_resnet_28x10(num_classes=10)
        base_best = train_one(
            base_model,
            base_train_loader,
            base_test_loader,
            device,
            args.epochs,
            args.lr,
            args.weight_decay,
            writer,
            os.path.join(args.output_dir, "cifar_baseline_best.pth"),
        )
        results["baseline_best_acc"] = base_best

    if args.policy_json and os.path.isfile(args.policy_json):
        policy_tfms = get_policy_train_transforms(args.policy_json)
        pol_train_loader, pol_test_loader = build_loaders(
            args.data_root, args.batch_size, args.workers, policy_tfms, test_tfms, args.download
        )
        pol_model = wide_resnet_28x10(num_classes=10)
        pol_best = train_one(
            pol_model,
            pol_train_loader,
            pol_test_loader,
            device,
            args.epochs,
            args.lr,
            args.weight_decay,
            writer,
            os.path.join(args.output_dir, "cifar_policy_best.pth"),
        )
        results["policy_best_acc"] = pol_best

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(results, f)
    print(json.dumps(results))
    writer.close()


if __name__ == "__main__":
    main()
