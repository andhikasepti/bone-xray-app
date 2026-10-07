"""
Tahap 1: Melatih model klasifikasi normal vs tumor pada BTXRD.

Transfer learning dengan ResNet18 pralatih ImageNet (hanya fully-connected
layer akhir yang dilatih ulang pada baseline ini).

Cara pakai:
    python src/train_classifier.py \
        --excel data/dataset.xlsx \
        --images data/images \
        --epochs 10 \
        --out models/classifier_baseline.pt
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models

from dataset import BoneXrayDataset
from preprocess import get_train_transform, get_eval_transform


def build_model(num_classes=2, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_loss_acc(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", required=True, help="Path ke dataset.xlsx")
    parser.add_argument("--images", required=True, help="Folder berisi citra")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="models/classifier_baseline.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    # NOTE: split train/val di sini acak per-citra untuk baseline cepat.
    # Jika dataset punya ID pasien, sebaiknya pisahkan per pasien untuk
    # menghindari data leakage (lihat catatan Bab 9).
    full_dataset = BoneXrayDataset(args.excel, args.images, transform=get_train_transform())
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = get_eval_transform()

    print(f"Distribusi kelas (seluruh data): {full_dataset.class_counts()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_loss_acc(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.out)
            print(f"  -> Model terbaik disimpan ke {args.out} (val_acc={val_acc:.4f})")

    print(f"Selesai. Val accuracy terbaik: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
