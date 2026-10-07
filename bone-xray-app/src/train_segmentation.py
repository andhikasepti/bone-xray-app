"""
Tahap 3: Segmentasi tumor dengan U-Net (segmentation-models-pytorch).

Mask biner di-generate dari shape polygon pada anotasi LabelMe (lihat
annotations.py). Augmentasi (albumentations) hanya untuk data training;
early stopping berdasarkan val_dice.

Cara pakai:
    python src/train_segmentation.py \
        --images data/images \
        --annotations data/Annotations \
        --masks-dir masks \
        --epochs 50 --patience 8 \
        --out models/segmentation_unet.pt
"""

import argparse
import os

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from annotations import build_all_masks

IMAGE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=0),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.ElasticTransform(alpha=1, sigma=30, p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class BoneSegmentationDataset(Dataset):
    def __init__(self, filenames, images_dir, masks_dir, transform):
        self.filenames = filenames
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        base = os.path.splitext(fname)[0]

        image = np.array(Image.open(os.path.join(self.images_dir, fname)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.masks_dir, base + ".png")))

        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"].unsqueeze(0).float()

        return image, mask


def dice_coefficient(pred_logits, true_mask, eps=1e-7):
    pred_mask = (torch.sigmoid(pred_logits) > 0.5).float()
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum()
    return (2 * intersection + eps) / (union + eps)


def iou_score(pred_logits, true_mask, eps=1e-7):
    pred_mask = (torch.sigmoid(pred_logits) > 0.5).float()
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    return (intersection + eps) / (union + eps)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        total_loss += loss.item() * images.size(0)
        total_dice += dice_coefficient(outputs, masks).item() * images.size(0)
        total_iou += iou_score(outputs, masks).item() * images.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_dice / n, total_iou / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--masks-dir", default="masks")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="models/segmentation_unet.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    # 1. Generate mask biner dari polygon
    image_filenames = build_all_masks(args.annotations, args.masks_dir)
    print(f"Jumlah citra dengan mask segmentasi: {len(image_filenames)}")

    # 2. Split train/val/test
    train_files, temp_files = train_test_split(image_filenames, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    train_ds = BoneSegmentationDataset(train_files, args.images, args.masks_dir, get_train_transform())
    val_ds = BoneSegmentationDataset(val_files, args.images, args.masks_dir, get_eval_transform())
    test_ds = BoneSegmentationDataset(test_files, args.images, args.masks_dir, get_eval_transform())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 3. Model: U-Net, encoder ResNet18 pralatih ImageNet
    model = smp.Unet(
        encoder_name="resnet18", encoder_weights="imagenet",
        in_channels=3, classes=1, activation=None,
    ).to(device)

    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode="binary")
    loss_fn = lambda pred, target: bce_loss(pred, target) + dice_loss(pred, target)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Lanjutkan dari checkpoint sebelumnya jika ada (berguna kalau training
    # sempat terputus -- lihat catatan di README soal Colab)
    if os.path.exists(args.out):
        model.load_state_dict(torch.load(args.out, map_location=device))
        _, best_val_dice, _ = evaluate(model, val_loader, loss_fn, device)
        print(f"Melanjutkan dari checkpoint {args.out} (val_dice awal={best_val_dice:.4f})")
    else:
        best_val_dice = 0.0
        print("Tidak ada checkpoint, mulai dari awal.")

    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.out)
            print(f"  -> Model terbaik disimpan (val_dice={val_dice:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping di epoch {epoch}.")
                break

    # 4. Evaluasi akhir di test set
    model.load_state_dict(torch.load(args.out, map_location=device))
    test_loss, test_dice, test_iou = evaluate(model, test_loader, loss_fn, device)
    print("\n=== Hasil Evaluasi pada Test Set ===")
    print(f"Dice score : {test_dice:.4f}")
    print(f"IoU        : {test_iou:.4f}")
    print("Pembanding dari paper BTXRD (konsistensi antar dua dokter): Dice 0.90 | IoU 0.91")


if __name__ == "__main__":
    main()
