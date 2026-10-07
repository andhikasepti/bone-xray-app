"""
Evaluasi terpadu untuk ketiga tahap. Pilih tahap lewat --task.

Contoh pakai:
    python src/evaluate.py --task classifier \
        --excel data/dataset.xlsx --images data/images \
        --model models/classifier_baseline.pt

    python src/evaluate.py --task detector \
        --yolo-dir yolo_dataset --model models/runs/tumor_detector/weights/best.pt

    python src/evaluate.py --task segmentation \
        --images data/images --annotations data/Annotations \
        --masks-dir masks --model models/segmentation_unet.pt
"""

import argparse

import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader


def evaluate_classifier(args):
    from dataset import BoneXrayDataset
    from preprocess import get_eval_transform
    from train_classifier import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BoneXrayDataset(args.excel, args.images, transform=get_eval_transform())
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = build_model().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1).cpu()

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.cpu().tolist())

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    print("=== Confusion Matrix ===")
    print(f"                Prediksi Normal   Prediksi Tumor")
    print(f"Aktual Normal   {tn:>15}   {fp:>14}")
    print(f"Aktual Tumor    {fn:>15}   {tp:>14}")
    print()
    print("=== Metrik Klasifikasi ===")
    print(f"Akurasi      : {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Sensitivitas : {sensitivity:.4f}")
    print(f"Spesifisitas : {specificity:.4f}")
    print(f"Precision    : {precision_score(all_labels, all_preds):.4f}")
    print(f"F1-score     : {f1_score(all_labels, all_preds):.4f}")
    print(f"ROC-AUC      : {roc_auc_score(all_labels, all_probs):.4f}")


def evaluate_detector(args):
    from ultralytics import YOLO

    model = YOLO(args.model)
    metrics = model.val(data=f"{args.yolo_dir}/data.yaml", split="test")

    print("=== Metrik Deteksi Objek ===")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall   : {metrics.box.mr:.4f}")


def evaluate_segmentation(args):
    import segmentation_models_pytorch as smp
    from train_segmentation import (BoneSegmentationDataset, dice_coefficient,
                                      get_eval_transform, iou_score)
    from annotations import build_all_masks
    from sklearn.model_selection import train_test_split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_filenames = build_all_masks(args.annotations, args.masks_dir)
    _, temp_files = train_test_split(image_filenames, test_size=0.3, random_state=42)
    _, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    test_ds = BoneSegmentationDataset(test_files, args.images, args.masks_dir, get_eval_transform())
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    total_dice, total_iou, n = 0.0, 0.0, 0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            total_dice += dice_coefficient(outputs, masks).item() * images.size(0)
            total_iou += iou_score(outputs, masks).item() * images.size(0)
            n += images.size(0)

    print("=== Metrik Segmentasi ===")
    print(f"Dice score : {total_dice / n:.4f}")
    print(f"IoU        : {total_iou / n:.4f}")
    print("Pembanding paper BTXRD (konsistensi antar dua dokter): Dice 0.90 | IoU 0.91")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["classifier", "detector", "segmentation"])
    parser.add_argument("--model", required=True, help="Path ke bobot model terlatih")
    # argumen untuk classifier
    parser.add_argument("--excel", help="[classifier] path ke dataset.xlsx")
    parser.add_argument("--images", help="[classifier/segmentation] folder citra")
    # argumen untuk detector
    parser.add_argument("--yolo-dir", help="[detector] folder dataset format YOLO (hasil train_detector.py)")
    # argumen untuk segmentation
    parser.add_argument("--annotations", help="[segmentation] folder anotasi LabelMe JSON")
    parser.add_argument("--masks-dir", default="masks", help="[segmentation] folder cache mask biner")
    args = parser.parse_args()

    if args.task == "classifier":
        evaluate_classifier(args)
    elif args.task == "detector":
        evaluate_detector(args)
    elif args.task == "segmentation":
        evaluate_segmentation(args)


if __name__ == "__main__":
    main()
