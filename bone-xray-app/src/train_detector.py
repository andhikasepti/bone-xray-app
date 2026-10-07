"""
Tahap 2: Deteksi objek (bounding box tumor) dengan YOLOv8 (Ultralytics).

Membangun dataset format YOLO dari anotasi LabelMe (shape rectangle),
lalu melatih YOLOv8n sebagai baseline.

Cara pakai:
    python src/train_detector.py \
        --images data/images \
        --annotations data/Annotations \
        --epochs 50 \
        --yolo-dir yolo_dataset \
        --project models/runs --name tumor_detector
"""

import argparse
import os
import shutil

from sklearn.model_selection import train_test_split
from ultralytics import YOLO

from annotations import parse_all_annotations, to_yolo_line


def build_split(data, split_name, images_dir, yolo_dir):
    os.makedirs(f"{yolo_dir}/images/{split_name}", exist_ok=True)
    os.makedirs(f"{yolo_dir}/labels/{split_name}", exist_ok=True)

    skipped = 0
    for item in data:
        src_img = os.path.join(images_dir, item["image_filename"])
        if not os.path.exists(src_img):
            skipped += 1
            continue

        dst_img = f"{yolo_dir}/images/{split_name}/{item['image_filename']}"
        shutil.copy(src_img, dst_img)

        label_filename = os.path.splitext(item["image_filename"])[0] + ".txt"
        dst_label = f"{yolo_dir}/labels/{split_name}/{label_filename}"
        lines = [to_yolo_line(b, item["img_width"], item["img_height"]) for b in item["boxes"]]
        with open(dst_label, "w") as f:
            f.write("\n".join(lines))

    if skipped:
        print(f"[{split_name}] {skipped} citra dilewati (file gambar tidak ditemukan)")
    print(f"[{split_name}] selesai: {len(data) - skipped} citra disalin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Folder citra (data/images)")
    parser.add_argument("--annotations", required=True, help="Folder anotasi LabelMe JSON (data/Annotations)")
    parser.add_argument("--yolo-dir", default="yolo_dataset", help="Folder sementara untuk dataset format YOLO")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--project", default="models/runs")
    parser.add_argument("--name", default="tumor_detector")
    args = parser.parse_args()

    # 1. Parse semua anotasi, hanya citra dengan minimal satu bounding box
    parsed = parse_all_annotations(args.annotations)
    print(f"Jumlah citra dengan minimal satu bounding box: {len(parsed)}")

    # 2. Split train/val/test
    train_data, temp_data = train_test_split(parsed, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # 3. Bangun dataset format YOLO
    build_split(train_data, "train", args.images, args.yolo_dir)
    build_split(val_data, "val", args.images, args.yolo_dir)
    build_split(test_data, "test", args.images, args.yolo_dir)

    data_yaml_path = f"{args.yolo_dir}/data.yaml"
    with open(data_yaml_path, "w") as f:
        f.write(f"""
path: {os.path.abspath(args.yolo_dir)}
train: images/train
val: images/val
test: images/test

names:
  0: tumor
""")

    # 4. Training
    model = YOLO("yolov8n.pt")  # bobot pralatih COCO, transfer learning
    model.train(
        data=data_yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=args.patience,
    )

    # 5. Evaluasi di test set
    metrics = model.val(data=data_yaml_path, split="test")
    print("\n=== Hasil Evaluasi pada Test Set ===")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall   : {metrics.box.mr:.4f}")
    print(f"\nBobot model terbaik: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
