"""
Utilitas bersama untuk membaca anotasi LabelMe JSON (folder data/Annotations).

Format tiap file JSON (satu file per citra tumor):
    {
      "shapes": [
        {"label": "...", "shape_type": "rectangle", "points": [[x1,y1],[x2,y2]]},
        {"label": "...", "shape_type": "polygon",   "points": [[x,y], ...]}
      ],
      "imagePath": "IMG000001.jpeg",
      "imageWidth": ..., "imageHeight": ...
    }

- shape "rectangle" -> dipakai untuk deteksi objek (Tahap 2 / YOLO)
- shape "polygon"   -> dipakai untuk segmentasi (Tahap 3 / U-Net)

Semua tumor diperlakukan sebagai SATU kelas ("tumor") untuk Tahap 2 & 3,
konsisten dengan tujuan buku (melokalisasi & memetakan tumor, bukan
mengklasifikasikan jenisnya). Field `label` per shape (mis. "other mt")
tetap tersimpan jika suatu saat ingin dikembangkan ke deteksi/segmentasi
multi-kelas.
"""

import glob
import json
import os

from PIL import Image, ImageDraw


def load_annotation(json_path):
    with open(json_path) as f:
        return json.load(f)


def get_rectangles(ann):
    """Ambil semua bounding box (shape rectangle) dari satu anotasi."""
    boxes = []
    for shape in ann["shapes"]:
        if shape["shape_type"] != "rectangle":
            continue
        (x1, y1), (x2, y2) = shape["points"]
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        boxes.append({
            "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max,
            "label": shape["label"],
        })
    return boxes


def polygon_shapes_to_mask(ann):
    """Gabungkan semua shape polygon dalam satu anotasi jadi satu mask biner (PIL 'L')."""
    w, h = ann["imageWidth"], ann["imageHeight"]
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    has_polygon = False
    for shape in ann["shapes"]:
        if shape["shape_type"] != "polygon":
            continue
        points = [tuple(p) for p in shape["points"]]
        draw.polygon(points, outline=1, fill=1)
        has_polygon = True

    return mask, has_polygon


def parse_all_annotations(annotations_dir):
    """
    Baca semua file JSON di folder anotasi.
    Return list of dict: {image_filename, img_width, img_height, boxes}
    Hanya menyertakan citra yang punya minimal satu bounding box rectangle.
    """
    json_files = glob.glob(os.path.join(annotations_dir, "*.json"))
    parsed = []

    for jp in json_files:
        ann = load_annotation(jp)
        boxes = get_rectangles(ann)
        if not boxes:
            continue
        parsed.append({
            "image_filename": ann["imagePath"],
            "img_width": ann["imageWidth"],
            "img_height": ann["imageHeight"],
            "boxes": boxes,
        })

    return parsed


def to_yolo_line(box, img_w, img_h, class_id=0):
    """Konversi satu bounding box ke format YOLO (ternormalisasi 0-1)."""
    x_center = (box["x_min"] + box["x_max"]) / 2 / img_w
    y_center = (box["y_min"] + box["y_max"]) / 2 / img_h
    width = (box["x_max"] - box["x_min"]) / img_w
    height = (box["y_max"] - box["y_min"]) / img_h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def build_all_masks(annotations_dir, masks_dir):
    """
    Generate mask biner (PNG) dari semua shape polygon, simpan ke masks_dir.
    Return list nama file citra yang punya mask.
    """
    os.makedirs(masks_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(annotations_dir, "*.json"))
    filenames = []

    for jp in json_files:
        ann = load_annotation(jp)
        mask, has_polygon = polygon_shapes_to_mask(ann)
        if not has_polygon:
            continue

        base = os.path.splitext(ann["imagePath"])[0]
        mask.save(os.path.join(masks_dir, base + ".png"))
        filenames.append(ann["imagePath"])

    return filenames
