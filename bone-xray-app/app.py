"""
Bone X-ray Screening Tool (versi belajar) -- gabungan Tahap 1-3.

Alur: unggah citra -> klasifikasi (normal/tumor) -> jika tumor, jalankan
deteksi (bounding box) dan segmentasi (mask area tumor).

Jalankan dengan:
    streamlit run app.py

CATATAN PENTING: aplikasi ini dibuat untuk tujuan edukasi dan riset dalam
menyusun buku. Ini BUKAN alat diagnosis medis dan tidak boleh dipakai
untuk pengambilan keputusan klinis.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import streamlit as st
import torch
from PIL import Image

CLASSIFIER_PATH = "models/classifier_baseline.pt"
DETECTOR_PATH = "models/runs/tumor_detector/weights/best.pt"
SEGMENTATION_PATH = "models/segmentation_unet.pt"

CLASS_NAMES = ["Normal", "Tumor"]

st.set_page_config(page_title="Bone X-ray Screening Tool (Belajar)", layout="centered")


@st.cache_resource
def load_classifier():
    from train_classifier import build_model
    from preprocess import get_eval_transform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    loaded = False
    if os.path.exists(CLASSIFIER_PATH):
        model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
        loaded = True
    model.eval().to(device)
    return model, get_eval_transform(), device, loaded


@st.cache_resource
def load_detector():
    if not os.path.exists(DETECTOR_PATH):
        return None
    from ultralytics import YOLO
    return YOLO(DETECTOR_PATH)


@st.cache_resource
def load_segmentation():
    if not os.path.exists(SEGMENTATION_PATH):
        return None, None, None
    import segmentation_models_pytorch as smp
    from train_segmentation import get_eval_transform as seg_eval_transform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(SEGMENTATION_PATH, map_location=device))
    model.eval().to(device)
    return model, seg_eval_transform(), device


def predict_classification(image):
    model, transform, device, loaded = load_classifier()
    if not loaded:
        return None, None, None
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()
    return CLASS_NAMES[pred_idx], probs[pred_idx].item(), probs.tolist()


def predict_segmentation(image):
    model, transform, device = load_segmentation()
    if model is None:
        return None
    tensor = transform(image=np.array(image))["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor)
        pred_mask = (torch.sigmoid(pred) > 0.5).float().cpu().squeeze()
    return pred_mask


def main():
    st.title("🦴 Bone X-ray Screening Tool")
    st.caption("Versi belajar -- klasifikasi, deteksi, dan segmentasi tumor tulang")

    st.warning(
        "⚠️ Alat ini dibuat untuk tujuan edukasi dan riset penyusunan buku. "
        "**Bukan alat diagnosis medis.** Hasil prediksi tidak menggantikan "
        "penilaian dokter."
    )

    _, _, _, clf_loaded = load_classifier()
    detector = load_detector()

    missing = []
    if not clf_loaded:
        missing.append(f"- Klasifikasi: `{CLASSIFIER_PATH}` belum ada (jalankan `train_classifier.py`)")
    if detector is None:
        missing.append(f"- Deteksi: `{DETECTOR_PATH}` belum ada (jalankan `train_detector.py`)")
    if load_segmentation()[0] is None:
        missing.append(f"- Segmentasi: `{SEGMENTATION_PATH}` belum ada (jalankan `train_segmentation.py`)")
    if missing:
        st.info("Beberapa model belum tersedia:\n" + "\n".join(missing))

    uploaded_file = st.file_uploader("Unggah citra X-ray tulang (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Citra yang diunggah", use_column_width=True)

    if not st.button("Analisis Citra"):
        return

    # --- Tahap 1: Klasifikasi ---
    label, confidence, all_probs = predict_classification(image)
    if label is None:
        st.error("Model klasifikasi belum tersedia.")
        return

    st.subheader("1. Klasifikasi")
    st.metric("Prediksi", label, f"{confidence * 100:.1f}% keyakinan")
    for name, prob in zip(CLASS_NAMES, all_probs):
        st.progress(prob, text=f"{name}: {prob * 100:.1f}%")

    if label == "Normal":
        st.success("Model memprediksi citra ini normal -- deteksi & segmentasi dilewati.")
        return

    # --- Tahap 2: Deteksi ---
    st.subheader("2. Deteksi Objek (Bounding Box)")
    if detector is None:
        st.info("Model deteksi belum tersedia.")
    else:
        results = detector.predict(np.array(image), conf=0.25, verbose=False)
        pred = results[0]
        img_with_boxes = pred.plot()[:, :, ::-1]
        st.image(img_with_boxes, caption=f"{len(pred.boxes)} area terdeteksi")

    # --- Tahap 3: Segmentasi ---
    st.subheader("3. Segmentasi (Area Tumor)")
    pred_mask = predict_segmentation(image)
    if pred_mask is None:
        st.info("Model segmentasi belum tersedia.")
    else:
        import matplotlib.pyplot as plt

        img_resized = image.resize((256, 256))
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_resized)
        axes[0].set_title("Citra Asli")
        axes[1].imshow(img_resized)
        axes[1].imshow(pred_mask, cmap="Reds", alpha=0.5)
        axes[1].set_title("Area Tumor Terdeteksi")
        for ax in axes:
            ax.axis("off")
        st.pyplot(fig)

        area_percent = pred_mask.mean().item() * 100
        st.caption(f"Perkiraan luas area tumor: {area_percent:.2f}% dari citra")


if __name__ == "__main__":
    main()
