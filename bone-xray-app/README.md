# Bone X-ray Screening Tool (versi belajar)

Pipeline deep learning untuk klasifikasi, deteksi objek, dan segmentasi
tumor tulang pada citra X-ray, memakai dataset publik **BTXRD** (*Bone
Tumor X-ray Radiograph dataset*, Scientific Data, 2024). Dibuat sebagai
bahan belajar sekaligus contoh kasus untuk buku *Deteksi Tumor Tulang
pada Citra X-ray dengan Deep Learning*.

⚠️ **Disclaimer:** proyek ini untuk tujuan edukasi dan riset. **Bukan
alat diagnosis medis.**

## Struktur Project

```
bone-xray-app/
├── data/
│   ├── dataset.xlsx          # metadata BTXRD (kolom: image_id, tumor, benign, malignant, dst.)
│   ├── images/                # citra X-ray (.jpeg)
│   └── Annotations/            # anotasi LabelMe JSON (rectangle=bbox, polygon=mask)
├── notebooks/                 # notebook eksplorasi/eksperimen dari Google Colab (.ipynb)
├── models/                    # bobot model terlatih (dibuat otomatis oleh skrip training)
├── src/
│   ├── dataset.py              # dataset loader PyTorch (Tahap 1)
│   ├── preprocess.py           # transform & augmentasi (Tahap 1)
│   ├── annotations.py          # utilitas parsing LabelMe JSON (dipakai Tahap 2 & 3)
│   ├── train_classifier.py     # Tahap 1: training klasifikasi normal/tumor
│   ├── train_detector.py       # Tahap 2: training deteksi objek (YOLOv8)
│   ├── train_segmentation.py   # Tahap 3: training segmentasi (U-Net)
│   └── evaluate.py             # evaluasi terpadu (--task classifier/detector/segmentation)
├── app.py                      # aplikasi Streamlit (gabungan ketiga tahap)
├── requirements.txt
└── README.md
```

## Tiga Tahap

| Tahap | Teknik | Output | Skrip |
|---|---|---|---|
| 1 | Klasifikasi | Normal vs Tumor | `train_classifier.py` |
| 2 | Deteksi objek | Bounding box lokasi tumor | `train_detector.py` |
| 3 | Segmentasi | Mask area tumor presisi | `train_segmentation.py` |

Ketiganya memakai **transfer learning** (ResNet18 untuk Tahap 1 & 3,
YOLOv8n untuk Tahap 2) dan **satu kelas** (`tumor`) -- fokus
melokalisasi/memetakan tumor, bukan mengklasifikasikan jenisnya.

## Riwayat Pengembangan

Ketiga tahap awalnya dikembangkan dan diuji coba secara eksploratif di
**Google Colab** (lihat `notebooks/`), lalu dirapikan jadi skrip modular
di `src/` supaya:
- bisa dijalankan ulang secara konsisten (bukan cell-per-cell manual)
- lebih mudah di-review dan dikembangkan bersama co-author lewat GitHub
- komponennya (dataset, parsing anotasi, training) bisa dipakai ulang antar tahap

Hasil eksperimen awal (baseline, data & split apa adanya):
- Klasifikasi: val_acc ~0.69 (epoch awal, lihat log training untuk angka final)
- Deteksi: lihat output `train_detector.py` (mAP50, precision, recall)
- Segmentasi: Val Dice ~0.65 (pembanding paper: konsistensi antar dua dokter Dice 0.90)

Catatan dari eksperimen: ada indikasi overfitting pada segmentasi baseline
(train_loss terus turun, val_dice mentok) -- sudah ditangani dengan
augmentasi (albumentations) dan early stopping di `train_segmentation.py`.

## Cara Menjalankan

### 1. Setup environment

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Siapkan data

Unduh BTXRD (cek lisensi & ketentuan sitasi), lalu susun sesuai struktur
di atas:
- `data/dataset.xlsx` -- metadata (kolom wajib: `image_id`, `tumor`)
- `data/images/` -- semua citra
- `data/Annotations/` -- file JSON LabelMe (hanya ada untuk citra tumor)

### 3. Training Tahap 1: Klasifikasi

```bash
python src/train_classifier.py \
    --excel data/dataset.xlsx \
    --images data/images \
    --epochs 10 \
    --out models/classifier_baseline.pt
```

### 4. Training Tahap 2: Deteksi Objek

```bash
python src/train_detector.py \
    --images data/images \
    --annotations data/Annotations \
    --epochs 50 \
    --project models/runs --name tumor_detector
```

Bobot terbaik tersimpan di `models/runs/tumor_detector/weights/best.pt`.

### 5. Training Tahap 3: Segmentasi

```bash
python src/train_segmentation.py \
    --images data/images \
    --annotations data/Annotations \
    --epochs 50 --patience 8 \
    --out models/segmentation_unet.pt
```

Skrip ini otomatis melanjutkan dari checkpoint di `--out` jika sudah ada
(berguna kalau training sempat terputus).

### 6. Evaluasi

```bash
python src/evaluate.py --task classifier --excel data/dataset.xlsx --images data/images --model models/classifier_baseline.pt
python src/evaluate.py --task detector --yolo-dir yolo_dataset --model models/runs/tumor_detector/weights/best.pt
python src/evaluate.py --task segmentation --images data/images --annotations data/Annotations --model models/segmentation_unet.pt
```

### 7. Jalankan aplikasi

```bash
streamlit run app.py
```

Aplikasi akan memuat model yang tersedia di `models/` secara otomatis;
tahap yang modelnya belum ada akan ditandai "belum tersedia".

## Catatan Penting untuk Bab 9 (Buku)

- **Pemisahan data per pasien:** jika BTXRD punya ID pasien, pisahkan
  train/val/test berdasarkan pasien (bukan per citra) untuk menghindari
  data leakage. Split saat ini masih acak per-citra.
- **Ketidakseimbangan kelas:** cek `dataset.class_counts()` sebelum
  training; pertimbangkan class weighting/oversampling bila timpang.
- **Sensitivitas/recall vs spesifisitas/precision:** dalam konteks
  skrining medis, melewatkan kasus tumor (sensitivitas/recall rendah)
  biasanya lebih berisiko daripada false alarm.
- **Konteks angka segmentasi:** Dice model (vs satu ground truth) tidak
  satu level dengan Dice konsistensi antar-dokter (manusia vs manusia)
  di paper asli -- keduanya tetap berguna sebagai acuan relatif.

## Kontribusi (untuk co-author)

1. Clone repo, buat branch baru untuk tiap bab/eksperimen.
2. Notebook eksplorasi baru taruh di `notebooks/`; kode yang sudah stabil
   dipindah ke `src/` sebagai fungsi/skrip supaya bisa dipakai ulang.
3. Jangan commit file di `data/images/`, `data/Annotations/`, maupun
   `models/*.pt` (sudah di-exclude lewat `.gitignore`) -- file ini besar
   dan sebaiknya dibagikan lewat Drive/link unduhan terpisah.
