"""
Dataset loader untuk klasifikasi (Tahap 1): normal vs tumor.

Membaca metadata dari dataset.xlsx (format asli BTXRD), bukan CSV.
Kolom yang dipakai:
    - image_id  -> nama file citra (mis. IMG000001.jpeg)
    - tumor     -> 0 = normal, 1 = tumor

Kolom lain di dataset.xlsx (benign, malignant, lokasi anatomi one-hot,
jenis tumor one-hot) tidak dipakai di Tahap 1, tapi tetap bisa diakses
lewat `df_full` untuk analisis lanjutan atau klasifikasi jinak/ganas.
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BoneXrayDataset(Dataset):
    """Dataset klasifikasi normal vs tumor dari BTXRD."""

    def __init__(self, excel_path, images_dir, transform=None):
        self.df_full, self.df = self._load_labels(excel_path)
        self.images_dir = images_dir
        self.transform = transform

    def _load_labels(self, excel_path):
        df_full = pd.read_excel(excel_path)

        required_cols = {"image_id", "tumor"}
        if not required_cols.issubset(df_full.columns):
            raise ValueError(
                f"dataset.xlsx harus punya kolom {required_cols}, "
                f"ditemukan: {list(df_full.columns)}. "
                "Sesuaikan nama kolom jika formatmu berbeda."
            )

        df = df_full.rename(columns={"image_id": "filename", "tumor": "label"})
        df = df[["filename", "label"]].reset_index(drop=True)
        return df_full, df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label

    def class_counts(self):
        """Cek ketidakseimbangan kelas (normal vs tumor)."""
        return self.df["label"].value_counts().to_dict()
