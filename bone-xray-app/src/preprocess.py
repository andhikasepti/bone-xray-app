"""
Praproses & augmentasi citra untuk klasifikasi (Tahap 1).

Augmentasi dipilih yang "aman" secara medis: flip, rotasi kecil, sedikit
perubahan kontras/kecerahan. Dihindari augmentasi yang bisa mengubah
makna klinis citra (distorsi bentuk berlebihan, dsb).
"""

from torchvision import transforms

IMAGE_SIZE = 224  # ukuran standar untuk model pralatih ImageNet

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform():
    """Dipakai untuk data validasi/test -- tanpa augmentasi."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
