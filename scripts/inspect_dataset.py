"""
inspect_dataset.py — Veri Setini Inceleme Script'i

Bu dosya DACL10K veri setini kesfetmek icin kullanilir.
Henuz model egitmeden verinin nasil gorundugunu anlamaya yarar.

Kullanim:
    python scripts/inspect_dataset.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import Config
from src.utils import log
from src.dataset import DACL10K_CLASSES, NUM_CLASSES


def main():
    """Veri setini inceler ve ozet bilgi yazdirir."""

    log("=" * 50)
    log("DACL10K Veri Seti Incelemesi")
    log("=" * 50)

    # 1. Sinif bilgilerini goster
    print(f"\nToplam sinif sayisi: {NUM_CLASSES}")
    print("\nSiniflar:")
    for i, cls in enumerate(DACL10K_CLASSES):
        print(f"  {i:2d}. {cls}")

    # 2. Veri seti klasorunu kontrol et
    print(f"\nVeri seti klasoru: {Config.DATA_DIR}")

    if not os.path.exists(Config.DATA_DIR):
        print(f"Veri seti klasoru bulunamadi: {Config.DATA_DIR}")
        print("Lutfen DACL10K veri setini 'data/dacl10k/' klasorune yerlestirin.")
        print("\nBeklenen yapi:")
        print("  data/dacl10k/annotations/train/*.npy")
        print("  data/dacl10k/annotations/validation/*.npy")
        print("  data/dacl10k/images/train/*.npy")
        print("  data/dacl10k/images/validation/*.npy")
        return

    # 3. Egitim verisini kontrol et
    print("\n--- Egitim Verisi ---")
    _check_split("train", Config.TRAIN_IMAGES, Config.TRAIN_MASKS)

    # 4. Validation verisini kontrol et
    print("\n--- Validation Verisi ---")
    _check_split("validation", Config.VAL_IMAGES, Config.VAL_MASKS)

    log("\nInceleme tamamlandi.")


def _check_split(split_name, images_dir, masks_dir):
    """Bir split'in (train/validation) icerigini kontrol eder."""

    # Gorsel klasoru
    print(f"\nGorsel klasoru: {images_dir}")
    if not os.path.exists(images_dir):
        print("  HATA: Klasor bulunamadi!")
        return

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".npy")])
    print(f"  Dosya sayisi: {len(image_files)}")

    # Mask klasoru
    print(f"Mask klasoru: {masks_dir}")
    if not os.path.exists(masks_dir):
        print("  HATA: Klasor bulunamadi!")
        return

    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(".npy")])
    print(f"  Dosya sayisi: {len(mask_files)}")

    # Dosya sayilari eslesme kontrolu
    if len(image_files) != len(mask_files):
        print(f"  UYARI: Gorsel ({len(image_files)}) ve mask ({len(mask_files)}) sayisi farkli!")

    if not image_files:
        return

    # Ilk dosyayi oku ve boyutlarini goster
    first_img = np.load(os.path.join(images_dir, image_files[0]))
    first_mask = np.load(os.path.join(masks_dir, mask_files[0]))

    print(f"\nOrnek dosya: {image_files[0]}")
    print(f"  Gorsel boyutu : {first_img.shape} dtype={first_img.dtype}")
    print(f"  Mask boyutu   : {first_mask.shape} dtype={first_mask.dtype}")
    print(f"  Gorsel deger araligi: [{first_img.min()}, {first_img.max()}]")
    print(f"  Mask benzersiz degerler: {np.unique(first_mask)}")

    # Mask kanallarindaki sinif dagilimi (ilk 100 dosya)
    print(f"\nSinif dagilimi (ilk 100 {split_name} dosyasi uzerinden):")
    class_counts = np.zeros(NUM_CLASSES, dtype=int)
    sample_count = min(100, len(mask_files))

    for filename in mask_files[:sample_count]:
        mask = np.load(os.path.join(masks_dir, filename))
        # Her kanalda en az bir 1 var mi?
        for c in range(NUM_CLASSES):
            if mask[:, :, c].any():
                class_counts[c] += 1

    for i, cls in enumerate(DACL10K_CLASSES):
        bar = "#" * (class_counts[i] * 40 // sample_count) if class_counts[i] > 0 else ""
        print(f"  {cls:20s}: {class_counts[i]:3d}/{sample_count}  {bar}")


if __name__ == "__main__":
    main()
