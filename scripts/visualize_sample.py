"""
visualize_sample.py — Veri Setinden Ornek Gorsellestirme

Bu script DACL10K veri setinden rastgele ornekler alip
gorsel + mask overlay olarak kaydeder.

Kullanim:
    python scripts/visualize_sample.py
    python scripts/visualize_sample.py --index 42
    python scripts/visualize_sample.py --count 5
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from src.config import Config
from src.dataset import DACL10K_CLASSES, NUM_CLASSES
from src.utils import log, ensure_dir


# Her sinif icin bir renk (RGB)
CLASS_COLORS = [
    (255, 0, 0),       # Crack — kirmizi
    (255, 100, 0),     # ACrack — turuncu
    (255, 255, 0),     # Spalling — sari
    (200, 200, 0),     # Efflorescence — koyu sari
    (0, 255, 0),       # ExposedRebars — yesil
    (0, 200, 100),     # Cavity — deniz yesili
    (0, 255, 255),     # Restformwork — cyan
    (0, 150, 255),     # Rockpocket — acik mavi
    (0, 0, 255),       # Hollowareas — mavi
    (150, 0, 255),     # Rust — mor
    (255, 0, 255),     # Weathering — pembe
    (255, 0, 150),     # Graffiti — koyu pembe
    (128, 128, 0),     # Wetspot — zeytin
    (128, 0, 0),       # Bearing — koyu kirmizi
    (0, 128, 0),       # Drainage — koyu yesil
    (0, 0, 128),       # EJoint — lacivert
    (128, 128, 128),   # JTape — gri
    (200, 150, 100),   # PEquipment — bej
    (100, 50, 0),      # WConccor — kahverengi
]


def create_overlay(image, mask, alpha=0.4):
    """
    Gorsel uzerine mask overlay olusturur.

    Args:
        image: (512, 512, 3) uint8 numpy array
        mask: (512, 512, 19) uint8 numpy array
        alpha: overlay seffafligi (0=gorsel, 1=mask)

    Returns:
        (512, 512, 3) uint8 numpy overlay goruntusu
    """
    overlay = image.copy().astype(np.float32)

    for c in range(NUM_CLASSES):
        channel_mask = mask[:, :, c]  # (512, 512)
        if not channel_mask.any():
            continue

        color = np.array(CLASS_COLORS[c], dtype=np.float32)
        # Mask olan piksellere renk karıstir
        for ch in range(3):
            overlay[:, :, ch] = np.where(
                channel_mask > 0,
                overlay[:, :, ch] * (1 - alpha) + color[ch] * alpha,
                overlay[:, :, ch]
            )

    return overlay.astype(np.uint8)


def create_legend(active_classes):
    """Aktif siniflarin renk aciklamasini iceren bir gorsel olusturur."""
    line_height = 25
    width = 250
    height = line_height * len(active_classes) + 10
    legend = Image.new("RGB", (width, height), (255, 255, 255))

    from PIL import ImageDraw
    draw = ImageDraw.Draw(legend)

    for i, (cls_idx, cls_name) in enumerate(active_classes):
        y = 5 + i * line_height
        color = CLASS_COLORS[cls_idx]
        # Renk kutusu
        draw.rectangle([5, y, 25, y + 18], fill=color)
        # Sinif adi
        draw.text((30, y), cls_name, fill=(0, 0, 0))

    return legend


def visualize_one(index, images_dir, masks_dir, output_dir):
    """Tek bir ornegi gorsellestirip kaydeder."""
    files = sorted([f for f in os.listdir(images_dir) if f.endswith(".npy")])

    if index >= len(files):
        print(f"HATA: index {index} cok buyuk (toplam {len(files)} dosya)")
        return

    filename = files[index]
    image = np.load(os.path.join(images_dir, filename))
    mask = np.load(os.path.join(masks_dir, filename))

    # Hangi siniflar bu gorselde var?
    active = []
    for c in range(NUM_CLASSES):
        if mask[:, :, c].any():
            active.append((c, DACL10K_CLASSES[c]))

    log(f"  {filename}: {[a[1] for a in active]}")

    # Overlay olustur
    overlay = create_overlay(image, mask)

    # Gorselleri yan yana birlestir
    h, w = image.shape[:2]
    combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
    combined[:, :w, :] = image       # Sol: orijinal
    combined[:, w:, :] = overlay     # Sag: overlay

    # Kaydet
    base_name = filename.replace(".npy", "")
    out_path = os.path.join(output_dir, f"{base_name}_viz.png")
    Image.fromarray(combined).save(out_path)
    log(f"  Kaydedildi: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="DACL10K ornek gorsellestirme")
    parser.add_argument("--index", type=int, default=None, help="Belirli bir gorselin indeksi")
    parser.add_argument("--count", type=int, default=3, help="Rastgele gorsel sayisi (varsayilan: 3)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"],
                        help="Hangi split (varsayilan: train)")
    args = parser.parse_args()

    # Klasorleri ayarla
    if args.split == "train":
        images_dir = Config.TRAIN_IMAGES
        masks_dir = Config.TRAIN_MASKS
    else:
        images_dir = Config.VAL_IMAGES
        masks_dir = Config.VAL_MASKS

    output_dir = os.path.join(Config.OUTPUT_DIR, "visualizations")
    ensure_dir(output_dir)

    log("=" * 50)
    log("DACL10K Gorsellestirme")
    log("=" * 50)

    files = sorted([f for f in os.listdir(images_dir) if f.endswith(".npy")])
    total = len(files)

    if args.index is not None:
        # Belirli bir gorsel
        indices = [args.index]
    else:
        # Rastgele secim
        np.random.seed(42)
        indices = np.random.choice(total, size=min(args.count, total), replace=False)
        indices = sorted(indices)

    log(f"Split: {args.split}, Toplam: {total}, Secilen: {len(indices)}")

    for idx in indices:
        visualize_one(idx, images_dir, masks_dir, output_dir)

    log(f"\nTum gorseller kaydedildi: {output_dir}")


if __name__ == "__main__":
    main()
