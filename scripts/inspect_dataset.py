"""
inspect_dataset.py — Veri Setini İnceleme Script'i

Bu dosya DACL10K veri setini keşfetmek için kullanılır.
Henüz model eğitmeden verinin nasıl göründüğünü anlamaya yarar.

Kullanım:
    python scripts/inspect_dataset.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils import log
from src.dataset import DACL10K_CLASSES


def main():
    """Veri setini inceler ve özet bilgi yazdırır."""

    log("=" * 50)
    log("DACL10K Veri Seti İncelemesi")
    log("=" * 50)

    # 1. Sınıf bilgilerini göster
    print(f"\nToplam sınıf sayısı: {len(DACL10K_CLASSES)}")
    print("\nSınıflar:")
    for i, cls in enumerate(DACL10K_CLASSES):
        print(f"  {i+1:2d}. {cls}")

    # 2. Annotation dosyasını kontrol et
    ann_path = Config.TRAIN_ANNOTATIONS
    print(f"\nAnnotation dosyası: {ann_path}")

    if os.path.exists(ann_path):
        # İlk birkaç annotation'ı oku ve özetle
        with open(ann_path, "r") as f:
            lines = f.readlines()

        print(f"Toplam annotation sayısı: {len(lines)}")

        # İlk annotation'ın yapısını göster
        if lines:
            first_ann = json.loads(lines[0])
            print(f"\nİlk annotation örneği:")
            print(f"  Görsel adı  : {first_ann.get('imageName', 'Bilinmiyor')}")
            print(f"  Boyut       : {first_ann.get('imageWidth', '?')} x {first_ann.get('imageHeight', '?')}")
            print(f"  Shape sayısı: {len(first_ann.get('shapes', []))}")

            # Hangi sınıflar var?
            labels = [s["label"] for s in first_ann.get("shapes", [])]
            print(f"  Sınıflar    : {labels}")
    else:
        print(f"Annotation dosyası bulunamadı: {ann_path}")
        print("Lütfen DACL10K veri setini 'data/' klasörüne yerleştirin.")

    # 3. Görsellerin varlığını kontrol et
    images_dir = Config.DATA_DIR
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"\nGörsel klasöründe {len(image_files)} adet görsel bulundu.")
    else:
        print(f"\nGörsel klasörü bulunamadı: {images_dir}")

    log("İnceleme tamamlandı.")


if __name__ == "__main__":
    main()
