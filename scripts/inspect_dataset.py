"""
inspect_dataset.py — Veri Setini İnceleme Script'i

Bu dosya DACL10K veri setini keşfetmek için kullanılır.
Henüz model eğitmeden verinin nasıl göründüğünü anlamaya yarar.

Kullanım:
    python scripts/inspect_dataset.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.dataset import DACL10K_CLASSES, NUM_CLASSES
from src.utils import log


def main():
    """Veri setini inceler ve özet bilgi yazar."""

    log("=" * 55)
    log("DACL10K Veri Seti İncelemesi")
    log("=" * 55)

    # 1. Sınıf bilgilerini göster
    print(f"\nToplam sınıf sayısı: {NUM_CLASSES}")
    print("\nSınıflar:")
    for i, cls in enumerate(DACL10K_CLASSES):
        print(f"  {i:2d}. {cls}")

    # 2. Veri seti klasörünü kontrol et
    print(f"\nVeri seti klasörü: {Config.DATA_DIR}")

    if not os.path.exists(Config.DATA_DIR):
        print(f"\nVeri seti klasörü bulunamadı: {Config.DATA_DIR}")
        print("\nVeri setini kurmak için:")
        print("  1. pip install dataset-tools")
        print("  2. python scripts/setup_dataset.py")
        print("\nBeklenen yapı:")
        print("  data/dacl10k/images/train/*.jpg")
        print("  data/dacl10k/images/validation/*.jpg")
        print("  data/dacl10k/annotations/train/*.json")
        print("  data/dacl10k/annotations/validation/*.json")
        return

    # 3. Eğitim verisini kontrol et
    print("\n--- Eğitim Verisi ---")
    _split_kontrol("train", Config.TRAIN_IMAGES, Config.TRAIN_MASKS)

    # 4. Validation verisini kontrol et
    print("\n--- Validation Verisi ---")
    _split_kontrol("validation", Config.VAL_IMAGES, Config.VAL_MASKS)

    log("\nİnceleme tamamlandı.")


def _sinif_dagilimi_hesapla(annotations_dir, json_dosyalar):
    """İlk 100 annotation dosyasındaki sınıf dağılımını sayar."""
    import numpy as np
    sinif_sayilari = np.zeros(NUM_CLASSES, dtype=int)
    ornek_sayisi = min(100, len(json_dosyalar))
    for json_dosyasi in json_dosyalar[:ornek_sayisi]:
        json_yolu = os.path.join(annotations_dir, json_dosyasi)
        with open(json_yolu, "r", encoding="utf-8") as f:
            ann = json.load(f)
        for sekil in ann.get("shapes", []):
            sinif_adi = sekil.get("label", "")
            if sinif_adi in DACL10K_CLASSES:
                sinif_sayilari[DACL10K_CLASSES.index(sinif_adi)] += 1
    return sinif_sayilari


def _split_kontrol(split_adi, images_dir, annotations_dir):
    """Bir split'in (train/validation) içeriğini kontrol eder."""

    # Görsel klasörü
    print(f"\nGörsel klasörü: {images_dir}")
    if not os.path.exists(images_dir):
        print("  HATA: Klasör bulunamadı!")
        return

    gorsel_dosyalar = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(".jpg") or f.lower().endswith(".jpeg")
    ])
    print(f"  Dosya sayısı: {len(gorsel_dosyalar)}")

    # Annotation klasörü
    print(f"Annotation klasörü: {annotations_dir}")
    if not os.path.exists(annotations_dir):
        print("  HATA: Klasör bulunamadı!")
        return

    json_dosyalar = sorted([
        f for f in os.listdir(annotations_dir)
        if f.endswith(".json")
    ])
    print(f"  Dosya sayısı: {len(json_dosyalar)}")

    # Dosya sayıları eşleşme kontrolü
    if len(gorsel_dosyalar) != len(json_dosyalar):
        print(f"  UYARI: Görsel ({len(gorsel_dosyalar)}) ve annotation ({len(json_dosyalar)}) sayısı farklı!")

    if not gorsel_dosyalar:
        return

    # İlk dosyayı oku ve boyutlarını göster
    from PIL import Image
    import numpy as np
    from src.dataset import _json_annotation_to_mask

    ornek_gorsel_yolu = os.path.join(images_dir, gorsel_dosyalar[0])
    gorsel = Image.open(ornek_gorsel_yolu)
    genislik, yukseklik = gorsel.size

    # İlk JSON annotation'ı oku
    temel_ad = os.path.splitext(gorsel_dosyalar[0])[0]
    ornek_json_yolu = os.path.join(annotations_dir, temel_ad + ".json")

    print(f"\nÖrnek dosya: {gorsel_dosyalar[0]}")
    print(f"  Görsel boyutu: {yukseklik}x{genislik} (YxG), mod={gorsel.mode}")

    if os.path.exists(ornek_json_yolu):
        mask = _json_annotation_to_mask(ornek_json_yolu, yukseklik, genislik)
        print(f"  Mask boyutu : {mask.shape} dtype={mask.dtype}")
        aktif_siniflar = [
            DACL10K_CLASSES[i] for i in range(NUM_CLASSES) if mask[:, :, i].any()
        ]
        print(f"  Bu görseldeki sınıflar: {aktif_siniflar if aktif_siniflar else 'yok (boş annotation)'}")
    else:
        print(f"  UYARI: Annotation bulunamadı: {ornek_json_yolu}")

    # Sınıf dağılımı (ilk 100 dosya)
    print(f"\nSınıf dağılımı (ilk 100 {split_adi} dosyası üzerinden):")
    sinif_sayilari = _sinif_dagilimi_hesapla(annotations_dir, json_dosyalar)
    for i, cls in enumerate(DACL10K_CLASSES):
        bar = "#" * (sinif_sayilari[i] * 40 // max(sinif_sayilari.max(), 1))
        print(f"  {cls:20s}: {sinif_sayilari[i]:3d}  {bar}")


if __name__ == "__main__":
    main()
