"""
download_dataset.py — DACL10K Veri Setini İndirme Script'i

Bu dosya DACL10K veri setini HuggingFace'ten indirir
ve projemizin beklediği klasör yapısına dönüştürür.

Beklenen çıktı yapısı:
    data/dacl10k/
    ├── annotations/
    │   ├── train/*.json
    │   └── validation/*.json
    └── images/
        ├── train/*.jpg
        └── validation/*.jpg

Kullanım:
    python scripts/download_dataset.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import log, ensure_dir


def main():
    log("=" * 50)
    log("DACL10K Veri Seti İndirme")
    log("=" * 50)

    # 1. HuggingFace datasets kütüphanesini import et
    try:
        from datasets import load_dataset
    except ImportError:
        print("Hata: 'datasets' kütüphanesi kurulu değil.")
        print("Kurmak için: pip install datasets")
        return

    # 2. Hedef klasörleri oluştur
    base_dir = "data/dacl10k"
    for split in ["train", "validation"]:
        ensure_dir(os.path.join(base_dir, "annotations", split))
        ensure_dir(os.path.join(base_dir, "images", split))

    # 3. Veri setini HuggingFace'ten indir
    log("DACL10K HuggingFace'ten indiriliyor... (bu biraz sürebilir)")

    # Sadece train ve validation split'lerini indir
    dataset = load_dataset("Voxel51/dacl10k")

    log(f"İndirme tamamlandı. Split'ler: {list(dataset.keys())}")

    # 4. Her split için görselleri ve annotation'ları kaydet
    for split_name in ["train", "validation"]:
        if split_name not in dataset:
            log(f"Uyarı: '{split_name}' split'i bulunamadı, atlanıyor.")
            continue

        split_data = dataset[split_name]
        log(f"\n{split_name} split'i işleniyor: {len(split_data)} görsel")

        images_dir = os.path.join(base_dir, "images", split_name)
        annotations_dir = os.path.join(base_dir, "annotations", split_name)

        for idx in range(len(split_data)):
            item = split_data[idx]

            # HuggingFace dataset yapısını kontrol et
            # İlk öğeyi yazdırarak yapıyı anlayalım
            if idx == 0:
                log(f"Veri yapısı (ilk öğe): {list(item.keys())}")

            # Görseli kaydet
            if "image" in item:
                image = item["image"]
                # Dosya adını belirle
                if "imageName" in item:
                    image_name = item["imageName"]
                else:
                    image_name = f"dacl10k_{split_name}_{idx:05d}.jpg"

                image_path = os.path.join(images_dir, image_name)
                if not os.path.exists(image_path):
                    image.save(image_path)

            # Annotation'ı kaydet
            # HuggingFace formatına bağlı olarak uyarlanacak
            if "annotation" in item or "shapes" in item:
                import json
                ann_name = image_name.replace(".jpg", ".json").replace(".png", ".json")
                ann_path = os.path.join(annotations_dir, ann_name)

                if not os.path.exists(ann_path):
                    ann_data = {}
                    for key in item:
                        if key != "image":  # Görsel hariç tüm alanları al
                            ann_data[key] = item[key]
                    with open(ann_path, "w") as f:
                        json.dump(ann_data, f)

            # İlerleme göster
            if (idx + 1) % 500 == 0:
                log(f"  {idx + 1}/{len(split_data)} işlendi")

        log(f"{split_name} tamamlandı: {len(split_data)} görsel kaydedildi")

    log("\n" + "=" * 50)
    log("İndirme tamamlandı!")
    log(f"Veri seti konumu: {base_dir}")
    log("Doğrulamak için: python scripts/inspect_dataset.py")


if __name__ == "__main__":
    main()
