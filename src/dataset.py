"""
dataset.py — DACL10K Veri Seti Okuyucu

Bu dosya DACL10K veri setini Python içinde kullanılabilir hale getirir.

DACL10K nedir?
Köprü hasarlarını içeren ~9.920 görsellik bir veri setidir.
19 farklı sınıf vardır: çatlak, pas, dökülme gibi hasar türleri
ve köprü bileşenleri (drenaj, mesnet vb.).

Veri formatı (orijinal LabelMe formatı):
- Görseller: .jpg dosyaları (değişken boyut, RGB)
- Annotationlar: .json dosyaları (LabelMe formatı, polygon şekilleri)

JSON annotation yapısı:
    {
        "imageName": "dacl10k_v2_train_0001.jpg",
        "imageWidth": 1600,
        "imageHeight": 1200,
        "shapes": [
            {
                "label": "Crack",
                "points": [[x1,y1], [x2,y2], ...],
                "shape_type": "polygon"
            }
        ]
    }

Bu dosyanın görevi:
Görselleri ve JSON annotationları okuyup, polygon koordinatlarından
mask oluşturarak modelin kullanabileceği formata çevirmek.

Beklenen klasör yapısı:
    data/dacl10k/
    ├── images/
    │   ├── train/       ← .jpg görseller
    │   └── validation/  ← .jpg görseller
    └── annotations/
        ├── train/       ← .json annotation dosyaları
        └── validation/  ← .json annotation dosyaları
"""

import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

# DACL10K'deki 19 sınıf (mask kanallarının sırası)
# 13 hasar türü + 6 köprü bileşeni
DACL10K_CLASSES = [
    # Beton hasarları
    "Crack",            # Çatlak
    "ACrack",           # Timsah çatlağı (Alligator Crack)
    "Spalling",         # Dökülme
    "Efflorescence",    # Tuz çiçeklenmesi
    "ExposedRebars",    # Açığa çıkmış donatı
    "Cavity",           # Boşluk
    "Restformwork",     # Kalıp izi
    "Rockpocket",       # Taş cep
    "Hollowareas",      # İçi boş alanlar
    # Genel hasarlar
    "Rust",             # Pas
    "Weathering",       # Aşınma
    "Graffiti",         # Grafiti
    "Wetspot",          # Islak leke
    # Köprü bileşenleri
    "Bearing",          # Mesnet
    "Drainage",         # Drenaj
    "EJoint",           # Genleşme derzi (Expansion Joint)
    "JTape",            # Derz bandı (Joint Tape)
    "PEquipment",       # Koruyucu ekipman (Protective Equipment)
    "WConccor",         # Beton korozyonu (Washouts/Concrete Corrosion)
]

# Sınıf sayısı
NUM_CLASSES = len(DACL10K_CLASSES)  # 19


def _polygon_to_mask_channel(points, height, width):
    """
    Tek bir polygon'dan binary mask oluşturur.

    PIL kütüphanesi kullanarak polygon'un içini 1 ile doldurur.

    Args:
        points: Polygon köşe noktaları [[x1,y1], [x2,y2], ...]
        height:  Mask yüksekliği (piksel)
        width:   Mask genişliği (piksel)

    Returns:
        numpy array: (height, width) — 0/1 binary mask
    """
    channel = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(channel)
    # PIL polygon için (x, y) tuple listesi bekliyor
    point_list = [(int(p[0]), int(p[1])) for p in points]
    if len(point_list) >= 3:
        drawer.polygon(point_list, fill=1)
    return np.array(channel, dtype=np.uint8)


def _json_annotation_to_mask(json_path, height, width):
    """
    LabelMe formatındaki JSON annotation dosyasından 19 kanallı mask oluşturur.

    LabelMe formatı:
        {
            "imageName": "dacl10k_v2_train_0001.jpg",
            "imageWidth": 1600,
            "imageHeight": 1200,
            "shapes": [
                {
                    "label": "Crack",
                    "points": [[x1,y1], [x2,y2], ...],
                    "shape_type": "polygon"
                }
            ]
        }

    Args:
        json_path: JSON dosyasının tam yolu
        height:    Görselin yüksekliği
        width:     Görselin genişliği

    Returns:
        numpy array: (height, width, 19) — her kanal bir sınıf
    """
    with open(json_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    # 19 kanallı boş mask (başlangıçta hepsi 0)
    mask = np.zeros((height, width, NUM_CLASSES), dtype=np.uint8)

    # Her shape'i (polygon) işle
    for shape in annotation.get("shapes", []):
        class_name = shape.get("label", "")

        # Tanınan bir sınıf değilse atla
        if class_name not in DACL10K_CLASSES:
            continue

        class_idx = DACL10K_CLASSES.index(class_name)
        points = shape.get("points", [])

        if len(points) < 3:
            continue

        # Bu polygon'un maskini oluştur ve ilgili kanala OR ile ekle
        # (aynı sınıftan birden fazla polygon olabilir)
        polygon_mask = _polygon_to_mask_channel(points, height, width)
        mask[:, :, class_idx] = np.maximum(mask[:, :, class_idx], polygon_mask)

    return mask


class DACL10KDataset(Dataset):
    """
    DACL10K veri seti için PyTorch Dataset sınıfı.

    Datasetninja.com versiyonunda görseller .jpg, annotationlar .json
    (Supervisely formatı, polygon şekilleri) olarak saklanır.

    Bu sınıf:
    1. Klasördeki .jpg dosya listesini okur
    2. Her veri noktası için: görsel + JSON'dan oluşturulmuş mask döndürür
    """

    def __init__(self, images_dir, annotations_dir, processor=None):
        """
        Args:
            images_dir:       Görsel .jpg dosyalarının bulunduğu klasör
                              (ör: "data/dacl10k/images/train/")
            annotations_dir:  JSON annotation dosyalarının bulunduğu klasör
                              (ör: "data/dacl10k/annotations/train/")
            processor:        SAM3 Processor (görseli modele uygun formata çevirir)
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.processor = processor

        # .jpg dosya listesini oluştur (sıralı)
        self.file_list = self._build_file_list()

        print(f"[dataset] {len(self.file_list)} görsel yüklendi.")

    def _build_file_list(self):
        """
        Görsel klasöründeki .jpg dosyalarını listeler.

        Annotation klasöründe de aynı isimli .json dosyaların olduğunu varsayar.
        Örneğin: images/train/dacl10k_v2_train_0001.jpg
                 annotations/train/dacl10k_v2_train_0001.json

        Returns:
            list: Sıralı dosya adları listesi (uzantısız, ör: "dacl10k_v2_train_0001")
        """
        jpg_files = sorted([
            os.path.splitext(f)[0]  # Uzantıyı çıkar: "resim.jpg" → "resim"
            for f in os.listdir(self.images_dir)
            if f.lower().endswith(".jpg") or f.lower().endswith(".jpeg")
        ])
        return jpg_files

    def __len__(self):
        """Veri setindeki toplam görsel sayısı."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Tek bir veri noktası döndürür.

        Args:
            idx: İstenen görselin sıra numarası

        Returns:
            dict: Görseli ve mask bilgisini içeren sözlük
        """
        base_name = self.file_list[idx]

        # Görseli oku (.jpg → PIL → numpy)
        image_path = os.path.join(self.images_dir, base_name + ".jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(self.images_dir, base_name + ".jpeg")
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        height, width = image.shape[:2]

        # JSON annotation'dan mask oluştur — (height, width, 19)
        json_path = os.path.join(self.annotations_dir, base_name + ".json")
        if os.path.exists(json_path):
            mask = _json_annotation_to_mask(json_path, height, width)
        else:
            # JSON bulunamazsa boş mask döndür
            mask = np.zeros((height, width, NUM_CLASSES), dtype=np.uint8)

        # Tüm sınıflardan tek binary mask oluştur (herhangi bir hasar var mı?)
        # (height, width, 19) → (height, width) — herhangi bir kanalda 1 varsa 1
        binary_mask = mask.any(axis=2).astype(np.float32)

        # Eğer processor varsa, görseli modele uygun formata çevir
        if self.processor is not None:
            pil_image = Image.fromarray(image)
            inputs = self.processor(images=pil_image, return_tensors="pt")
            # Batch boyutunu kaldır (processor [1, C, H, W] verir, biz [C, H, W] istiyoruz)
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs["ground_truth_mask"] = torch.tensor(binary_mask)
            inputs["multi_class_mask"] = torch.tensor(
                mask.astype(np.float32)
            ).permute(2, 0, 1)  # (19, height, width)
            return inputs

        # Processor yoksa ham veriyi döndür
        return {
            "image": image,               # (height, width, 3) numpy — RGB
            "mask": binary_mask,          # (height, width) numpy — tek binary mask
            "multi_class_mask": mask,     # (height, width, 19) numpy — sınıf başına mask
            "filename": base_name,
        }
