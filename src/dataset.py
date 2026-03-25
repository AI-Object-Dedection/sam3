"""
dataset.py — DACL10K Veri Seti Okuyucu

Bu dosya DACL10K veri setini Python içinde kullanılabilir hale getirir.

DACL10K nedir?
Köprü hasarlarını içeren ~9.920 görsellik bir veri setidir.
19 farklı sınıf vardır: çatlak, pas, dökülme gibi hasar türleri
ve köprü bileşenleri (drenaj, mesnet vb.).

Veri formatı (datasetninja.com versiyonu):
- Görseller: .jpg dosyaları (değişken boyut, RGB)
- Annotationlar: .json dosyaları (Supervisely formatı, polygon şekilleri)

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
    kanal = Image.new("L", (width, height), 0)
    cizici = ImageDraw.Draw(kanal)
    # PIL polygon için (x, y) tuple listesi bekliyor
    nokta_listesi = [(int(p[0]), int(p[1])) for p in points]
    if len(nokta_listesi) >= 3:
        cizici.polygon(nokta_listesi, fill=1)
    return np.array(kanal, dtype=np.uint8)


def _json_annotation_to_mask(json_yolu, height, width):
    """
    Supervisely formatındaki JSON annotation dosyasından 19 kanallı mask oluşturur.

    Supervisely formatı:
        {
            "size": {"height": ..., "width": ...},
            "objects": [
                {
                    "classTitle": "Crack",
                    "points": {
                        "exterior": [[x1,y1], [x2,y2], ...]
                    }
                }
            ]
        }

    Args:
        json_yolu: JSON dosyasının tam yolu
        height:    Görselin yüksekliği
        width:     Görselin genişliği

    Returns:
        numpy array: (height, width, 19) — her kanal bir sınıf
    """
    with open(json_yolu, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    # 19 kanallı boş mask (başlangıçta hepsi 0)
    mask = np.zeros((height, width, NUM_CLASSES), dtype=np.uint8)

    # Her annotated nesneyi işle
    for nesne in annotation.get("objects", []):
        sinif_adi = nesne.get("classTitle", "")

        # Tanınan bir sınıf değilse atla
        if sinif_adi not in DACL10K_CLASSES:
            continue

        sinif_idx = DACL10K_CLASSES.index(sinif_adi)
        dis_noktalar = nesne.get("points", {}).get("exterior", [])

        if len(dis_noktalar) < 3:
            continue

        # Bu nesnenin maskini oluştur ve ilgili kanala OR ile ekle
        # (aynı sınıftan birden fazla nesne olabilir)
        nesne_maski = _polygon_to_mask_channel(dis_noktalar, height, width)
        mask[:, :, sinif_idx] = np.maximum(mask[:, :, sinif_idx], nesne_maski)

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
        self.dosya_listesi = self._build_file_list()

        print(f"[dataset] {len(self.dosya_listesi)} görsel yüklendi.")

    def _build_file_list(self):
        """
        Görsel klasöründeki .jpg dosyalarını listeler.

        Annotation klasöründe de aynı isimli .json dosyaların olduğunu varsayar.
        Örneğin: images/train/dacl10k_v2_train_0001.jpg
                 annotations/train/dacl10k_v2_train_0001.json

        Returns:
            list: Sıralı dosya adları listesi (uzantısız, ör: "dacl10k_v2_train_0001")
        """
        jpg_dosyalari = sorted([
            os.path.splitext(f)[0]  # Uzantıyı çıkar: "resim.jpg" → "resim"
            for f in os.listdir(self.images_dir)
            if f.lower().endswith(".jpg") or f.lower().endswith(".jpeg")
        ])
        return jpg_dosyalari

    def __len__(self):
        """Veri setindeki toplam görsel sayısı."""
        return len(self.dosya_listesi)

    def __getitem__(self, idx):
        """
        Tek bir veri noktası döndürür.

        Args:
            idx: İstenen görselin sıra numarası

        Returns:
            dict: Görseli ve mask bilgisini içeren sözlük
        """
        temel_ad = self.dosya_listesi[idx]

        # Görseli oku (.jpg → PIL → numpy)
        gorsel_yolu = os.path.join(self.images_dir, temel_ad + ".jpg")
        if not os.path.exists(gorsel_yolu):
            gorsel_yolu = os.path.join(self.images_dir, temel_ad + ".jpeg")
        gorsel = np.array(Image.open(gorsel_yolu).convert("RGB"), dtype=np.uint8)
        height, width = gorsel.shape[:2]

        # JSON annotation'dan mask oluştur — (height, width, 19)
        json_yolu = os.path.join(self.annotations_dir, temel_ad + ".json")
        if os.path.exists(json_yolu):
            mask = _json_annotation_to_mask(json_yolu, height, width)
        else:
            # JSON bulunamazsa boş mask döndür
            mask = np.zeros((height, width, NUM_CLASSES), dtype=np.uint8)

        # Tüm sınıflardan tek binary mask oluştur (herhangi bir hasar var mı?)
        # (height, width, 19) → (height, width) — herhangi bir kanalda 1 varsa 1
        binary_mask = mask.any(axis=2).astype(np.float32)

        # Eğer processor varsa, görseli modele uygun formata çevir
        if self.processor is not None:
            pil_gorsel = Image.fromarray(gorsel)
            inputs = self.processor(images=pil_gorsel, return_tensors="pt")
            # Batch boyutunu kaldır (processor [1, C, H, W] verir, biz [C, H, W] istiyoruz)
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs["ground_truth_mask"] = torch.tensor(binary_mask)
            inputs["multi_class_mask"] = torch.tensor(
                mask.astype(np.float32)
            ).permute(2, 0, 1)  # (19, height, width)
            return inputs

        # Processor yoksa ham veriyi döndür
        return {
            "image": gorsel,               # (height, width, 3) numpy — RGB
            "mask": binary_mask,           # (height, width) numpy — tek binary mask
            "multi_class_mask": mask,      # (height, width, 19) numpy — sınıf başına mask
            "filename": temel_ad,
        }
