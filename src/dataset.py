"""
dataset.py — DACL10K Veri Seti Okuyucu

Bu dosya DACL10K veri setini Python içinde kullanılabilir hale getirir.

DACL10K nedir?
Köprü hasarlarını içeren ~10.000 görsellik bir veri setidir.
Her görselde hasarlar poligon (çokgen) olarak işaretlenmiştir.
19 farklı sınıf vardır: çatlak, pas, döküntü gibi hasar türleri
ve köprü bileşenleri (drenaj, rulman vb.).

Annotation formatı:
JSONL formatında — her satır bir JSON objesidir.
Her obje bir görselin bilgilerini içerir:
- imageName: görsel dosya adı
- shapes: işaretlenmiş alanların listesi (poligon koordinatları + sınıf adı)

Bu dosyanın görevi:
Görselleri ve maskeleri okuyup, modelin kullanabileceği formata çevirmek.
"""

import json
import os

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from src.config import Config

# DACL10K'deki 19 sınıf
# 13 hasar türü + 6 köprü bileşeni
DACL10K_CLASSES = [
    # Beton hasarları
    "Crack",
    "Alligator Crack",
    "Spalling",
    "Efflorescence",
    "Exposed Rebars",
    "Cavity",
    "Restformwork",
    "Rockpocket",
    "Hollowareas",
    # Genel hasarlar
    "Rust",
    "Weathering",
    "Graffiti",
    "Wetspot",
    # Köprü bileşenleri
    "Bearing",
    "Drainage",
    "Expansion Joint",
    "Joint Tape",
    "Protective Equipment",
    "Washouts/Concrete Corrosion",
]


class DACL10KDataset(Dataset):
    """
    DACL10K veri seti için PyTorch Dataset sınıfı.

    PyTorch Dataset nedir?
    PyTorch'un veri yükleme sistemidir. __getitem__ ile tek tek veri noktalarına
    erişim sağlar. DataLoader ile birlikte batch'ler halinde veri verir.

    Bu sınıf:
    1. JSONL annotation dosyasını okur
    2. Her bir görsel için: görseli + binary mask'ı döndürür
    """

    def __init__(self, annotations_path, images_dir, processor=None):
        """
        Args:
            annotations_path: JSONL annotation dosyasının yolu
            images_dir: Görsellerin bulunduğu klasör yolu
            processor: SAM3 Processor (görseli modele uygun formata çevirir)
        """
        self.images_dir = images_dir
        self.processor = processor

        # Annotation dosyasını oku
        self.annotations = self._load_annotations(annotations_path)

        print(f"[dataset] {len(self.annotations)} görsel yüklendi.")

    def _load_annotations(self, path):
        """
        JSONL dosyasını satır satır okuyup listeye çevirir.

        JSONL = her satır bağımsız bir JSON objesi
        """
        annotations = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:  # Boş satırları atla
                    annotations.append(json.loads(line))
        return annotations

    def __len__(self):
        """Veri setindeki toplam görsel sayısı."""
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Tek bir veri noktası döndürür.

        Args:
            idx: İstenilen görselin sıra numarası

        Returns:
            dict: Görseli ve mask bilgisini içeren sözlük
        """
        # Annotation bilgisini al
        ann = self.annotations[idx]

        # Görseli oku
        image_path = os.path.join(self.images_dir, ann["imageName"])
        image = Image.open(image_path).convert("RGB")

        # Görselin boyutlarını al
        width = ann["imageWidth"]
        height = ann["imageHeight"]

        # Poligonlardan binary mask oluştur
        mask = self._create_mask(ann["shapes"], width, height)

        # Eğer processor varsa, görseli modele uygun formata çevir
        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")
            # Batch boyutunu kaldır (processor [1, C, H, W] verir, biz [C, H, W] istiyoruz)
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs["ground_truth_mask"] = mask
            return inputs

        # Processor yoksa ham veriyi döndür
        return {
            "image": image,
            "mask": mask,
            "image_name": ann["imageName"],
        }

    def _create_mask(self, shapes, width, height):
        """
        Poligon koordinatlarından binary mask oluşturur.

        Binary mask nedir?
        Görüntüyle aynı boyutta, hasar olan piksellerde 1,
        olmayan yerlerde 0 bulunan bir matristir.

        Args:
            shapes: Poligon bilgileri listesi (her biri label + koordinat)
            width: Görsel genişliği
            height: Görsel yüksekliği

        Returns:
            numpy array: Binary mask (height x width)
        """
        # Boş bir mask oluştur (tüm değerler 0 = arka plan)
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        # Her poligonu mask'a çiz
        for shape in shapes:
            # Koordinatları tuple listesine çevir
            # Orijinal format: [[x1, y1], [x2, y2], ...]
            # Pillow formatı: [(x1, y1), (x2, y2), ...]
            points = [tuple(point) for point in shape["points"]]

            if len(points) >= 3:  # Poligon en az 3 nokta olmalı
                draw.polygon(points, fill=1)  # Hasar olan bölgeyi 1 ile doldur

        return np.array(mask, dtype=np.float32)
