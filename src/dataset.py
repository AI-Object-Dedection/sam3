"""
dataset.py — DACL10K Veri Seti Okuyucu

Bu dosya DACL10K veri setini Python icinde kullanilabilir hale getirir.

DACL10K nedir?
Kopru hasarlarini iceren ~10.000 gorsellik bir veri setidir.
19 farkli sinif vardir: catlak, pas, dokulme gibi hasar turleri
ve kopru bilesenleri (drenaj, mesnet vb.).

Veri formati (Kaggle versiyonu):
- Gorseller: .npy dosyalari (512, 512, 3) uint8 — RGB gorseller
- Mask'lar: .npy dosyalari (512, 512, 19) uint8 — her kanal bir sinif

Bu dosyanin gorevi:
Gorselleri ve maskeleri okuyup, modelin kullanabilecegi formata cevirmek.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

# DACL10K'deki 19 sinif (mask kanallarinin sirasi)
# 13 hasar turu + 6 kopru bileseni
# Not: Sinif adlari DACL10K'nin resmi kisaltmalaridir
DACL10K_CLASSES = [
    # Beton hasarlari
    "Crack",            # Catlak
    "ACrack",           # Timsah catlagi (Alligator Crack)
    "Spalling",         # Dokulme
    "Efflorescence",    # Tuz ciceklenmesi
    "ExposedRebars",    # Aciga cikmis donati
    "Cavity",           # Bosluk
    "Restformwork",     # Kalip izi
    "Rockpocket",       # Tas cep
    "Hollowareas",      # Ici bos alanlar
    # Genel hasarlar
    "Rust",             # Pas
    "Weathering",       # Asinma
    "Graffiti",         # Grafiti
    "Wetspot",          # Islak leke
    # Kopru bilesenleri
    "Bearing",          # Mesnet
    "Drainage",         # Drenaj
    "EJoint",           # Genlesme derzi (Expansion Joint)
    "JTape",            # Derz bandi (Joint Tape)
    "PEquipment",       # Koruyucu ekipman (Protective Equipment)
    "WConccor",         # Beton korozyonu (Washouts/Concrete Corrosion)
]

# Sinif sayisi
NUM_CLASSES = len(DACL10K_CLASSES)  # 19


class DACL10KDataset(Dataset):
    """
    DACL10K veri seti icin PyTorch Dataset sinifi.

    Kaggle versiyonunda gorseller ve mask'lar .npy dosyalari olarak saklanir.
    Her gorsel 512x512x3 (RGB), her mask 512x512x19 (sinif basina binary mask).

    Bu sinif:
    1. Klasordeki .npy dosya listesini okur
    2. Her veri noktasi icin: gorsel + mask dondurur
    """

    def __init__(self, images_dir, masks_dir, processor=None):
        """
        Args:
            images_dir: Gorsel .npy dosyalarinin bulundugu klasor
                        (orn: "data/dacl10k/images/train/")
            masks_dir:  Mask .npy dosyalarinin bulundugu klasor
                        (orn: "data/dacl10k/annotations/train/")
            processor:  SAM3 Processor (gorseli modele uygun formata cevirir)
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.processor = processor

        # .npy dosya listesini olustur (sirali)
        self.file_list = self._build_file_list()

        print(f"[dataset] {len(self.file_list)} gorsel yuklendi.")

    def _build_file_list(self):
        """
        Gorsel klasorundeki .npy dosyalarini listeler.

        Annotation klasorunde de ayni isimli dosyalarin oldugunu varsayar.
        Ornegin: images/train/dacl10k_v2_train_0000.npy
                 annotations/train/dacl10k_v2_train_0000.npy

        Returns:
            list: Sirali dosya adlari listesi
        """
        files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.endswith(".npy")
        ])
        return files

    def __len__(self):
        """Veri setindeki toplam gorsel sayisi."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Tek bir veri noktasi dondurur.

        Args:
            idx: Istenilen gorselin sira numarasi

        Returns:
            dict: Gorseli ve mask bilgisini iceren sozluk
        """
        filename = self.file_list[idx]

        # Gorseli oku — (512, 512, 3) uint8
        image_path = os.path.join(self.images_dir, filename)
        image = np.load(image_path)

        # Mask'i oku — (512, 512, 19) uint8
        mask_path = os.path.join(self.masks_dir, filename)
        mask = np.load(mask_path)

        # Tum siniflardan tek bir binary mask olustur (herhangi bir hasar var mi?)
        # (512, 512, 19) -> (512, 512) — herhangi bir kanalda 1 varsa 1
        binary_mask = mask.any(axis=2).astype(np.float32)

        # Eger processor varsa, gorseli modele uygun formata cevir
        if self.processor is not None:
            # PIL Image'a cevir (processor bunu bekliyor)
            from PIL import Image
            pil_image = Image.fromarray(image)

            inputs = self.processor(images=pil_image, return_tensors="pt")
            # Batch boyutunu kaldir (processor [1, C, H, W] verir, biz [C, H, W] istiyoruz)
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs["ground_truth_mask"] = torch.tensor(binary_mask)
            inputs["multi_class_mask"] = torch.tensor(mask.astype(np.float32)).permute(2, 0, 1)  # (19, 512, 512)
            return inputs

        # Processor yoksa ham veriyi dondur
        return {
            "image": image,                # (512, 512, 3) numpy
            "mask": binary_mask,           # (512, 512) numpy — tek binary mask
            "multi_class_mask": mask,      # (512, 512, 19) numpy — sinif basina mask
            "filename": filename,
        }
