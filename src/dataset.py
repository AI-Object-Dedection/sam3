"""
dataset.py — DACL10K Veri Seti Okuyucu

Bu dosya DACL10K veri setini Python içinde kullanılabilir hale getirir.

DACL10K nedir?
Köprü hasarlarını içeren ~9.920 görsellik bir veri setidir.
19 farklı sınıf vardır: çatlak, pas, dökülme gibi hasar türleri
ve köprü bileşenleri (drenaj, mesnet vb.).

Veri formatı (Kaggle / numpy versiyonu):
- Görseller  : .npy dosyaları — (512, 512, 3) uint8 (RGB)
- Annotationlar: .npy dosyaları — (512, 512, 19) uint8 (her kanal bir sınıf, 0/1)

Klasör yapısı:
    data/dacl10k/
    ├── images/
    │   ├── train/       ← dacl10k_v2_train_XXXX.npy
    │   └── validation/  ← dacl10k_v2_val_XXXX.npy
    └── annotations/
        ├── train/       ← dacl10k_v2_train_XXXX.npy
        └── validation/  ← dacl10k_v2_val_XXXX.npy

Bu dosyanın görevi:
.npy dosyalarını okuyarak modelin kullanabileceği formata çevirmek.
"""

import json
import os
import random

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.config import Config

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


def _json_annotation_to_mask(json_path, height, width):
    """
    DatasetNinja/DACL10K JSON annotation dosyasını çok kanallı maskeye çevirir.

    JSON içinde her şekil `shapes` listesinde tutulur:
    - `label`: sınıf adı
    - `points`: polygon noktaları

    Dönen mask boyutu: (height, width, 19)
    """
    mask = np.zeros((height, width, NUM_CLASSES), dtype=np.uint8)

    with open(json_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    for shape in annotation.get("shapes", []):
        class_name = shape.get("label")
        points = shape.get("points", [])

        if class_name not in DACL10K_CLASSES or len(points) < 3:
            continue

        class_index = DACL10K_CLASSES.index(class_name)
        single_mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(single_mask)
        polygon = [(int(x), int(y)) for x, y in points]
        draw.polygon(polygon, outline=1, fill=1)

        mask[:, :, class_index] = np.maximum(
            mask[:, :, class_index],
            np.array(single_mask, dtype=np.uint8),
        )

    return mask


class DACL10KDataset(Dataset):
    """
    DACL10K veri seti için PyTorch Dataset sınıfı.

    Kaggle versiyonunda görseller ve maskler .npy formatında saklanır.
    - Görsel: (512, 512, 3) uint8 — RGB
    - Mask  : (512, 512, 19) uint8 — her kanal bir hasar sınıfı (0/1)

    Bu sınıf:
    1. Klasördeki .npy dosya listesini okur
    2. Her veri noktası için: görsel + mask döndürür

    Döndürülen veri formatı (processor varsa, SAM3 için):
        pixel_values     : (3, 1008, 1008)  — görselin işlenmiş hali
        input_ids        : (seq_len,)        — metin tokenları
        attention_mask   : (seq_len,)        — metin attention maskı
        ground_truth_mask: (288, 288)        — gerçek hasar maskı (0/1)
    """

    def __init__(
        self,
        images_dir,
        annotations_dir,
        processor=None,
        max_samples=None,
        is_train=False,
        use_augmentation=False,
    ):
        """
        Args:
            images_dir:       Görsel .npy dosyalarının bulunduğu klasör
            annotations_dir:  Mask .npy dosyalarının bulunduğu klasör
            processor:        SAM3 Processor (görseli ve metni modele uygun formata çevirir)
            max_samples:      Kaç görsel kullanılacak (None = hepsi, 100 = test için)
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.processor = processor
        self.is_train = is_train
        self.use_augmentation = use_augmentation

        # .npy dosya listesini oluştur (sıralı)
        self.file_list = self._build_file_list()

        # Sadece belirli sayıda örnek kullan (test için)
        if max_samples is not None:
            self.file_list = self.file_list[:max_samples]

        # Metin tokenlarını bir kez hesapla — her örnek için aynı metin ("damage")
        # NOT: Burada sadece string saklıyoruz, tokenization __getitem__'de değil
        # başlangıçta yapılır ama sadece bir kez
        self._input_ids = None
        self._attention_mask = None

        print(f"[dataset] {len(self.file_list)} gorsel yuklendi.")

        if processor is not None:
            from src.config import Config
            text_inputs = processor.tokenizer(
                Config.TEXT_PROMPT,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            self._input_ids = text_inputs["input_ids"].squeeze(0)
            self._attention_mask = text_inputs["attention_mask"].squeeze(0)
            print("[dataset] Tokenizer hazir.")

    def _apply_augmentations(self, image, mask):
        """
        Eğitim sırasında veri çeşitliliği için augmentasyon uygular.

        image: (H, W, 3) uint8
        mask : (H, W, C) uint8
        """
        from src.config import Config

        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).permute(2, 0, 1).float()

        _, h, w = image_t.shape

        if random.random() < Config.AUG_CROP_RESIZE_PROB:
            crop_scale = random.uniform(0.75, 1.0)
            crop_h = max(1, int(h * crop_scale))
            crop_w = max(1, int(w * crop_scale))
            top = random.randint(0, h - crop_h) if h > crop_h else 0
            left = random.randint(0, w - crop_w) if w > crop_w else 0
            image_t = TF.resized_crop(
                image_t,
                top=top,
                left=left,
                height=crop_h,
                width=crop_w,
                size=[h, w],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            mask_t = TF.resized_crop(
                mask_t,
                top=top,
                left=left,
                height=crop_h,
                width=crop_w,
                size=[h, w],
                interpolation=InterpolationMode.NEAREST,
                antialias=False,
            )

        if random.random() < Config.AUG_ROTATION_PROB:
            angle = random.uniform(-Config.AUG_ROTATION_DEGREES, Config.AUG_ROTATION_DEGREES)
            image_t = TF.rotate(image_t, angle, interpolation=InterpolationMode.BILINEAR)
            mask_t = TF.rotate(mask_t, angle, interpolation=InterpolationMode.NEAREST)

        if random.random() < Config.AUG_PERSPECTIVE_PROB:
            max_warp_w = int(w * 0.06)
            max_warp_h = int(h * 0.06)
            start_points = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
            end_points = [
                [random.randint(0, max_warp_w), random.randint(0, max_warp_h)],
                [w - 1 - random.randint(0, max_warp_w), random.randint(0, max_warp_h)],
                [w - 1 - random.randint(0, max_warp_w), h - 1 - random.randint(0, max_warp_h)],
                [random.randint(0, max_warp_w), h - 1 - random.randint(0, max_warp_h)],
            ]
            image_t = TF.perspective(
                image_t,
                startpoints=start_points,
                endpoints=end_points,
                interpolation=InterpolationMode.BILINEAR,
            )
            mask_t = TF.perspective(
                mask_t,
                startpoints=start_points,
                endpoints=end_points,
                interpolation=InterpolationMode.NEAREST,
            )

        if random.random() < Config.AUG_BRIGHTNESS_CONTRAST_PROB:
            brightness_factor = random.uniform(*Config.AUG_BRIGHTNESS_RANGE)
            contrast_factor = random.uniform(*Config.AUG_CONTRAST_RANGE)
            image_t = TF.adjust_brightness(image_t, brightness_factor)
            image_t = TF.adjust_contrast(image_t, contrast_factor)

        if random.random() < Config.AUG_BLUR_PROB:
            image_t = TF.gaussian_blur(image_t, kernel_size=Config.AUG_BLUR_KERNEL)

        if random.random() < Config.AUG_NOISE_PROB:
            noise = torch.randn_like(image_t) * Config.AUG_NOISE_STD
            image_t = (image_t + noise).clamp(0.0, 1.0)

        image_aug = (image_t.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        mask_aug = (mask_t.permute(1, 2, 0).numpy() > 0.5).astype(np.uint8)
        return image_aug, mask_aug

    def _build_file_list(self):
        """
        Görsel klasöründeki .npy dosyalarını listeler.

        Görsel ve annotation dosyalarının aynı ada sahip olduğunu varsayar.
        Örnek: images/train/dacl10k_v2_train_0001.npy
               annotations/train/dacl10k_v2_train_0001.npy

        Returns:
            list: Sıralı dosya adları listesi (uzantısız)
        """
        npy_files = sorted([
            os.path.splitext(f)[0]  # Uzantıyı çıkar: "resim.npy" → "resim"
            for f in os.listdir(self.images_dir)
            if f.lower().endswith((".npy", ".jpg", ".jpeg", ".png"))
        ])
        return npy_files

    def _v1_to_v2_isim(self, temel_ad):
        """
        v1 görsel adını v2 annotation adına çevirir.

        Neden gerekli: görseller eski isimlendirme (v1) ile yüklendi,
        annotation'lar yeni isimlendirme (v2) ile geliyor.

        Örnekler:
            dacl10k_train_00000      → dacl10k_v2_train_0000
            dacl10k_validation_00000 → dacl10k_v2_validation_0000
        """
        import re
        eslesme = re.match(r"dacl10k_(train|validation)_(\d+)$", temel_ad)
        if eslesme:
            kategori = eslesme.group(1)   # "train" veya "validation"
            num = int(eslesme.group(2))   # sayısal kısım (5 basamak → int → 4 basamak)
            return f"dacl10k_v2_{kategori}_{num:04d}"
        return None

    def _dosya_bul(self, klasor, temel_ad, uzantilar):
        """
        Verilen temel ada ait ilk mevcut dosya yolunu bulur.

        Önce verilen adı dener. Bulamazsa v1→v2 dönüşümü yaparak tekrar dener.
        Bu sayede v1 isimli görseller (dacl10k_train_00000.jpg) ile
        v2 isimli annotation'lar (dacl10k_v2_train_0000.json) eşleşebilir.
        """
        # Önce direkt dene
        for uzanti in uzantilar:
            dosya_yolu = os.path.join(klasor, temel_ad + uzanti)
            if os.path.exists(dosya_yolu):
                return dosya_yolu
        # Bulunamadıysa v1→v2 dönüşümünü dene
        v2_ad = self._v1_to_v2_isim(temel_ad)
        if v2_ad:
            for uzanti in uzantilar:
                dosya_yolu = os.path.join(klasor, v2_ad + uzanti)
                if os.path.exists(dosya_yolu):
                    return dosya_yolu
        return None

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

        # Görseli oku — (512, 512, 3) uint8
        image_path = self._dosya_bul(
            self.images_dir,
            base_name,
            [".npy", ".jpg", ".jpeg", ".png"],
        )
        if image_path is None:
            raise FileNotFoundError(f"Görsel bulunamadı: {base_name}")

        if image_path.lower().endswith(".npy"):
            image = np.load(image_path)
        else:
            image = np.array(Image.open(image_path).convert("RGB"))

        # Mask'ı oku — (512, 512, 19) uint8
        mask_path = self._dosya_bul(
            self.annotations_dir,
            base_name,
            [".npy", ".json"],
        )
        if mask_path is None:
            mask = np.zeros((image.shape[0], image.shape[1], NUM_CLASSES), dtype=np.uint8)
        elif mask_path.lower().endswith(".npy"):
            mask = np.load(mask_path)
        else:
            mask = _json_annotation_to_mask(
                mask_path,
                height=image.shape[0],
                width=image.shape[1],
            )

        if self.is_train and self.use_augmentation:
            image, mask = self._apply_augmentations(image, mask)

        # Tüm 19 sınıfı tek binary mask'a birleştir
        # (512, 512, 19) → (512, 512): herhangi bir kanalda 1 varsa 1 yap
        binary_mask = mask.any(axis=2).astype(np.float32)

        # Eğer processor varsa, görseli modele uygun formata çevir
        if self.processor is not None:
            from src.config import Config

            pil_image = Image.fromarray(image)

            # Görseli işle — image_processor sadece görseli alır
            image_inputs = self.processor.image_processor(
                images=pil_image,
                return_tensors="pt",
            )
            # Batch boyutunu kaldır: (1, 3, H, W) → (3, H, W)
            pixel_values = image_inputs["pixel_values"].squeeze(0)

            # Mask'ı SAM3 çıktı boyutuna (288x288) küçült
            # Model her zaman 288x288 mask üretir — ground truth da aynı boyutta olmalı
            mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
            mask_resized = mask_pil.resize(
                (Config.MASK_OUTPUT_SIZE, Config.MASK_OUTPUT_SIZE),
                Image.NEAREST,  # En yakın komşu — 0/1 değerleri korur
            )
            mask_resized = np.array(mask_resized, dtype=np.float32) / 255.0

            # Her sınıfın maskini de 288x288'e küçült — (19, 288, 288)
            # Multi-class eğitim için: her kanal bir sınıfın maskleri
            sinif_maskleri = []
            for kanal in range(NUM_CLASSES):
                kanal_mask = mask[:, :, kanal].astype(np.float32)
                kanal_pil = Image.fromarray((kanal_mask * 255).astype(np.uint8))
                kanal_resized = kanal_pil.resize(
                    (Config.MASK_OUTPUT_SIZE, Config.MASK_OUTPUT_SIZE),
                    Image.NEAREST,
                )
                sinif_maskleri.append(np.array(kanal_resized, dtype=np.float32) / 255.0)
            # (19, 288, 288) tensor
            multi_class_mask = torch.tensor(np.stack(sinif_maskleri, axis=0))

            return {
                "pixel_values"     : pixel_values,                 # (3, 1008, 1008)
                "input_ids"        : self._input_ids.clone(),      # (seq_len,)
                "attention_mask"   : self._attention_mask.clone(), # (seq_len,)
                "ground_truth_mask": torch.tensor(mask_resized),   # (288, 288)
                "multi_class_mask" : multi_class_mask,             # (19, 288, 288)
            }

        # Processor yoksa ham veriyi döndür (inspect_dataset.py gibi araçlar için)
        return {
            "image"           : image,         # (512, 512, 3) numpy — RGB
            "mask"            : binary_mask,   # (512, 512) numpy — tek binary mask
            "multi_class_mask": mask,          # (512, 512, 19) numpy — sınıf başına mask
            "filename"        : base_name,
        }
