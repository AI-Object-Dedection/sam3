"""
config.py — Projenin Merkezi Ayar Dosyası

Bu dosya projenin tüm ayarlarını tek bir yerde toplar.
Herhangi bir ayarı değiştirmek istediğinizde sadece buraya bakmanız yeterli.

Örneğin:
- Hangi model kullanılacak?
- Kaç epoch eğitim yapılacak?
- Learning rate ne olacak?

Bu dosya projenin "kontrol paneli" gibidir.
"""

import os

import torch


def _env_int(name, default):
    """Ortam degiskeninden integer ayar okur."""
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_str(name, default):
    """Ortam degiskeninden metin ayar okur (Colab/Drive yollari icin)."""
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _env_optional_int(name, default):
    """Bos birakilirsa None/default, doluysa integer ayar okur."""
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    if value.lower() == "none":
        return None
    return int(value)


class Config:
    """Projenin tüm ayarlarını tutan sınıf."""

    # ---- Model Ayarları ----
    MODEL_NAME = os.environ.get("SAM3_MODEL_NAME", "facebook/sam3")

    # ---- Dataset Ayarları ----
    # NOT: Kaggle versiyonu numpy (.npy) formatında
    # Görseller: (512, 512, 3) uint8 | Mask'lar: (512, 512, 19) uint8
    # NOT: Colab'da bu yollar Drive'daki veri setini gosterecek sekilde
    # SAM3_DATA_DIR ortam degiskeni ile degistirilebilir (kodu degistirmeye gerek yok)
    DATA_DIR = _env_str("SAM3_DATA_DIR", "data/dacl10k/")
    if not DATA_DIR.endswith("/"):
        DATA_DIR += "/"
    TRAIN_MASKS = f"{DATA_DIR}annotations/train/"          # Eğitim mask'ları
    VAL_MASKS = f"{DATA_DIR}annotations/validation/"       # Validation mask'ları
    TRAIN_IMAGES = f"{DATA_DIR}images/train/"              # Eğitim görselleri
    VAL_IMAGES = f"{DATA_DIR}images/validation/"           # Validation görselleri

    # ---- Eğitim Ayarları ----
    BATCH_SIZE = 1         # Bir seferde kaç görsel işleneceği
    LEARNING_RATE = 1e-4   # Öğrenme hızı (ne kadar büyük adımlarla öğrensin)
    NUM_EPOCHS = _env_int("NUM_EPOCHS", 10)  # Tüm veriyi kaç kez görecek
    IMAGE_SIZE = 1024      # Görsellerin yeniden boyutlandırılacağı piksel sayısı

    # SAM3 mask decoder her zaman 288x288 boyutunda mask üretir
    MASK_OUTPUT_SIZE = 288

    # Modele verilen metin ipucu — modele "ne aradığını" söylüyoruz
    TEXT_PROMPT = "damage"

    # ---- Loss Ayarları ----
    # Seçenekler: "bce", "dice", "bce_dice", "focal", "focal_dice"
    LOSS_TYPE = "bce_dice"
    BCE_WEIGHT = 0.5
    DICE_WEIGHT = 0.5
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # ---- Augmentation Ayarları ----
    USE_AUGMENTATION = True
    AUG_CROP_RESIZE_PROB = 0.5
    AUG_ROTATION_PROB = 0.4
    AUG_ROTATION_DEGREES = 8
    AUG_BRIGHTNESS_CONTRAST_PROB = 0.5
    AUG_BRIGHTNESS_RANGE = (0.8, 1.2)
    AUG_CONTRAST_RANGE = (0.8, 1.2)
    AUG_BLUR_PROB = 0.2
    AUG_BLUR_KERNEL = 3
    AUG_NOISE_PROB = 0.2
    AUG_NOISE_STD = 0.03
    AUG_PERSPECTIVE_PROB = 0.2

    # ---- Optimizer/Scheduler Ayarları ----
    LR_SCHEDULER = "cosine_warmup"  # Seçenekler: "none", "cosine_warmup", "onecycle"
    WARMUP_RATIO = 0.1               # Toplam adımların yüzde kaçı warmup olsun
    MIN_LR_RATIO = 0.05              # Cosine sonunda LR = LEARNING_RATE * MIN_LR_RATIO

    # ---- DDP Ayarları (Kaggle 2xGPU için) ----
    DDP_BACKEND = "nccl"
    DDP_NUM_WORKERS = 2
    DDP_FIND_UNUSED_PARAMETERS = True

    # ---- Early Stopping Ayarları ----
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 2
    EARLY_STOPPING_MIN_DELTA = 1e-4

    # Test için kaç görsel kullanılacak (None = tüm veri seti)
    MAX_TRAIN_SAMPLES = _env_optional_int("MAX_TRAIN_SAMPLES", None)
    MAX_VAL_SAMPLES = _env_optional_int("MAX_VAL_SAMPLES", None)
    CHECKPOINT_EVERY_STEPS = _env_int("CHECKPOINT_EVERY_STEPS", 500)

    # ---- LoRA Ayarları ----
    LORA_RANK = 8          # LoRA'nın rank değeri (düşük = daha az parametre)
    LORA_ALPHA = 16        # LoRA'nın ölçekleme faktörü
    LORA_DROPOUT = 0.1     # LoRA dropout oranı

    # ---- Klasör Ayarları ----
    # NOT: Colab'da checkpointlerin Drive'a kaydedilmesi icin
    # SAM3_CHECKPOINT_DIR ortam degiskeni kullanilabilir
    # (Kaggle/connection kopmasinda checkpoint kaybolmamasi icin)
    CHECKPOINT_DIR = _env_str("SAM3_CHECKPOINT_DIR", "checkpoints/")
    OUTPUT_DIR = _env_str("SAM3_OUTPUT_DIR", "outputs/")

    # ---- Cihaz Ayarı ----
    # GPU varsa GPU kullan, yoksa CPU kullan
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
