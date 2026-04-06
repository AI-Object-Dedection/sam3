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

import torch


class Config:
    """Projenin tüm ayarlarını tutan sınıf."""

    # ---- Model Ayarları ----
    MODEL_NAME = "facebook/sam3"  # HuggingFace'teki model adı

    # ---- Dataset Ayarları ----
    # NOT: Kaggle versiyonu numpy (.npy) formatında
    # Görseller: (512, 512, 3) uint8 | Mask'lar: (512, 512, 19) uint8
    DATA_DIR = "data/dacl10k/"                             # DACL10K ana klasörü
    TRAIN_MASKS = "data/dacl10k/annotations/train/"        # Eğitim mask'ları (.npy, 512x512x19)
    VAL_MASKS = "data/dacl10k/annotations/validation/"     # Validation mask'ları (.npy)
    TRAIN_IMAGES = "data/dacl10k/images/train/"            # Eğitim görselleri (.npy, 512x512x3)
    VAL_IMAGES = "data/dacl10k/images/validation/"         # Validation görselleri (.npy)

    # ---- Eğitim Ayarları ----
    BATCH_SIZE = 1         # Bir seferde kaç görsel işleneceği
    LEARNING_RATE = 1e-4   # Öğrenme hızı (ne kadar büyük adımlarla öğrensin)
    NUM_EPOCHS = 10        # Tüm veriyi kaç kez görecek
    IMAGE_SIZE = 1024      # Görsellerin yeniden boyutlandırılacağı piksel sayısı

    # SAM3 mask decoder her zaman 288x288 boyutunda mask üretir
    MASK_OUTPUT_SIZE = 288

    # Modele verilen metin ipucu — modele "ne aradığını" söylüyoruz
    TEXT_PROMPT = "damage"

    # Test için kaç görsel kullanılacak (None = tüm veri seti)
    MAX_TRAIN_SAMPLES = None  # Tüm 6935 eğitim görseli
    MAX_VAL_SAMPLES = None    # Tüm 975 validation görseli

    # ---- LoRA Ayarları ----
    LORA_RANK = 8          # LoRA'nın rank değeri (düşük = daha az parametre)
    LORA_ALPHA = 16        # LoRA'nın ölçekleme faktörü
    LORA_DROPOUT = 0.1     # LoRA dropout oranı

    # ---- Klasör Ayarları ----
    CHECKPOINT_DIR = "checkpoints/"  # Model ağırlıklarının kaydedileceği yer
    OUTPUT_DIR = "outputs/"          # Çıktıların kaydedileceği yer

    # ---- Cihaz Ayarı ----
    # GPU varsa GPU kullan, yoksa CPU kullan
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
