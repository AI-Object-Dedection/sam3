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
    DATA_DIR = "data/"                      # Ham veri klasörü
    TRAIN_ANNOTATIONS = "data/train.jsonl"  # Eğitim annotation dosyası
    VAL_ANNOTATIONS = "data/val.jsonl"      # Validation annotation dosyası

    # ---- Eğitim Ayarları ----
    BATCH_SIZE = 2         # Bir seferde kaç görsel işleneceği
    LEARNING_RATE = 1e-4   # Öğrenme hızı (ne kadar büyük adımlarla öğrensin)
    NUM_EPOCHS = 10        # Tüm veriyi kaç kez görecek
    IMAGE_SIZE = 1024      # Görsellerin yeniden boyutlandırılacağı piksel sayısı

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
