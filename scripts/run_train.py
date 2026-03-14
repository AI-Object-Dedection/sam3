"""
run_train.py — Eğitimi Başlatan Script

Bu dosya src/ klasöründeki modülleri çağırarak eğitim sürecini başlatır.

Kullanım:
    python scripts/run_train.py
"""

import sys
import os

# Proje kök dizinini Python'un arama yoluna ekle
# Bu sayede "from src.config import Config" gibi importlar çalışır
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils import set_seed, log
from src.model import load_model, load_processor
from src.lora import apply_lora
from src.train import train


def main():
    """Eğitim sürecini adım adım çalıştırır."""

    log("=" * 50)
    log("SAM3 Fine-Tuning — Eğitim Başlıyor")
    log("=" * 50)

    # 1. Rastgeleliği sabitle
    set_seed()

    # 2. Model ve processor'ı yükle
    model = load_model()
    processor = load_processor()

    # 3. LoRA uygula
    model = apply_lora(model)

    # 4. Dataset ve DataLoader hazırla
    # TODO: Dataset ve DataLoader burada oluşturulacak
    log("Dataset hazırlanıyor...")
    train_dataloader = None  # Placeholder
    val_dataloader = None    # Placeholder

    # 5. Eğitimi başlat
    # TODO: Gerçek dataloader'lar hazır olunca aktif edilecek
    # train(model, train_dataloader, val_dataloader)
    log("Not: Dataset ve DataLoader henüz hazır değil.")
    log("Eğitim döngüsü dataloader'lar hazır olunca çalıştırılacak.")

    log("Script tamamlandı.")


if __name__ == "__main__":
    main()
