"""
run_eval.py — Değerlendirmeyi Başlatan Script

Bu dosya eğitilmiş modeli validation/test verisi üzerinde değerlendirir.

Kullanım:
    python scripts/run_eval.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils import set_seed, log
from src.model import load_model, load_processor
from src.evaluate import evaluate


def main():
    """Değerlendirme sürecini çalıştırır."""

    log("=" * 50)
    log("SAM3 Fine-Tuning — Değerlendirme Başlıyor")
    log("=" * 50)

    set_seed()

    # 1. Model ve processor'ı yükle
    model = load_model()
    processor = load_processor()

    # 2. Checkpoint yükle (eğer eğitilmiş model varsa)
    # TODO: Eğitilmiş ağırlıkları yükleme kodu eklenecek
    log("Not: Checkpoint yükleme henüz implement edilmedi.")

    # 3. Validation DataLoader hazırla
    # TODO: DataLoader burada oluşturulacak
    val_dataloader = None  # Placeholder

    # 4. Değerlendirmeyi çalıştır
    # TODO: Gerçek dataloader hazır olunca aktif edilecek
    # results = evaluate(model, val_dataloader)
    log("Not: DataLoader henüz hazır değil.")

    log("Script tamamlandı.")


if __name__ == "__main__":
    main()
