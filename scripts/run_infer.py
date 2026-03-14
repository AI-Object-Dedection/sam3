"""
run_infer.py — Tek Görsel Üzerinde Tahmin Yapan Script

Bu dosya eğitilmiş modelle tek bir görsel üzerinde segmentation yapar.

Kullanım:
    python scripts/run_infer.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils import log
from src.model import load_model, load_processor
from src.infer import run_inference


def main():
    """Tek görsel üzerinde inference çalıştırır."""

    log("=" * 50)
    log("SAM3 Fine-Tuning — Inference Başlıyor")
    log("=" * 50)

    # 1. Model ve processor'ı yükle
    model = load_model()
    processor = load_processor()

    # 2. Checkpoint yükle (eğitilmiş model varsa)
    # TODO: Eğitilmiş ağırlıkları yükleme kodu eklenecek

    # 3. Test görseli belirle
    # TODO: Kullanıcıdan argüman olarak alınabilir
    test_image = "data/test_image.jpg"  # Örnek yol
    output_path = "outputs/prediction.png"

    # 4. Tahmin yap
    if os.path.exists(test_image):
        mask = run_inference(model, processor, test_image, output_path)
        log(f"Tahmin boyutu: {mask.shape}")
    else:
        log(f"Test görseli bulunamadı: {test_image}")
        log("Lütfen 'data/' klasörüne bir test görseli ekleyin.")

    log("Script tamamlandı.")


if __name__ == "__main__":
    main()
