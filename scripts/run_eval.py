"""
run_eval.py — Değerlendirmeyi Başlatan Script

Eğitilmiş modeli (LoRA adapter) yükler ve validation verisi üzerinde değerlendirir.

Kullanım:
    python scripts/run_eval.py

NOT: Önce run_train.py ile eğitim yapılmış olmalı,
checkpoints/ klasöründe en az bir epoch_X_lora klasörü bulunmalıdır.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader  # noqa: E402
from peft import PeftModel  # noqa: E402

from src.config import Config  # noqa: E402
from src.utils import set_seed, log  # noqa: E402
from src.model import load_model, load_processor  # noqa: E402
from src.dataset import DACL10KDataset  # noqa: E402
from src.evaluate import evaluate  # noqa: E402


def main():
    """Değerlendirme sürecini çalıştırır."""

    log("=" * 50)
    log("SAM3 Fine-Tuning — Değerlendirme Başlıyor")
    log(f"Cihaz: {Config.DEVICE}")
    log("=" * 50)

    set_seed()

    # 1. Base model ve processor'ı yükle
    model = load_model()
    processor = load_processor()

    # 2. Eğitilmiş LoRA adapter'ı yükle
    # En son kaydedilen epoch'u bul
    checkpoint_dir = Config.CHECKPOINT_DIR
    adapter_klasorleri = sorted([
        d for d in os.listdir(checkpoint_dir)
        if d.startswith("epoch_") and d.endswith("_lora")
        and os.path.isdir(os.path.join(checkpoint_dir, d))
    ])

    if not adapter_klasorleri:
        log(f"HATA: {checkpoint_dir} klasöründe LoRA checkpoint bulunamadı!")
        log("Önce 'python scripts/run_train.py' çalıştırın.")
        return

    # En son epoch'u kullan
    son_adapter = os.path.join(checkpoint_dir, adapter_klasorleri[-1])
    log(f"LoRA adapter yükleniyor: {son_adapter}")
    model = PeftModel.from_pretrained(model, son_adapter)
    log("LoRA adapter yüklendi.")

    # 3. Validation dataset ve DataLoader
    log("Validation dataset hazırlanıyor...")
    val_dataset = DACL10KDataset(
        images_dir=Config.VAL_IMAGES,
        annotations_dir=Config.VAL_MASKS,
        processor=processor,
        max_samples=Config.MAX_VAL_SAMPLES,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
    )

    log(f"Validation batch sayısı: {len(val_dataloader)}")

    # 4. Değerlendirmeyi çalıştır
    results = evaluate(model, val_dataloader)

    log("=" * 50)
    log("Sonuçlar:")
    log(f"  Ortalama IoU  : {results['mean_iou']:.4f}")
    log(f"  Ortalama Loss : {results['mean_loss']:.4f}")
    log(f"  Örnek sayısı  : {results['num_samples']}")
    log("=" * 50)


if __name__ == "__main__":
    main()
