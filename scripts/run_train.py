"""
run_train.py — Eğitimi Başlatan Script

Bu dosya src/ klasöründeki modülleri çağırarak eğitim sürecini başlatır.

Kullanım:
    python scripts/run_train.py

NOT: GPU olmadan çok yavaş çalışır.
Eğitim için Google Colab'da colab_faz4.ipynb kullanılması önerilir.
"""

import sys
import os

# Proje kök dizinini Python'un arama yoluna ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader  # noqa: E402

from src.config import Config  # noqa: E402
from src.utils import set_seed, log  # noqa: E402
from src.model import load_model, load_processor  # noqa: E402
from src.lora import load_or_apply_lora  # noqa: E402
from src.dataset import DACL10KDataset  # noqa: E402
from src.train import train  # noqa: E402


def main():
    """Eğitim sürecini adım adım çalıştırır."""

    log("=" * 50)
    log("SAM3 Fine-Tuning — Eğitim Başlıyor")
    log(f"Cihaz: {Config.DEVICE}")
    log("=" * 50)

    # 1. Rastgeleliği sabitle — her çalıştırmada aynı sonuç
    set_seed()

    # 2. Model ve processor'ı yükle
    model = load_model()
    processor = load_processor()

    # 3. LoRA uygula — önceki checkpoint varsa üstüne devam, yoksa sıfırdan
    model = load_or_apply_lora(model)

    # 4. Dataset oluştur
    log("Dataset hazırlanıyor...")
    log(f"Eğitim örnek sayısı    : {Config.MAX_TRAIN_SAMPLES or 'tümü'}")
    log(f"Validation örnek sayısı: {Config.MAX_VAL_SAMPLES or 'tümü'}")
    log(f"Augmentasyon aktif mi  : {Config.USE_AUGMENTATION}")

    train_dataset = DACL10KDataset(
        images_dir=Config.TRAIN_IMAGES,
        annotations_dir=Config.TRAIN_MASKS,
        processor=processor,
        max_samples=Config.MAX_TRAIN_SAMPLES,
        is_train=True,
        use_augmentation=Config.USE_AUGMENTATION,
    )

    val_dataset = DACL10KDataset(
        images_dir=Config.VAL_IMAGES,
        annotations_dir=Config.VAL_MASKS,
        processor=processor,
        max_samples=Config.MAX_VAL_SAMPLES,
        is_train=False,
        use_augmentation=False,
    )

    # 5. DataLoader oluştur — veriyi batch'ler halinde modele sunar
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,        # Her epoch'ta veriyi karıştır
        pin_memory=True,     # GPU'ya veri aktarımını hızlandırır
        num_workers=0,       # Windows'ta 0 bırak (çoklu process sorun çıkarır)
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    log(f"Eğitim batch sayısı    : {len(train_dataloader)}")
    log(f"Validation batch sayısı: {len(val_dataloader)}")

    # 6. Eğitimi başlat
    train(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
