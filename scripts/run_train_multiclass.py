"""
run_train_multiclass.py — Multi-Class Eğitim

Binary "damage" yerine her hasar sınıfı ("Crack", "Rust" vb.) için
ayrı ayrı eğitim yapar. Model artık hangi bölgede hangi hasar olduğunu öğrenir.

Kullanım:
    python scripts/run_train_multiclass.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader  # noqa: E402
from peft import PeftModel               # noqa: E402

from src.config import Config            # noqa: E402
from src.utils import set_seed, log      # noqa: E402
from src.model import load_model, load_processor  # noqa: E402
from src.lora import apply_lora          # noqa: E402
from src.dataset import DACL10KDataset  # noqa: E402
from src.train import train_multiclass  # noqa: E402


def main():
    log("=" * 50)
    log("SAM3 Multi-Class Egitim Basliyor")
    log(f"Cihaz: {Config.DEVICE}")
    log("=" * 50)

    set_seed()

    # 1. Model ve processor yukle
    model     = load_model()
    processor = load_processor()

    # 2. Onceki en iyi checkpoint'ten baslat (epoch_4_lora)
    #    Sifirdan egitmek yerine mevcut bilgiyi kullan — cok daha hizli
    onceki_checkpoint = "checkpoints/epoch_4_lora"
    if os.path.exists(onceki_checkpoint):
        log(f"Onceki checkpoint yukleniyor: {onceki_checkpoint}")
        model = PeftModel.from_pretrained(model, onceki_checkpoint, is_trainable=True)
        log("Checkpoint yuklendi — onceki bilgiden devam edilecek.")
    else:
        log("Onceki checkpoint bulunamadi — LoRA sifirdan uygulanacak.")
        model = apply_lora(model)

    # 3. Dataset
    log("Dataset hazirlanıyor...")
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

    train_dataloader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=True, pin_memory=True, num_workers=0,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, pin_memory=True, num_workers=0,
    )

    log(f"Egitim batch: {len(train_dataloader)} | Val batch: {len(val_dataloader)}")

    # 4. Multi-class egitimi baslat
    train_multiclass(model, train_dataloader, val_dataloader, processor)


if __name__ == "__main__":
    main()
