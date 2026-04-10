"""
run_infer.py — Inference Script

Eğitilmiş modeli görseller üzerinde çalıştırır.
İki mod:
  - binary    : hasar var/yok (hızlı)
  - multiclass: 19 sınıf ayrı ayrı, renkli (yavaş ama bilgilendirici)

Kullanım:
    python scripts/run_infer.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peft import PeftModel  # noqa: E402

from src.config import Config  # noqa: E402
from src.utils import log  # noqa: E402
from src.model import load_model, load_processor  # noqa: E402
from src.infer import run_inference, run_multiclass_inference  # noqa: E402

# "binary" veya "multiclass"
MOD = "multiclass"

# Kaç görsel işlenecek
GORSEL_SAYISI = 5


def main():
    log("=" * 50)
    log(f"SAM3 — Inference ({MOD} mod)")
    log("=" * 50)

    # 1. Model ve processor yukle
    model = load_model()
    processor = load_processor()

    # 2. En iyi LoRA checkpoint yukle (mc_epoch_1 — val IoU en yuksek, overfitting yok)
    checkpoint = "checkpoints/mc_epoch_1_lora"
    if not os.path.exists(checkpoint):
        log(f"HATA: Checkpoint bulunamadi: {checkpoint}")
        return

    log(f"LoRA adapter yukleniyor: {checkpoint}")
    model = PeftModel.from_pretrained(model, checkpoint)
    log("LoRA adapter yuklendi.")

    # 3. Test gorsellerini sec — validation setinden ilk N gorsel
    val_klasoru = Config.VAL_IMAGES
    gorseller = sorted([
        f for f in os.listdir(val_klasoru) if f.endswith(".npy")
    ])[:GORSEL_SAYISI]

    if not gorseller:
        log(f"HATA: {val_klasoru} klasorunde gorsel bulunamadi.")
        return

    log(f"{len(gorseller)} gorsel uzerinde {MOD} inference yapilacak...")

    # 4. Her gorsel icin tahmin yap
    for gorsel_adi in gorseller:
        gorsel_yolu = os.path.join(val_klasoru, gorsel_adi)

        if MOD == "multiclass":
            run_multiclass_inference(
                model=model,
                processor=processor,
                gorsel_yolu=gorsel_yolu,
                cikti_yolu=Config.OUTPUT_DIR,
            )
        else:
            run_inference(
                model=model,
                processor=processor,
                gorsel_yolu=gorsel_yolu,
                cikti_yolu=Config.OUTPUT_DIR,
            )

    log("=" * 50)
    log(f"Tamamlandi! Sonuclar: {Config.OUTPUT_DIR}")
    log("=" * 50)


if __name__ == "__main__":
    main()
