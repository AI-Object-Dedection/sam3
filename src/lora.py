"""
lora.py — LoRA Fine-Tuning Katmanı

Bu dosya modele LoRA uygular.

LoRA (Low-Rank Adaptation) nedir?
Normalde bir modeli eğitmek için TÜM parametreleri güncellemek gerekir.
SAM3 gibi büyük modellerde bu milyonlarca parametre demektir — çok maliyetli.

LoRA, modelin sadece küçük bir kısmına "ek katmanlar" (adaptörler) ekler.
Eğitim sırasında sadece bu ek katmanlar güncellenir.
Orijinal model ağırlıkları donmuş (frozen) kalır.

Avantajları:
- Çok daha az bellek kullanır
- Çok daha hızlı eğitim
- Orijinal model bozulmaz

PEFT nedir?
HuggingFace'in "Parameter-Efficient Fine-Tuning" kütüphanesi.
LoRA ve benzeri yöntemleri kolayca uygulamayı sağlar.
"""

import os

from peft import LoraConfig, PeftModel, get_peft_model

from src.config import Config
from src.utils import load_training_state, log


def apply_lora(model):
    """
    Modele LoRA adaptörlerini ekler.

    Adımlar:
    1. LoRA konfigürasyonu oluştur (rank, alpha, dropout)
    2. Bu konfigürasyonu modele uygula
    3. Artık sadece LoRA parametreleri eğitilebilir

    Args:
        model: Orijinal SAM3 modeli

    Returns:
        model: LoRA uygulanmış model (sadece LoRA katmanları eğitilebilir)
    """
    log("LoRA konfigürasyonu oluşturuluyor...")

    # LoRA ayarlarını tanımla
    lora_config = LoraConfig(
        r=Config.LORA_RANK,             # Rank: ek katmanların boyutu
        lora_alpha=Config.LORA_ALPHA,    # Alpha: ölçekleme faktörü
        lora_dropout=Config.LORA_DROPOUT,  # Dropout: aşırı öğrenmeyi önler
        target_modules=["q_proj", "v_proj"],  # Hangi katmanlara LoRA uygulanacak
    )

    # LoRA'yı modele uygula
    model = get_peft_model(model, lora_config)

    log("LoRA uygulandı.")
    print_trainable_params(model)

    return model


def _resume_checkpoint_path():
    """
    Devam edilebilecek son checkpoint'in yolunu bulur.

    Eğitim durumu dosyasındaki (training_state.json) `last_epoch` bilgisine
    bakar ve ona karşılık gelen `epoch_{N}_lora` klasörünü arar.

    Returns:
        str: Checkpoint klasörünün yolu. Devam edilecek bir şey yoksa None.
    """
    state = load_training_state(Config.CHECKPOINT_DIR)
    if state is None:
        return None

    last_epoch = state.get("last_epoch", 0)
    if last_epoch <= 0:
        return None

    path = os.path.join(Config.CHECKPOINT_DIR, f"epoch_{last_epoch}_lora")
    return path if os.path.isdir(path) else None


def load_or_apply_lora(model):
    """
    Önceki bir checkpoint varsa onun üzerine devam eder, yoksa sıfırdan LoRA uygular.

    Bu fonksiyon sayesinde eğitim yarıda kesilse bile (oturum kapanması,
    bağlantı kopması) tekrar çalıştırıldığında kaldığı yerden devam eder —
    her seferinde sıfırdan başlamaz.

    Args:
        model: Base SAM3 modeli (henüz LoRA uygulanmamış)

    Returns:
        model: LoRA'lı model — ya önceki ağırlıklarla ya da sıfırdan
    """
    ckpt = _resume_checkpoint_path()
    if ckpt is not None:
        log(f"Önceki checkpoint bulundu, ÜSTÜNE DEVAM ediliyor: {ckpt}")
        # is_trainable=True → yüklenen LoRA katmanları yine eğitilebilir kalır
        model = PeftModel.from_pretrained(model, ckpt, is_trainable=True)
        print_trainable_params(model)
        return model

    log("Önceki checkpoint yok — LoRA sıfırdan başlatılıyor.")
    return apply_lora(model)


def print_trainable_params(model):
    """
    Modeldeki eğitilebilir parametre sayısını gösterir.

    Bu fonksiyon LoRA'nın ne kadar verimli olduğunu görmenizi sağlar.
    Toplam parametre sayısı ile eğitilebilir parametre sayısını karşılaştırır.

    Args:
        model: LoRA uygulanmış model
    """
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        total_params += param.numel()       # Toplam parametre sayısı
        if param.requires_grad:
            trainable_params += param.numel()  # Eğitilebilir parametre sayısı

    # Yüzde hesapla
    percentage = 100 * trainable_params / total_params

    print(f"\n{'='*50}")
    print(f"Toplam parametreler     : {total_params:,}")
    print(f"Eğitilebilir parametreler: {trainable_params:,}")
    print(f"Eğitilebilir oran       : {percentage:.2f}%")
    print(f"{'='*50}\n")
