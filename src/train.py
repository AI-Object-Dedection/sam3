"""
train.py — Eğitim Döngüsü

Bu dosya modelin eğitim sürecini yönetir.

Eğitim döngüsü (training loop) nedir?
Model, veri setini tekrar tekrar görür ve her seferinde:
1. Tahmin yapar
2. Tahmini doğruyla karşılaştırır (loss hesaplar)
3. Hatalarından öğrenir (parametreleri günceller)

Bu döngüye "epoch" denir. Örneğin 10 epoch = veri setini 10 kez görmek.

Bu dosya şu ana fonksiyonları içerir:
- train_one_epoch() → tek bir epoch eğitim
- train()           → tüm eğitim sürecini yönetir
"""

import torch

from src.config import Config
from src.utils import log, ensure_dir


def train_one_epoch(model, dataloader, optimizer, epoch):
    """
    Modeli tek bir epoch boyunca eğitir.

    Args:
        model: Eğitilecek model (LoRA uygulanmış)
        dataloader: Eğitim verisi (batch'ler halinde)
        optimizer: Parametre güncelleyici (AdamW gibi)
        epoch: Kaçıncı epoch olduğu (log için)

    Returns:
        float: Bu epoch'taki ortalama loss değeri
    """
    model.train()  # Modeli eğitim moduna al
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        # ---- İleri Geçiş (Forward Pass) ----
        # Model tahmin yapar
        # TODO: Gerçek forward pass burada implement edilecek

        # ---- Loss Hesaplama ----
        # Tahmini doğruyla karşılaştır
        # TODO: Loss hesaplama burada implement edilecek
        loss = torch.tensor(0.0)  # Placeholder

        # ---- Geri Yayılım (Backward Pass) ----
        # Hataları geri yayarak gradientleri hesapla
        optimizer.zero_grad()  # Önceki gradientleri temizle
        loss.backward()        # Gradientleri hesapla
        optimizer.step()       # Parametreleri güncelle

        total_loss += loss.item()

        # Her 10 batch'te bir durum bilgisi yazdır
        if (batch_idx + 1) % 10 == 0:
            log(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss


def train(model, train_dataloader, val_dataloader=None):
    """
    Tüm eğitim sürecini yönetir.

    Args:
        model: Eğitilecek model
        train_dataloader: Eğitim verisi
        val_dataloader: Doğrulama verisi (opsiyonel)
    """
    log("Eğitim başlıyor...")
    log(f"Epoch sayısı: {Config.NUM_EPOCHS}")
    log(f"Learning rate: {Config.LEARNING_RATE}")
    log(f"Cihaz: {Config.DEVICE}")

    # Checkpoint klasörünü oluştur
    ensure_dir(Config.CHECKPOINT_DIR)

    # Optimizer: parametreleri güncelleyen algoritma
    # AdamW, modern derin öğrenmede en yaygın kullanılan optimizer'dır
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
    )

    # Her epoch için eğitim yap
    for epoch in range(Config.NUM_EPOCHS):
        log(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} başladı")

        # Eğitim
        train_loss = train_one_epoch(model, train_dataloader, optimizer, epoch)
        log(f"Epoch {epoch+1} tamamlandı | Ortalama Loss: {train_loss:.4f}")

        # Validation (eğer val_dataloader verilmişse)
        if val_dataloader is not None:
            # TODO: evaluate fonksiyonu çağrılacak
            log(f"Epoch {epoch+1} | Validation çalıştırılacak...")

        # Her epoch sonunda modeli kaydet
        checkpoint_path = f"{Config.CHECKPOINT_DIR}/epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        log(f"Checkpoint kaydedildi: {checkpoint_path}")

    log("Eğitim tamamlandı!")
