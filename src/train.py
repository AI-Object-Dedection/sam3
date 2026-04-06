"""
train.py — Eğitim Döngüsü

Bu dosya modelin eğitim sürecini yönetir.

Eğitim döngüsü (training loop) nedir?
Model, veri setini tekrar tekrar görür ve her seferinde:
1. Tahmin yapar  (forward pass)
2. Tahmini doğruyla karşılaştırır  (loss hesaplar)
3. Hatalarından öğrenir  (backward pass + optimizer)

Bu döngüye "epoch" denir. Örneğin 3 epoch = veri setini 3 kez görmek.

SAM3 akışı (tam pipeline):
    Görsel + "damage" metni
    → Vision Encoder   (görseli analiz eder)
    → Text Encoder     (metni analiz eder)
    → DETR Encoder     (görsel + metin birleşir)
    → DETR Decoder     (nesneleri bulur)
    → Mask Decoder     (piksel piksel mask üretir)

Çıktı olarak outputs.semantic_seg kullanıyoruz: (B, 1, 288, 288) boyutlu ham logits.
"""

import torch
import torch.nn as nn

from src.config import Config
from src.evaluate import calculate_iou
from src.utils import log, ensure_dir


def train_one_epoch(model, dataloader, optimizer, loss_fn, scaler, epoch):
    """
    Modeli tek bir epoch boyunca eğitir.

    Args:
        model:      Eğitilecek model (LoRA uygulanmış)
        dataloader: Eğitim verisi (batch'ler halinde)
        optimizer:  Parametre güncelleyici (AdamW)
        loss_fn:    Kayıp fonksiyonu (BCEWithLogitsLoss)
        scaler:     Mixed precision için GradScaler
        epoch:      Kaçıncı epoch olduğu (log için)

    Returns:
        tuple: (ortalama_loss, ortalama_iou)
    """
    model.train()  # Modeli eğitim moduna al
    total_loss = 0.0
    total_iou = 0.0

    for batch_idx, batch in enumerate(dataloader):
        # Batch içindeki tensor'ları GPU'ya taşı
        pixel_values = batch["pixel_values"].to(Config.DEVICE)
        input_ids = batch["input_ids"].to(Config.DEVICE)
        attention_mask = batch["attention_mask"].to(Config.DEVICE)
        gt_mask = batch["ground_truth_mask"].to(Config.DEVICE)  # (B, 288, 288)

        # ---- İleri Geçiş (Forward Pass) ----
        # Mixed precision: bazı hesaplamalar 16-bit ile yapılır → daha az bellek
        with torch.amp.autocast("cuda"):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # semantic_seg: (B, 1, 288, 288) → squeeze → (B, 288, 288)
            # Ham logits (sigmoid uygulanmamış) — loss_fn bunu bekliyor
            pred_mask = outputs.semantic_seg.squeeze(1)

            # ---- Loss Hesaplama ----
            # BCEWithLogitsLoss: her piksel için "hasar var mı?" sorusunun cevabını değerlendirir
            # Otomatik olarak sigmoid uygular, ayrıca yapmanıza gerek yok
            loss = loss_fn(pred_mask, gt_mask)

        # ---- Geri Yayılım (Backward Pass) ----
        optimizer.zero_grad()          # Önceki gradientleri temizle
        scaler.scale(loss).backward()  # Gradientleri hesapla (mixed precision ile)
        scaler.step(optimizer)         # Parametreleri güncelle
        scaler.update()                # Scaler'ı güncelle

        # İstatistikleri biriktir
        iou = calculate_iou(pred_mask.detach(), gt_mask)
        total_loss += loss.item()
        total_iou += iou

        # Her 10 batch'te bir durum bilgisi yazdır
        if (batch_idx + 1) % 10 == 0:
            log(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} "
                f"| Loss: {loss.item():.4f} | IoU: {iou:.4f}")

    avg_loss = total_loss / max(len(dataloader), 1)
    avg_iou = total_iou / max(len(dataloader), 1)
    return avg_loss, avg_iou


def train(model, train_dataloader, val_dataloader=None):
    """
    Tüm eğitim sürecini yönetir.

    Args:
        model:             Eğitilecek model (LoRA uygulanmış)
        train_dataloader:  Eğitim verisi
        val_dataloader:    Doğrulama verisi (opsiyonel)
    """
    from src.evaluate import evaluate

    log("Eğitim başlıyor...")
    log(f"Epoch sayısı   : {Config.NUM_EPOCHS}")
    log(f"Learning rate  : {Config.LEARNING_RATE}")
    log(f"Cihaz          : {Config.DEVICE}")
    log(f"Text prompt    : '{Config.TEXT_PROMPT}'")
    log(f"Mask boyutu    : {Config.MASK_OUTPUT_SIZE}x{Config.MASK_OUTPUT_SIZE}")

    # Checkpoint klasörünü oluştur
    ensure_dir(Config.CHECKPOINT_DIR)

    # Kayıp fonksiyonu: Binary Cross Entropy (her piksel için hasar var/yok)
    loss_fn = nn.BCEWithLogitsLoss()

    # Optimizer: sadece LoRA parametrelerini eğit (frozen parametreler hariç)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE,
    )

    # Mixed precision scaler: GPU belleğinden tasarruf sağlar
    # CPU'da eğitiliyorsa scaler devre dışı
    use_amp = (Config.DEVICE == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Her epoch için eğitim yap
    for epoch in range(Config.NUM_EPOCHS):
        log(f"\n{'='*50}")
        log(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} başladı")

        # Eğitim
        train_loss, train_iou = train_one_epoch(
            model, train_dataloader, optimizer, loss_fn, scaler, epoch
        )
        log(f"Epoch {epoch+1} [train] | Loss: {train_loss:.4f} | IoU: {train_iou:.4f}")

        # Validation (eğer val_dataloader verilmişse)
        if val_dataloader is not None:
            results = evaluate(model, val_dataloader, loss_fn)
            log(f"Epoch {epoch+1} [val]   | Loss: {results['mean_loss']:.4f} "
                f"| IoU: {results['mean_iou']:.4f}")

        # Checkpoint kaydet — sadece LoRA adapter ağırlıklarını kaydet
        # (base model HuggingFace'ten yüklenebilir, sadece öğrenilen kısım kaydedilir)
        adapter_path = f"{Config.CHECKPOINT_DIR}/epoch_{epoch+1}_lora"
        model.save_pretrained(adapter_path)
        log(f"LoRA adapter kaydedildi: {adapter_path}")

    log(f"\n{'='*50}")
    log("Eğitim tamamlandı!")
