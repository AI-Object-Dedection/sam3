"""
evaluate.py — Model Değerlendirme

Bu dosya modelin ne kadar iyi öğrendiğini ölçer.

IoU (Intersection over Union) nedir?
Modelin tahmin ettiği mask ile gerçek mask'ın ne kadar örtüştüğünü ölçer.
1.0 = mükemmel eşleşme, 0.0 = hiç eşleşme yok.

Neden loss yetmez, IoU da bakıyoruz?
Loss sadece "ne kadar hata yaptın" der. IoU ise "mask şekli ne kadar doğru"
sorusunu cevaplar — bu görsel olarak daha anlamlıdır.
"""

import numpy as np
import torch

from src.config import Config
from src.utils import log


def calculate_iou(pred_logits, gt_mask, threshold=0.5):
    """
    Tahmin ve gerçek mask arasındaki IoU değerini hesaplar.

    Args:
        pred_logits: Modelin ham çıktısı — sigmoid uygulanmamış (tensor)
        gt_mask:     Gerçek mask (0/1 değerler, tensor)
        threshold:   Sigmoid sonrası bu değerin üstü = hasar var (varsayılan: 0.5)

    Returns:
        float: IoU değeri (0.0 ile 1.0 arası)
    """
    # Ham logits'i önce sigmoid ile 0-1 arasına getir, sonra eşik uygula
    pred_binary = (pred_logits.sigmoid() > threshold).float()

    # Kesişim: ikisinin de 1 olduğu pikseller
    intersection = (pred_binary * gt_mask).sum()

    # Birleşim: en az birinin 1 olduğu pikseller
    union = pred_binary.sum() + gt_mask.sum() - intersection

    if union == 0:
        # İkisi de tamamen boşsa (hiç hasar yok) → mükemmel eşleşme
        return 1.0

    return (intersection / union).item()


def evaluate(model, dataloader, loss_fn=None):
    """
    Modeli validation verisi üzerinde değerlendirir.

    Args:
        model:      Değerlendirilecek model
        dataloader: Validation verisi
        loss_fn:    Kayıp fonksiyonu (opsiyonel, verilirse loss da hesaplanır)

    Returns:
        dict: {"mean_iou": float, "mean_loss": float, "num_samples": int}
    """
    import torch.nn as nn

    model.eval()  # Dropout vb. kapat — sadece tahmin yap
    iou_scores = []
    loss_scores = []

    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()

    log("Validation başlıyor...")

    with torch.no_grad():  # Gradient hesaplama — gerek yok, zaman ve bellek tasarrufu
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(Config.DEVICE)
            input_ids = batch["input_ids"].to(Config.DEVICE)
            attention_mask = batch["attention_mask"].to(Config.DEVICE)
            gt_mask = batch["ground_truth_mask"].to(Config.DEVICE)

            # Forward pass (mixed precision ile)
            with torch.amp.autocast("cuda", enabled=(Config.DEVICE == "cuda")):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                # semantic_seg: (B, 1, 288, 288) → (B, 288, 288)
                pred_mask = outputs.semantic_seg.squeeze(1)
                loss = loss_fn(pred_mask, gt_mask)

            iou = calculate_iou(pred_mask, gt_mask)
            iou_scores.append(iou)
            loss_scores.append(loss.item())

    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    mean_loss = float(np.mean(loss_scores)) if loss_scores else 0.0

    log(f"Validation tamamlandı | Loss: {mean_loss:.4f} | IoU: {mean_iou:.4f} "
        f"| Örnek sayısı: {len(iou_scores)}")

    return {
        "mean_iou"   : mean_iou,
        "mean_loss"  : mean_loss,
        "num_samples": len(iou_scores),
    }
