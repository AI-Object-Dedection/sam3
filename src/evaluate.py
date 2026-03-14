"""
evaluate.py — Model Değerlendirme

Bu dosya modelin ne kadar iyi öğrendiğini ölçer.

Değerlendirme neden gerekli?
Modelin eğitim verisinde iyi performans göstermesi yetmez.
Daha önce hiç görmediği verilerde de iyi tahmin yapması gerekir.
Buna "genelleme" (generalization) denir.

Bu dosya:
1. Validation veya test verisi üzerinde modeli çalıştırır
2. Tahminleri doğru cevaplarla karşılaştırır
3. Metrik hesaplar (IoU gibi)

IoU (Intersection over Union) nedir?
Modelin tahmin ettiği mask ile gerçek mask'ın ne kadar örtüştüğünü ölçer.
1.0 = mükemmel eşleşme, 0.0 = hiç eşleşme yok.
"""

import numpy as np
import torch

from src.config import Config
from src.utils import log


def calculate_iou(prediction, ground_truth):
    """
    İki binary mask arasındaki IoU (Intersection over Union) değerini hesaplar.

    Basit anlatım:
    - Kesişim: İkisinin de 1 olduğu pikseller
    - Birleşim: En az birinin 1 olduğu pikseller
    - IoU = Kesişim / Birleşim

    Args:
        prediction: Modelin tahmini (binary mask, numpy array)
        ground_truth: Gerçek mask (binary mask, numpy array)

    Returns:
        float: IoU değeri (0.0 ile 1.0 arası)
    """
    # İkisinin de 1 olduğu yerler (kesişim)
    intersection = np.logical_and(prediction, ground_truth).sum()

    # En az birinin 1 olduğu yerler (birleşim)
    union = np.logical_or(prediction, ground_truth).sum()

    # Birleşim 0 ise (ikisi de tamamen boş), IoU = 0
    if union == 0:
        return 0.0

    return intersection / union


def evaluate(model, dataloader):
    """
    Modeli validation/test verisi üzerinde değerlendirir.

    Args:
        model: Değerlendirilecek model
        dataloader: Validation veya test verisi

    Returns:
        dict: Metrik sonuçları
    """
    model.eval()  # Modeli değerlendirme moduna al (dropout vb. kapatılır)
    iou_scores = []

    log("Değerlendirme başlıyor...")

    with torch.no_grad():  # Gradient hesaplamayı kapat (daha hızlı çalışır)
        for batch_idx, batch in enumerate(dataloader):
            # TODO: Gerçek tahmin kodu burada implement edilecek
            # 1. Görseli modele ver
            # 2. Modelin mask tahminini al
            # 3. Tahmini gerçek mask ile karşılaştır

            # Placeholder: gerçek implementasyon daha sonra eklenecek
            pass

    # Sonuçları özetle
    if iou_scores:
        mean_iou = np.mean(iou_scores)
    else:
        mean_iou = 0.0

    log(f"Değerlendirme tamamlandı | Ortalama IoU: {mean_iou:.4f}")

    results = {
        "mean_iou": mean_iou,
        "num_samples": len(iou_scores),
    }

    return results
