"""
infer.py — Tek Görsel Üzerinde Tahmin

Bu dosya eğitilmiş modeli tek bir görsel üzerinde çalıştırır.

Ne işe yarar?
Eğitim bittikten sonra modelin gerçekten çalışıp çalışmadığını
hızlıca test etmek için kullanılır.

Kullanım senaryosu:
"Şu köprü fotoğrafında hasar var mı?" sorusuna cevap üretir.
Modele bir görsel verilir, model bir segmentation mask döndürür.
"""

import os

import numpy as np
import torch
from PIL import Image

from src.config import Config
from src.utils import log, ensure_dir


def run_inference(model, processor, image_path, output_path=None):
    """
    Tek bir görsel üzerinde segmentation tahmini yapar.

    Args:
        model: Eğitilmiş model
        processor: SAM3 Processor
        image_path: Tahmin yapılacak görselin yolu
        output_path: Sonucun kaydedileceği yol (opsiyonel)

    Returns:
        numpy array: Tahmin edilen mask
    """
    log(f"Tahmin yapılıyor: {image_path}")

    # Görseli oku
    image = Image.open(image_path).convert("RGB")

    # Görseli modele uygun formata çevir
    inputs = processor(images=image, return_tensors="pt")

    # Tensörleri doğru cihaza taşı
    inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}

    # Tahmin yap (gradient hesaplamaya gerek yok)
    model.eval()
    with torch.no_grad():
        # TODO: Gerçek inference kodu burada implement edilecek
        # outputs = model(**inputs)
        # predicted_mask = outputs'tan mask çıkarma işlemi
        pass

    # Placeholder mask (gerçek implementasyon daha sonra gelecek)
    predicted_mask = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)

    # Sonucu kaydet (eğer output_path verilmişse)
    if output_path is not None:
        ensure_dir(os.path.dirname(output_path))
        mask_image = Image.fromarray((predicted_mask * 255).astype(np.uint8))
        mask_image.save(output_path)
        log(f"Tahmin kaydedildi: {output_path}")

    log("Tahmin tamamlandı.")
    return predicted_mask
