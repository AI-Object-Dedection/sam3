"""
utils.py — Yardımcı Fonksiyonlar

Bu dosya projede tekrar tekrar kullanılan küçük araçları toplar.
Örneğin:
- Rastgeleliği sabitlemek (her çalıştırmada aynı sonuç almak için)
- Klasör oluşturmak
- Zamanlı mesaj yazdırmak

Bu dosya bir "alet kutusu" gibidir — büyük iş yapmaz ama her yerde lazım olur.
"""

import os
import random
from datetime import datetime

import numpy as np
import torch


def set_seed(seed=42):
    """
    Rastgeleliği sabitler.

    Neden gerekli?
    Makine öğrenmesinde birçok işlem rastgeledir (ağırlık başlatma, veri karıştırma vb.).
    Seed sabitlersek, her çalıştırmada aynı sonuçları alırız.
    Bu da sonuçların tekrarlanabilir olmasını sağlar.

    Args:
        seed: Sabitlenecek sayı (varsayılan: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"[utils] Seed sabitlendi: {seed}")


def ensure_dir(path):
    """
    Verilen klasör yolu yoksa oluşturur.

    Neden gerekli?
    Eğitim sırasında checkpoint veya output kaydetmek istediğimizde
    klasörün önceden var olması gerekir. Bu fonksiyon bunu garanti eder.

    Args:
        path: Oluşturulacak klasörün yolu
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[utils] Klasör oluşturuldu: {path}")


def log(message):
    """
    Zamanlı mesaj yazdırır.

    Neden gerekli?
    Eğitim uzun sürebilir. Her adımda saat bilgisi görmek,
    süreyi takip etmeyi kolaylaştırır.

    Args:
        message: Yazdırılacak mesaj
    """
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}")
