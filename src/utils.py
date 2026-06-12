"""
utils.py — Yardımcı Fonksiyonlar

Bu dosya projede tekrar tekrar kullanılan küçük araçları toplar.
Örneğin:
- Rastgeleliği sabitlemek (her çalıştırmada aynı sonuç almak için)
- Klasör oluşturmak
- Zamanlı mesaj yazdırmak

Bu dosya bir "alet kutusu" gibidir — büyük iş yapmaz ama her yerde lazım olur.
"""

import json
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
    # flush=True: cikti dosyaya yonlendirildiginde (arka plan egitim) Python
    # ciktiyi tamponda biriktirir; flush ile her satir ANINDA log dosyasina yazilir,
    # boylece "Adim 7" ile ilerlemeyi gercek zamanli takip edebilirsin.
    print(f"[{now}] {message}", flush=True)


# Eğitim durumunu (kaçıncı epoch'tayız, en iyi loss vb.) tutan dosyanın adı.
# Checkpoint klasörünün içinde durur, böylece checkpointlerle birlikte taşınır.
TRAINING_STATE_FILE = "training_state.json"


def save_training_state(checkpoint_dir, state):
    """
    Eğitim durumunu checkpoint klasörüne JSON olarak kaydeder.

    Neden gerekli?
    Eğitim yarıda kesilirse (bağlantı kopması, oturum bitmesi),
    bir sonraki çalıştırmada kaldığımız yerden devam edebilmek için
    "kaçıncı epoch'ta kaldık" bilgisini saklamamız gerekir.

    Args:
        checkpoint_dir: Durumun kaydedileceği klasör
        state: Kaydedilecek bilgiler (sözlük), ör: {"last_epoch": 3, ...}
    """
    ensure_dir(checkpoint_dir)
    path = os.path.join(checkpoint_dir, TRAINING_STATE_FILE)
    # Atomik yazım: önce geçici dosyaya yaz, sonra yerine taşı.
    # Neden? Drive yedekleme döngüsü (her 120 sn) tam bu dosyayı kopyalarken
    # yazma yarıda kalırsa dosya bozulur ve resume "kayıt yok" sanır.
    # Geçici dosya + os.replace ile dosya ya eski ya yeni halde kalır, asla yarım.
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp_path, path)


def load_training_state(checkpoint_dir):
    """
    Daha önce kaydedilmiş eğitim durumunu okur.

    Args:
        checkpoint_dir: Durumun aranacağı klasör

    Returns:
        dict: Kayıtlı durum bilgisi. Hiç kayıt yoksa None döner
        (yani sıfırdan başlanacak demektir).
    """
    path = os.path.join(checkpoint_dir, TRAINING_STATE_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Dosya bozuksa sıfırdan başla (güvenli taraf)
        return None
