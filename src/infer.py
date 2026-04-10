"""
infer.py — Tek Görsel Üzerinde Tahmin

Bu dosya eğitilmiş modeli bir görsel üzerinde çalıştırır ve
hasar maskini görselleştirir.

Nasıl çalışır?
1. Görsel okunur (.jpg veya .npy)
2. SAM3 processor ile modele uygun formata çevrilir
3. "damage" metin ipucu ile modele verilir
4. Model semantic_seg çıktısı üretir (288x288 logits)
5. Sigmoid + eşik uygulanarak binary mask elde edilir
6. Orijinal görsel boyutuna geri büyütülür
7. Sonuç görselleştirilir: orijinal / mask / overlay
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.config import Config
from src.dataset import DACL10K_CLASSES
from src.utils import log, ensure_dir

# Her sınıf için renk (RGB, 0-255)
# 19 sınıf için 19 farklı renk
SINIF_RENKLERI = [
    (255, 0,   0  ),  # Crack          — kırmızı
    (255, 80,  0  ),  # ACrack         — turuncu kırmızı
    (255, 165, 0  ),  # Spalling       — turuncu
    (255, 220, 0  ),  # Efflorescence  — sarı
    (180, 255, 0  ),  # ExposedRebars  — sarı yeşil
    (0,   255, 0  ),  # Cavity         — yeşil
    (0,   255, 150),  # Restformwork   — yeşil cyan
    (0,   255, 255),  # Rockpocket     — cyan
    (0,   180, 255),  # Hollowareas    — açık mavi
    (0,   80,  255),  # Rust           — mavi
    (100, 0,   255),  # Weathering     — mor mavi
    (180, 0,   255),  # Graffiti       — mor
    (255, 0,   200),  # Wetspot        — pembe mor
    (255, 0,   100),  # Bearing        — pembe
    (150, 75,  0  ),  # Drainage       — kahve
    (128, 128, 128),  # EJoint         — gri
    (200, 200, 200),  # JTape          — açık gri
    (255, 180, 180),  # PEquipment     — açık pembe
    (180, 255, 180),  # WConccor       — açık yeşil
]


def gorsel_oku(gorsel_yolu):
    """
    Görseli okur — .jpg veya .npy formatını destekler.

    Args:
        gorsel_yolu: Görsel dosyasının yolu

    Returns:
        PIL.Image: RGB formatında görsel
    """
    if gorsel_yolu.endswith(".npy"):
        dizi = np.load(gorsel_yolu)  # (512, 512, 3) uint8
        return Image.fromarray(dizi)
    else:
        return Image.open(gorsel_yolu).convert("RGB")


def run_inference(model, processor, gorsel_yolu, cikti_yolu=None):
    """
    Tek bir görsel üzerinde hasar segmentation tahmini yapar.

    Args:
        model:       Eğitilmiş model (LoRA adapter yüklenmiş)
        processor:   SAM3 Processor
        gorsel_yolu: Tahmin yapılacak görselin yolu (.jpg veya .npy)
        cikti_yolu:  Sonucun kaydedileceği klasör (None ise kaydetmez)

    Returns:
        tuple: (orijinal_gorsel PIL, tahmin_mask numpy 0/1)
    """
    log(f"Gorsel okunuyor: {gorsel_yolu}")
    gorsel = gorsel_oku(gorsel_yolu)
    orijinal_genislik, orijinal_yukseklik = gorsel.size

    # Metin ipucunu tokenize et
    text_inputs = processor.tokenizer(
        Config.TEXT_PROMPT,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = text_inputs["input_ids"].to(Config.DEVICE)
    attention_mask = text_inputs["attention_mask"].to(Config.DEVICE)

    # Görseli processor ile işle
    image_inputs = processor.image_processor(images=gorsel, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(Config.DEVICE)

    # Tahmin yap
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(Config.DEVICE == "cuda")):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

    # semantic_seg: (1, 1, 288, 288) -> (288, 288) olasılık
    logits = outputs.semantic_seg.squeeze()  # (288, 288)
    olasilik = logits.sigmoid().cpu().numpy()

    # Eşik uygula: 0.5 üstü = hasar var
    binary_mask = (olasilik > 0.5).astype(np.float32)

    # Maskleri orijinal görsel boyutuna büyüt
    mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
    mask_buyuk = mask_pil.resize(
        (orijinal_genislik, orijinal_yukseklik), Image.NEAREST
    )
    tahmin_mask = np.array(mask_buyuk, dtype=np.float32) / 255.0

    # Sonucu görselleştir ve kaydet
    if cikti_yolu is not None:
        ensure_dir(cikti_yolu)
        gorsel_adi = os.path.splitext(os.path.basename(gorsel_yolu))[0]
        kayit_yolu = os.path.join(cikti_yolu, f"{gorsel_adi}_tahmin.png")
        gorselleştir(gorsel, tahmin_mask, kayit_yolu)

    log("Tahmin tamamlandi.")
    return gorsel, tahmin_mask


def gorselleştir(gorsel, tahmin_mask, kayit_yolu):
    """
    Tahmin sonucunu 3 panel olarak görselleştirir ve kaydeder.

    Paneller:
    1. Orijinal gorsel
    2. Tahmin maskleri (beyaz = hasar)
    3. Overlay (kirmizi = hasar bolgesi)

    Args:
        gorsel:      Orijinal gorsel (PIL.Image)
        tahmin_mask: Tahmin maskleri (numpy array, 0/1)
        kayit_yolu:  Sonucun kaydedilecegi dosya yolu
    """
    gorsel_np = np.array(gorsel) / 255.0  # (H, W, 3) — 0-1 arasi

    # Overlay: hasar bolgelerini kirmiziye boya
    overlay = gorsel_np.copy()
    overlay[tahmin_mask == 1, 0] = 1.0   # Kirmizi kanal
    overlay[tahmin_mask == 1, 1] *= 0.3  # Yesili azalt
    overlay[tahmin_mask == 1, 2] *= 0.3  # Maviyi azalt

    # Hasar yuzdesi hesapla
    hasar_yuzdesi = tahmin_mask.mean() * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(gorsel_np)
    axes[0].set_title("Orijinal Gorsel", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(tahmin_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Hasar Maskleri (Beyaz = Hasar)", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay — Hasar: %{hasar_yuzdesi:.1f}", fontsize=13)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(kayit_yolu, dpi=150, bbox_inches="tight")
    plt.close()

    log(f"Gorsellestirme kaydedildi: {kayit_yolu}")
    log(f"Hasar yuzdesi: %{hasar_yuzdesi:.1f}")


def run_multiclass_inference(model, processor, gorsel_yolu, cikti_yolu=None):
    """
    19 hasar sınıfının her biri için ayrı ayrı tahmin yapar.

    Her sınıf adı ("Crack", "Rust" vb.) ayrı metin ipucu olarak
    modele verilir. Sonuç renkli overlay olarak kaydedilir.

    Args:
        model:       Eğitilmiş model
        processor:   SAM3 Processor
        gorsel_yolu: Görsel yolu (.jpg veya .npy)
        cikti_yolu:  Sonuçların kaydedileceği klasör

    Returns:
        dict: {sinif_adi: mask_array} sözlüğü
    """
    log(f"Multi-class inference: {gorsel_yolu}")
    gorsel = gorsel_oku(gorsel_yolu)
    gorsel_np = np.array(gorsel) / 255.0

    # Görseli bir kez işle — tüm sınıflar için aynı pixel_values kullanılır
    image_inputs = processor.image_processor(images=gorsel, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(Config.DEVICE)

    # Her sınıf için tahmin yap
    sinif_maskler = {}
    tespit_edilenler = []

    model.eval()
    with torch.no_grad():
        for sinif_adi in DACL10K_CLASSES:
            # Bu sınıf için metin ipucu oluştur
            text_inputs = processor.tokenizer(
                sinif_adi,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = text_inputs["input_ids"].to(Config.DEVICE)
            attention_mask = text_inputs["attention_mask"].to(Config.DEVICE)

            with torch.amp.autocast("cuda", enabled=(Config.DEVICE == "cuda")):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            # Maskleri orijinal görsel boyutuna büyüt
            logits = outputs.semantic_seg.squeeze()
            olasilik = logits.sigmoid().cpu().numpy()
            binary_mask = (olasilik > 0.5).astype(np.float32)

            mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
            mask_buyuk = mask_pil.resize(gorsel.size, Image.NEAREST)
            mask_array = np.array(mask_buyuk, dtype=np.float32) / 255.0

            sinif_maskler[sinif_adi] = mask_array

            # Kayda değer hasar tespiti var mı? (>%1 piksel)
            if mask_array.mean() > 0.01:
                tespit_edilenler.append(sinif_adi)

    log(f"Tespit edilen hasar turleri: {tespit_edilenler if tespit_edilenler else 'Yok'}")

    # Renkli overlay oluştur
    if cikti_yolu is not None:
        ensure_dir(cikti_yolu)
        gorsel_adi = os.path.splitext(os.path.basename(gorsel_yolu))[0]
        kayit_yolu = os.path.join(cikti_yolu, f"{gorsel_adi}_multiclass.png")
        _gorselleştir_multiclass(gorsel_np, sinif_maskler, tespit_edilenler, kayit_yolu)

    return sinif_maskler


def _gorselleştir_multiclass(gorsel_np, sinif_maskler, tespit_edilenler, kayit_yolu):
    """
    Multi-class sonuçları renkli overlay olarak görselleştirir.

    Her hasar sınıfı farklı renkle gösterilir.
    Sadece tespit edilen sınıflar gösterilir.
    """
    # Renkli overlay oluştur
    overlay = gorsel_np.copy()

    for sinif_adi in tespit_edilenler:
        sinif_idx = DACL10K_CLASSES.index(sinif_adi)
        renk = [c / 255.0 for c in SINIF_RENKLERI[sinif_idx]]
        mask = sinif_maskler[sinif_adi]

        # Yarı saydam renk uygula
        overlay[mask == 1, 0] = overlay[mask == 1, 0] * 0.4 + renk[0] * 0.6
        overlay[mask == 1, 1] = overlay[mask == 1, 1] * 0.4 + renk[1] * 0.6
        overlay[mask == 1, 2] = overlay[mask == 1, 2] * 0.4 + renk[2] * 0.6

    # Görseli ve legend'ı çiz
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].imshow(gorsel_np)
    axes[0].set_title("Orijinal Gorsel", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Tespit Edilen Hasar Turleri", fontsize=13)
    axes[1].axis("off")

    # Legend ekle — sadece tespit edilen sınıflar
    if tespit_edilenler:
        from matplotlib.patches import Patch
        legend_elemanlar = []
        for sinif_adi in tespit_edilenler:
            sinif_idx = DACL10K_CLASSES.index(sinif_adi)
            renk = [c / 255.0 for c in SINIF_RENKLERI[sinif_idx]]
            yuzde = sinif_maskler[sinif_adi].mean() * 100
            legend_elemanlar.append(
                Patch(facecolor=renk, label=f"{sinif_adi} (%{yuzde:.1f})")
            )
        axes[1].legend(
            handles=legend_elemanlar,
            loc="lower right",
            fontsize=9,
            framealpha=0.8,
        )

    plt.tight_layout()
    plt.savefig(kayit_yolu, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Multi-class gorsellestirme kaydedildi: {kayit_yolu}")
