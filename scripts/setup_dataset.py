"""
setup_dataset.py — DACL10K Veri Seti Kurulum Scripti

Bu script datasetninja.com'dan DACL10K veri setini indirip
projenin beklediği klasör yapısına yerleştirir.

Kullanım (2 adım):

  Adım 1 — dataset-tools paketini kur (sadece bir kez):
      pip install dataset-tools

  Adım 2 — Bu scripti çalıştır:
      python scripts/setup_dataset.py

Sonuç olarak şu yapı oluşur:
    data/dacl10k/
    ├── images/
    │   ├── train/       ← .jpg görseller (6.935 adet)
    │   └── validation/  ← .jpg görseller (975 adet)
    └── annotations/
        ├── train/       ← .json annotation dosyaları (6.935 adet)
        └── validation/  ← .json annotation dosyaları (975 adet)
"""

import json
import os
import shutil
import sys


# Hedef klasörler (config.py ile aynı)
HEDEF_DIZIN = "data/dacl10k"
HEDEF_GORSELLER = {
    "train": "data/dacl10k/images/train",
    "validation": "data/dacl10k/images/validation",
}
HEDEF_ANNOTATIONLAR = {
    "train": "data/dacl10k/annotations/train",
    "validation": "data/dacl10k/annotations/validation",
}

# Datasetninja'nın indirdiği geçici klasör
INDIRME_DIZIN = "data/dacl10k_raw"


def klasorleri_olustur():
    """Gerekli tüm klasörleri oluşturur."""
    for yol in list(HEDEF_GORSELLER.values()) + list(HEDEF_ANNOTATIONLAR.values()):
        os.makedirs(yol, exist_ok=True)
    print("[kurulum] Klasörler oluşturuldu.")


def datasetninja_indir():
    """
    dataset-tools paketi ile DACL10K'yi datasetninja.com'dan indirir.

    Bu fonksiyon yaklaşık 4-5 GB veri indirir, internet hızına göre
    10-30 dakika sürebilir.
    """
    try:
        import dataset_tools as dtools
    except ImportError:
        print("\nHATA: 'dataset-tools' paketi bulunamadı.")
        print("Kurmak için: pip install dataset-tools")
        print("Kurulduktan sonra bu scripti tekrar çalıştırın.\n")
        sys.exit(1)

    print("[kurulum] DACL10K indiriliyor (datasetninja.com)...")
    print("[kurulum] Bu işlem ~4-5 GB veri indirir ve 10-30 dakika sürebilir.")
    print("[kurulum] Lütfen bekleyin...\n")

    os.makedirs(INDIRME_DIZIN, exist_ok=True)
    dtools.download(dataset="DACL10K", dst_dir=INDIRME_DIZIN)
    print("[kurulum] İndirme tamamlandı.")


def split_bul(ham_dizin):
    """
    İndirilen ham klasörde train ve validation split'lerini bulur.

    Datasetninja farklı isimlendirmeler kullanabilir:
    - train / validation
    - train / val
    - ds0 / ds1 (split ismi yoksa)

    Returns:
        dict: {"train": "yol/...", "validation": "yol/..."} veya None
    """
    # Olası split adları
    train_adaylari = ["train"]
    val_adaylari = ["validation", "val"]

    train_yolu = None
    val_yolu = None

    for klasor in os.listdir(ham_dizin):
        tam_yol = os.path.join(ham_dizin, klasor)
        if not os.path.isdir(tam_yol):
            continue
        if klasor.lower() in train_adaylari:
            train_yolu = tam_yol
        elif klasor.lower() in val_adaylari:
            val_yolu = tam_yol

    if train_yolu and val_yolu:
        return {"train": train_yolu, "validation": val_yolu}

    # Split bulunamazsa, tek bir "ds" klasörü olabilir
    print("[UYARI] train/validation klasörleri doğrudan bulunamadı.")
    print("[UYARI] Datasetninja klasör yapısı beklenden farklı olabilir.")
    print(f"[UYARI] Ham klasör içeriği: {os.listdir(ham_dizin)}")
    return None


def dosyalari_tasI(split_yolu, hedef_gorsel_dizin, hedef_annotation_dizin):
    """
    Tek bir split (train veya validation) için dosyaları taşır.

    Datasetninja yapısı:
        split/
        ├── img/
        │   └── resim.jpg
        └── ann/
            └── resim.jpg.json    ← "resim.jpg" + ".json" şeklinde isimlendirilmiş

    Hedef yapı:
        images/split/
        └── resim.jpg
        annotations/split/
        └── resim.json            ← ".jpg" kısmı kaldırıldı

    Args:
        split_yolu:             Datasetninja split klasörü (ör: data/dacl10k_raw/train)
        hedef_gorsel_dizin:     Görsellerin gideceği yer
        hedef_annotation_dizin: Annotationların gideceği yer
    """
    gorsel_kaynak = os.path.join(split_yolu, "img")
    annotation_kaynak = os.path.join(split_yolu, "ann")

    if not os.path.exists(gorsel_kaynak):
        print(f"  HATA: Görsel klasörü bulunamadı: {gorsel_kaynak}")
        return 0, 0

    if not os.path.exists(annotation_kaynak):
        print(f"  HATA: Annotation klasörü bulunamadı: {annotation_kaynak}")
        return 0, 0

    gorsel_sayisi = 0
    annotation_sayisi = 0

    # Görselleri taşı
    for dosya in os.listdir(gorsel_kaynak):
        if dosya.lower().endswith(".jpg") or dosya.lower().endswith(".jpeg"):
            shutil.copy2(
                os.path.join(gorsel_kaynak, dosya),
                os.path.join(hedef_gorsel_dizin, dosya),
            )
            gorsel_sayisi += 1

    # Annotationları taşı — "resim.jpg.json" → "resim.json" şeklinde yeniden adlandır
    for dosya in os.listdir(annotation_kaynak):
        if dosya.endswith(".json"):
            # "resim.jpg.json" → "resim.json"
            if dosya.endswith(".jpg.json"):
                yeni_ad = dosya[:-9] + ".json"  # ".jpg.json" (9 karakter) kaldır
            elif dosya.endswith(".jpeg.json"):
                yeni_ad = dosya[:-10] + ".json"
            else:
                yeni_ad = dosya

            shutil.copy2(
                os.path.join(annotation_kaynak, dosya),
                os.path.join(hedef_annotation_dizin, yeni_ad),
            )
            annotation_sayisi += 1

    return gorsel_sayisi, annotation_sayisi


def kurulumu_dogrula():
    """
    Kurulumun başarılı olduğunu kontrol eder.
    Görsel ve annotation sayılarını yazdırır.
    """
    print("\n[doğrulama] Kurulum kontrol ediliyor...")

    for split in ["train", "validation"]:
        gorsel_dizin = HEDEF_GORSELLER[split]
        ann_dizin = HEDEF_ANNOTATIONLAR[split]

        gorsel_sayisi = len([
            f for f in os.listdir(gorsel_dizin)
            if f.lower().endswith(".jpg")
        ]) if os.path.exists(gorsel_dizin) else 0

        ann_sayisi = len([
            f for f in os.listdir(ann_dizin)
            if f.endswith(".json")
        ]) if os.path.exists(ann_dizin) else 0

        durum = "✓" if gorsel_sayisi > 0 else "✗"
        print(f"  {durum} {split}: {gorsel_sayisi} görsel, {ann_sayisi} annotation")

    print("\nKurulum tamamlandı!")
    print("Test için: python scripts/inspect_dataset.py")


def main():
    print("=" * 55)
    print("DACL10K Veri Seti Kurulum Scripti")
    print("Kaynak: datasetninja.com/dacl10k")
    print("=" * 55)

    # Klasörleri oluştur
    klasorleri_olustur()

    # Daha önce indirilmişse tekrar indirme
    if os.path.exists(INDIRME_DIZIN) and os.listdir(INDIRME_DIZIN):
        print(f"[kurulum] Ham veri zaten mevcut: {INDIRME_DIZIN}")
        print("[kurulum] Yeniden indirme atlanıyor.")
    else:
        datasetninja_indir()

    # Datasetninja klasör yapısını bul
    print("\n[kurulum] Klasör yapısı analiz ediliyor...")

    # Datasetninja genellikle DACL10K/ alt klasörü oluşturur
    icindekiler = os.listdir(INDIRME_DIZIN) if os.path.exists(INDIRME_DIZIN) else []
    ham_kok = INDIRME_DIZIN

    # İçeride tek bir klasör varsa (ör: "DACL10K/") ona in
    if len(icindekiler) == 1 and os.path.isdir(os.path.join(INDIRME_DIZIN, icindekiler[0])):
        ham_kok = os.path.join(INDIRME_DIZIN, icindekiler[0])
        print(f"[kurulum] Alt klasör bulundu: {icindekiler[0]}/")

    splitler = split_bul(ham_kok)
    if not splitler:
        print("\nHATA: Veri yapısı tanınamadı.")
        print("Lütfen klasör yapısını manuel olarak kontrol edin:")
        print(f"  {ham_kok}/")
        sys.exit(1)

    # Dosyaları taşı
    for split_adi, split_yolu in splitler.items():
        print(f"\n[kurulum] {split_adi} verisi kopyalanıyor...")
        g, a = dosyalari_tasI(
            split_yolu,
            HEDEF_GORSELLER[split_adi],
            HEDEF_ANNOTATIONLAR[split_adi],
        )
        print(f"  → {g} görsel, {a} annotation kopyalandı")

    # Kurulumu doğrula
    kurulumu_dogrula()

    # Ham veriyi silmek isteyip istemediğini sor
    print(f"\nGeçici indirme klasörü: {INDIRME_DIZIN} (~4-5 GB)")
    cevap = input("Bu klasörü silmek ister misiniz? (e/h): ").strip().lower()
    if cevap == "e":
        shutil.rmtree(INDIRME_DIZIN)
        print(f"[kurulum] {INDIRME_DIZIN} silindi.")


if __name__ == "__main__":
    main()
