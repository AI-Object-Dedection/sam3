# CLAUDE.md — AI Agent Proje Talimatları

Bu dosya, bu projede çalışan tüm AI agent'lar için temel kılavuzdur.
Bu dosyayı her zaman oku ve buradaki kurallara uy.

## Proje Özeti

Bu bir **capstone (bitirme) projesidir**. SAM3 modelini DACL10K köprü hasarı veri seti üzerinde LoRA ile fine-tune ederek köprü hasarlarını otomatik segmente eden bir model üretmeyi amaçlıyoruz.

| Bileşen | Değer |
|---------|-------|
| Model | `facebook/sam3` (HuggingFace) |
| Dataset | DACL10K (~10.000 görsel, 19 sınıf, numpy .npy formatı) |
| Fine-tuning | LoRA (PEFT kütüphanesi) |
| Çıktı | Segmentation mask (hasar bölgeleri) |

## Ekip Profili — ÇOK ÖNEMLİ

- 3 kişilik ekip, **hiçbiri ML konusunda deneyimli değil**
- Bu ekibin **ilk gerçek fine-tuning projesi**
- Teknik terimler kullanırsan **mutlaka basitçe açıkla**
- Karmaşık çözümler önerme, **basit ve anlaşılır olanı tercih et**
- Türkçe iletişim kur

## Davranış Kuralları

### Yapılması Gerekenler

1. **Koddan önce açıklama yap** — Her dosya veya fonksiyon için önce ne yaptığını, neden var olduğunu kısaca açıkla
2. **Küçük adımlarla ilerle** — Tek seferde büyük değişiklikler yapma
3. **Basit kod yaz** — Okunabilirlik her zaman performanstan önce gelir
4. **Mevcut yapıyı koru** — Proje yapısını bozmadan çalış (aşağıdaki dosya sorumluluklarına bak)
5. **Test et** — Değişiklik yaptığında çalıştığını doğrula
6. **Dokümantasyonu güncelle** — Yeni bir şey eklersen `docs/` klasöründeki ilgili dosyayı güncelle

### Yapılmaması Gerekenler

1. **Gereksiz karmaşıklık oluşturma** — Basit bir for döngüsü yetiyorsa fonksiyonel zincirleme kullanma
2. **Çok ileri ML bilgisi varsayma** — Ekip yeni, terminolojiyi açıkla
3. **Büyük mimari değişiklikler yapma** — Önce açıklama yap, onay al
4. **Framework ekleme** — requirements.txt dışında yeni bağımlılık ekleme
5. **Tek mesajda her şeyi yazma** — Parça parça ilerle
6. **İngilizce yorum yazma** — Kod yorumları ve docstring'ler Türkçe olmalı

## Proje Yapısı ve Dosya Sorumlulukları

```
SAM3/
├── CLAUDE.md              ← AI agent talimatları (BU DOSYA)
├── README.md              ← Proje açıklaması
├── requirements.txt       ← Python bağımlılıkları
├── docs/                  ← Detaylı dokümantasyon
│   ├── 01_PROJE_OZETI.md
│   ├── 02_TEKNIK_REHBER.md
│   ├── 03_GELISTIRME_ADIMLARI.md
│   └── 04_VIBE_CODING_REHBERI.md
├── data/                  ← DACL10K veri seti (git'e dahil değil)
│   └── dacl10k/
│       ├── annotations/{train,validation}/*.json
│       └── images/{train,validation}/*.jpg
├── src/                   ← Ana Python kodları
│   ├── config.py          ← Merkezi ayarlar (tek kontrol paneli)
│   ├── dataset.py         ← Veri okuma ve mask oluşturma
│   ├── model.py           ← SAM3 model yükleme
│   ├── lora.py            ← LoRA uygulama
│   ├── train.py           ← Eğitim döngüsü
│   ├── evaluate.py        ← Model değerlendirme (IoU)
│   ├── infer.py           ← Tek görsel tahmini
│   └── utils.py           ← Yardımcı fonksiyonlar
├── scripts/               ← Çalıştırma dosyaları
│   ├── run_train.py       ← Eğitimi başlat
│   ├── run_eval.py        ← Değerlendirme başlat
│   ├── run_infer.py       ← Tekli tahmin yap
│   └── inspect_dataset.py ← Veri setini incele
├── checkpoints/           ← Model ağırlıkları (git'e dahil değil)
└── outputs/               ← Çıktılar (git'e dahil değil)
```

### Dosya Kuralları

- **Her dosyanın tek bir görevi vardır** — dataset.py sadece veri okur, model.py sadece model yükler
- **config.py tek kontrol noktasıdır** — Tüm ayarlar buradan okunur, başka yere sabit değer yazma
- **scripts/ dosyaları sadece çağırır** — Logic yazmaz, src/'deki fonksiyonları çağırır
- **utils.py küçük kalmalıdır** — Sadece birden fazla yerde kullanılan basit fonksiyonlar

## Kod Yazım Kuralları

```python
# ✅ DOĞRU: Basit, okunabilir, Türkçe yorum
def gorsel_yukle(dosya_yolu):
    """Görseli okur ve RGB formatına çevirir."""
    gorsel = Image.open(dosya_yolu).convert("RGB")
    return gorsel

# ❌ YANLIŞ: Karmaşık, İngilizce, gereksiz soyutlama
def load_and_preprocess_image(path, transforms=None, cache=True):
    """Load image with optional transforms and caching."""
    img = ImageLoader.get_instance().load(path, mode="RGB")
    if transforms:
        img = compose_transforms(transforms)(img)
    return img
```

### Adlandırma Kuralları

- Fonksiyon ve değişken adları: `snake_case` (Türkçe veya İngilizce kabul)
- Sınıf adları: `PascalCase`
- Sabitler: `UPPER_CASE`
- Dosya adları: `snake_case.py`

## Geliştirme Fazları

Proje 5 fazda ilerliyor. **Sırayı atlamayın.**
Detaylar için → `docs/03_GELISTIRME_ADIMLARI.md`

| Faz | İsim | Durum |
|-----|------|-------|
| 1 | Proje İskeleti | ✅ Tamamlandı |
| 2 | Veri Seti Entegrasyonu | ✅ Tamamlandı |
| 3 | Model + LoRA Kurulumu | ✅ Tamamlandı |
| 4 | Eğitim + Değerlendirme | ⬜ Bekliyor |
| 5 | Inference + Demo | ⬜ Bekliyor |

## Teknik Referanslar

Detaylı açıklamalar için → `docs/02_TEKNIK_REHBER.md`

### DACL10K Hızlı Bilgi

- Format: Supervisely JSON + JPG (datasetninja.com versiyonu)
- Görseller: `.jpg` (değişken boyut, RGB) | Annotationlar: `.json` (polygon şekilleri)
- Mask'lar runtime'da `_json_annotation_to_mask()` ile polygon'lardan oluşturulur
- 19 sınıf (13 hasar + 6 bileşen), her kanal bir sınıf
- Sınıf adları kısaltmalı: `ACrack`, `EJoint`, `PEquipment`, `WConccor` vb.
- Kurulum: `pip install dataset-tools` → `python scripts/setup_dataset.py`

### SAM3 Hızlı Bilgi

- Model: `facebook/sam3`
- Sınıflar: `Sam3Model`, `Sam3Processor` (transformers kütüphanesi)
- Görev: Verilen bir görselde nesneleri segmente etmek

### LoRA Hızlı Bilgi

- Kütüphane: `peft`
- Amaç: Modelin küçük bir kısmını eğiterek bellek ve zaman kazanmak
- Hedef katmanlar: `q_proj`, `v_proj`
- Ayarlar `config.py`'de: `LORA_RANK=8`, `LORA_ALPHA=16`, `LORA_DROPOUT=0.1`

## Hata Durumunda

Bir şey çalışmıyorsa şu sırayla kontrol et:

1. `config.py`'deki yollar doğru mu?
2. Veri seti `data/dacl10k/` altında mı?
3. `requirements.txt`'teki paketler kurulu mu?
4. GPU kullanılıyorsa CUDA sürümü uyumlu mu?
5. Model indirilebiliyor mu? (internet bağlantısı)

## Commit Mesaj Formatı

```
[faz-no] kısa açıklama

Örnek:
[faz-1] proje iskeleti oluşturuldu
[faz-2] dataset.py DACL10K formatına uyumlandı
[faz-3] LoRA konfigürasyonu eklendi
```
