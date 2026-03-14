# Proje Özeti

## Bu Proje Ne Yapıyor?

Köprülerdeki hasarları (çatlak, pas, dökülme vb.) fotoğraflardan otomatik olarak tespit eden bir yapay zeka modeli geliştiriyoruz.

### Basit Anlatım

```
Girdi:    Bir köprü fotoğrafı
İşlem:    Model fotoğrafı analiz eder
Çıktı:    Hasarlı bölgeleri gösteren bir maske (segmentation mask)
```

Örnek düşünün: Bir röntgen filminde doktorun hastalıklı bölgeyi işaretlemesi gibi, bizim modelimiz de köprü fotoğrafında hasarlı bölgeleri işaretliyor.

## 3 Ana Bileşen

### 1. Model — SAM3 (`facebook/sam3`)

**Ne:** Meta'nın geliştirdiği "Segment Anything Model 3" — görsellerdeki nesneleri otomatik parçalara ayırabilen büyük bir model.

**Neden bunu seçtik:** SAM3 zaten görselleri çok iyi anlıyor. Biz onu sıfırdan eğitmek yerine, köprü hasarlarını da tanıması için "ince ayar" (fine-tuning) yapacağız.

**Benzetme:** SAM3 = çok dil bilen bir tercüman. Biz ona Türkçe de öğretiyoruz (köprü hasarları = yeni bir "dil").

### 2. Dataset — DACL10K

**Ne:** ~10.000 köprü fotoğrafı ve her fotoğraftaki hasarların el ile işaretlenmiş hali.

**İçerik:**
- 6.935 eğitim görseli
- 975 doğrulama görseli
- 19 farklı hasar/bileşen sınıfı
- Her görsel için ayrı bir `.json` annotation dosyası

**Benzetme:** DACL10K = ders kitabı. Model bu kitaptan hasarları tanımayı öğrenecek.

### 3. Fine-tuning Yöntemi — LoRA

**Ne:** Modelin tamamını eğitmek yerine, sadece küçük "ek katmanlar" ekleyip onları eğitmek.

**Neden:** SAM3'ün milyonlarca parametresi var. Hepsini eğitmek:
- Çok fazla GPU belleği gerektirir
- Çok uzun sürer
- Orijinal modeli bozma riski taşır

LoRA ile sadece ~%1-2'sini eğitiyoruz.

**Benzetme:** LoRA = gözlük takmak. Gözünüzü (modeli) değiştirmiyorsunuz, sadece üzerine bir lens ekliyorsunuz.

## Veri Akışı

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  DACL10K    │     │   SAM3 +     │     │  Segmentation│
│  Görseller  │────>│   LoRA       │────>│  Mask        │
│  + Maskeler │     │   (Eğitim)   │     │  (Tahmin)    │
└─────────────┘     └──────────────┘     └─────────────┘
    Girdi              Model                Çıktı
```

## Proje Klasör Yapısı

```
SAM3/
├── CLAUDE.md              # AI agent talimatları
├── README.md              # Proje tanıtımı
├── requirements.txt       # Python paketleri
│
├── docs/                  # Dokümantasyon
│   ├── 01_PROJE_OZETI.md      # Bu dosya
│   ├── 02_TEKNIK_REHBER.md    # SAM3, LoRA, DACL10K detayları
│   ├── 03_GELISTIRME_ADIMLARI.md  # Adım adım geliştirme planı
│   └── 04_VIBE_CODING_REHBERI.md  # AI ile çalışma rehberi
│
├── src/                   # Ana kod (logic burada)
│   ├── config.py          # Tüm ayarlar — projenin kontrol paneli
│   ├── dataset.py         # DACL10K verisini okur ve hazırlar
│   ├── model.py           # SAM3 modelini yükler
│   ├── lora.py            # LoRA'yı modele uygular
│   ├── train.py           # Eğitim döngüsü
│   ├── evaluate.py        # Model performansını ölçer
│   ├── infer.py           # Tek fotoğraf üzerinde tahmin yapar
│   └── utils.py           # Yardımcı küçük fonksiyonlar
│
├── scripts/               # Çalıştırma dosyaları (src/'yi çağırır)
│   ├── run_train.py       # → python scripts/run_train.py
│   ├── run_eval.py        # → python scripts/run_eval.py
│   ├── run_infer.py       # → python scripts/run_infer.py
│   └── inspect_dataset.py # → python scripts/inspect_dataset.py
│
├── data/                  # Veri seti (git'e dahil değil)
├── checkpoints/           # Kaydedilen model ağırlıkları (git'e dahil değil)
└── outputs/               # Çıktılar (git'e dahil değil)
```

### Dosyalar Arası İlişki

```
scripts/run_train.py
    ├── src/config.py      (ayarları okur)
    ├── src/model.py       (modeli yükler)
    ├── src/lora.py        (LoRA uygular)
    ├── src/dataset.py     (veriyi hazırlar)
    └── src/train.py       (eğitimi çalıştırır)
         └── src/evaluate.py (her epoch sonunda değerlendirir)
```

## Ekip ve Bağlam

- 3 kişilik ekip
- ML'de yeni — ilk fine-tuning projesi
- Amaç: Öğrenmek + çalışan bir proje üretmek
- Vibe coding yaklaşımı ile AI destekli geliştirme
