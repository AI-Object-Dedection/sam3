# Capstone SAM3 — Köprü Hasarı Segmentasyonu

Bu proje, **SAM3** (Segment Anything Model 3) modelini **DACL10K** veri seti üzerinde **LoRA** ile fine-tune ederek köprü hasarlarını otomatik olarak segmente etmeyi amaçlar.

## Proje Özeti

| Bileşen | Değer |
|---------|-------|
| **Model** | facebook/sam3 |
| **Dataset** | DACL10K (9,920 görsel, 19 sınıf) |
| **Fine-tuning** | LoRA (PEFT kütüphanesi ile) |
| **Çıktı** | Segmentation mask (hasar bölgeleri) |

## Kurulum

```bash
# 1. Virtual environment oluştur
python -m venv venv
source venv/bin/activate  # macOS/Linux

# 2. Bağımlılıkları kur
pip install -r requirements.txt
```

## Klasör Yapısı

```
capstone-sam3/
├── data/                  # DACL10K veri seti
├── src/                   # Ana Python kodları
│   ├── config.py          # Merkezi ayarlar
│   ├── dataset.py         # Veri okuma ve hazırlama
│   ├── model.py           # SAM3 model yükleme
│   ├── lora.py            # LoRA uygulama
│   ├── train.py           # Eğitim döngüsü
│   ├── evaluate.py        # Model değerlendirme
│   ├── infer.py           # Tek görsel tahmini
│   └── utils.py           # Yardımcı fonksiyonlar
├── scripts/               # Çalıştırma dosyaları
│   ├── run_train.py       # Eğitimi başlat
│   ├── run_eval.py        # Değerlendirmeyi başlat
│   ├── run_infer.py       # Tekli tahmin yap
│   └── inspect_dataset.py # Veri setini incele
├── checkpoints/           # Kaydedilen model ağırlıkları
├── outputs/               # Model çıktıları ve görseller
├── requirements.txt       # Python bağımlılıkları
└── README.md              # Bu dosya
```

## Kullanım

```bash
# Veri setini incele
python scripts/inspect_dataset.py

# Eğitimi başlat
python scripts/run_train.py

# Değerlendirme yap
python scripts/run_eval.py

# Tek görsel üzerinde tahmin yap
python scripts/run_infer.py
```

## Ayarlar

Tüm proje ayarları `src/config.py` dosyasında bulunur:

- **Model adı**: `facebook/sam3`
- **Batch size**: 2
- **Learning rate**: 1e-4
- **Epoch sayısı**: 10
- **LoRA rank**: 8

## Ekip

3 kişilik bir ekip tarafından capstone projesi olarak geliştirilmektedir.
