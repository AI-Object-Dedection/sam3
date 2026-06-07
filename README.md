# SAM3 Bridge Damage Segmentation

Bu proje, `facebook/sam3` modelini DACL10K köprü hasarı veri seti üzerinde LoRA ile fine-tune ederek hasarlı bölgeler için segmentation mask üretmeyi amaçlar.

Proje bir capstone çalışmasıdır. Mevcut odak, modeli çalışır şekilde eğitmek, checkpoint almak, inference çıktısı üretmek ve bu çıktıyı web tabanlı prototipe bağlamaktır.

## Güncel Durum

| Başlık | Durum |
|---|---|
| Proje iskeleti | Tamamlandı |
| DACL10K veri okuyucu | `.npy` ve DatasetNinja `.jpg + .json` formatlarını destekliyor |
| SAM3 yükleme | Kaggle T4 için `fp16` ve `low_cpu_mem_usage=True` ile çalışıyor |
| LoRA entegrasyonu | Tamamlandı |
| Eğitim | Kaggle üzerinde çalışıyor, 1 full epoch hedefleniyor |
| Ara checkpoint | Her 500 batch'te LoRA checkpoint kaydediliyor |
| Web entegrasyonu | Sıradaki iş: eğitilmiş checkpoint'i backend inference hattına bağlamak |
| Rapor | Gerçek eğitim metrikleri ve demo ekran görüntüleri ile güncellenecek |

## Teknik Özet

| Bileşen | Değer |
|---|---|
| Model | `facebook/sam3` |
| Dataset | DACL10K bridge damage dataset |
| Veri formatı | DatasetNinja: `.jpg` görsel + `.json` polygon annotation |
| Sınıf sayısı | 19 |
| Fine-tuning | LoRA, PEFT |
| Loss | `bce_dice` |
| Scheduler | `cosine_warmup` |
| Varsayılan prompt | `damage` |
| Çıktı | 288x288 binary damage mask |

## Gerçekçi Eğitim Planı

Kaggle T4 üzerinde ölçülen hız yaklaşık `3.9 sn / batch` seviyesindedir.

| Koşu | Tahmini süre | Kullanım amacı |
|---|---:|---|
| 1000 train + 200 val, 2 epoch | 2.5-3 saat | Hızlı test |
| Full train + full val, 1 epoch | 8.5-9 saat | Teslim için minimum model |
| Full train + full val, 2 epoch | 17-18 saat | Daha iyi checkpoint adayı |
| 10 epoch | 85-90 saat | Kaggle T4 için pratik değil |

Teslim için önerilen hedef: önce `1 full epoch` tamamlamak, checkpoint almak, örnek inference üretmek ve web sitesine bağlamak. Zaman kalırsa 2. epoch denenebilir.

## Klasör Yapısı

```text
SAM3/
├── src/
│   ├── config.py       # Merkezi ayarlar ve env override desteği
│   ├── dataset.py      # DACL10K görsel/json okuma ve mask üretimi
│   ├── model.py        # SAM3 model ve processor yükleme
│   ├── lora.py         # LoRA uygulama
│   ├── train.py        # Eğitim döngüsü ve ara checkpoint
│   ├── evaluate.py     # IoU ve validation
│   ├── infer.py        # Tek görsel inference yardımcıları
│   └── utils.py
├── scripts/
│   ├── run_train.py
│   ├── run_train_ddp.py
│   ├── run_eval.py
│   ├── run_infer.py
│   └── inspect_dataset.py
├── docs/
├── data/               # Git'e dahil değil
├── checkpoints/        # Git'e dahil değil
└── outputs/            # Git'e dahil değil
```

Beklenen dataset yapısı:

```text
data/dacl10k/
├── images/
│   ├── train/*.jpg
│   └── validation/*.jpg
└── annotations/
    ├── train/*.json
    └── validation/*.json
```

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS için:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset Kontrol

```bash
python scripts/inspect_dataset.py
```

Bu komut train/validation görsel ve annotation sayılarını gösterir. Dataset yoksa önce DACL10K dosyaları `data/dacl10k/` altına yerleştirilmelidir.

## Eğitim

Standart eğitim:

```bash
python scripts/run_train.py
```

Kaggle veya terminalde daha kontrollü eğitim için env değişkenleri kullanılabilir:

```bash
set NUM_EPOCHS=1
set CHECKPOINT_EVERY_STEPS=500
python scripts/run_train.py
```

Linux/Kaggle:

```bash
NUM_EPOCHS=1 CHECKPOINT_EVERY_STEPS=500 python scripts/run_train.py
```

Hızlı test koşusu:

```bash
NUM_EPOCHS=2 MAX_TRAIN_SAMPLES=1000 MAX_VAL_SAMPLES=200 python scripts/run_train.py
```

## Kaggle Notları

Kaggle T4 üzerinde SAM3 modelini doğrudan `facebook/sam3` adıyla yüklemek bazen yavaş veya kararsız olabilir. Önerilen yöntem:

1. Hugging Face token ile modeli `/kaggle/working/hf_cache` altına indirmek.
2. `SAM3_MODEL_NAME` değişkenini local snapshot yoluna ayarlamak.
3. Eğitimi `NUM_EPOCHS=1` ve `CHECKPOINT_EVERY_STEPS=500` ile başlatmak.

Örnek:

```python
env["SAM3_MODEL_NAME"] = "/kaggle/working/hf_cache/models--facebook--sam3/snapshots/<snapshot_id>"
env["NUM_EPOCHS"] = "1"
env["CHECKPOINT_EVERY_STEPS"] = "500"
```

Ara checkpoint örneği:

```text
checkpoints/epoch_1_batch_500_lora/
checkpoints/epoch_1_batch_1000_lora/
...
checkpoints/epoch_1_lora/
```

## Değerlendirme

```bash
python scripts/run_eval.py
```

Değerlendirme metrikleri:

- Loss
- IoU

IoU, tahmin edilen mask ile gerçek mask arasındaki örtüşmeyi ölçer. `1.0` mükemmel, `0.0` hiç örtüşme yok anlamına gelir.

## Inference

```bash
python scripts/run_infer.py
```

Inference için önce eğitilmiş bir LoRA checkpoint gereklidir. Teslim senaryosunda önerilen checkpoint sırası:

1. `checkpoints/epoch_1_lora`
2. Eğer epoch tamamlanmadıysa son ara checkpoint, örn. `checkpoints/epoch_1_batch_6500_lora`

## Web Entegrasyonu İçin Minimum Hedef

Web sitesine bağlanacak minimum AI akışı:

1. Kullanıcı görsel yükler.
2. Backend görseli kaydeder.
3. Backend SAM3 + LoRA checkpoint ile mask üretir.
4. Mask dosyası `outputs/` altına kaydedilir.
5. Frontend orijinal görsel ve mask overlay gösterir.

Bu entegrasyon için backend tarafında en az bir endpoint gerekir:

```text
POST /predict
```

Girdi:

```text
image file
```

Çıktı:

```json
{
  "mask_url": "...",
  "overlay_url": "..."
}
```

## Rapor İçin Kullanılacak Gerçek Bilgiler

Rapor yazılırken mevcut duruma göre şu ifadeler kullanılmalıdır:

- Model, DACL10K üzerinde LoRA ile fine-tune edilmiştir.
- Kaggle T4 ortamında eğitim süresi nedeniyle 1 full epoch teslim hedefi olarak belirlenmiştir.
- Eğitim sırasında runtime kesintisi riskini azaltmak için her 500 batch'te ara checkpoint alınmıştır.
- Son model checkpoint'i inference ve web demo için kullanılacaktır.

Şu ifadeler ancak gerçekten tamamlandıysa kullanılmalıdır:

- “10 epoch eğitim tamamlandı.”
- “epoch_4_lora en iyi checkpoint olarak seçildi.”
- “Web entegrasyonu tamamen tamamlandı.”

## Ekip

Bu proje üç kişilik Software Engineering capstone ekibi tarafından geliştirilmektedir:

- Tarık Deveci
- Osman Yiğit Alver
- Ezgi Nilsu Kiraz
