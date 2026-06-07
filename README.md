# SAM3 Bridge Damage Segmentation

Bu proje, `facebook/sam3` modelini DACL10K köprü hasarı veri seti üzerinde LoRA ile fine-tune ederek hasarlı bölgeler için segmentation mask üretmeyi amaçlar.

Proje bir capstone çalışmasıdır. Önceki tam eğitim koşusunda model 10 epoch eğitilmiş ve metrikler `docs/05_FAZ4_EGITIM_RAPORU.md` altında raporlanmıştır. Mevcut odak, bu sonucu teslim akışına taşımak için checkpoint/inference çıktısını netleştirmek ve web tabanlı prototipe bağlamaktır.

## Güncel Durum

| Başlık | Durum |
|---|---|
| Proje iskeleti | Tamamlandı |
| DACL10K veri okuyucu | `.npy` ve DatasetNinja `.jpg + .json` formatlarını destekliyor |
| SAM3 yükleme | Kaggle T4 için `fp16` ve `low_cpu_mem_usage=True` ile çalışıyor |
| LoRA entegrasyonu | Tamamlandı |
| Önceki tam eğitim | 10 epoch tamamlandı, metrikler `docs/05_FAZ4_EGITIM_RAPORU.md` içinde |
| Güncel Kaggle koşusu | Yeniden üretim / entegrasyon checkpoint'i almak için kullanılıyor |
| Ara checkpoint | Yeni koşularda her 500 batch'te LoRA checkpoint kaydediliyor |
| Web entegrasyonu | Sıradaki iş: seçilen checkpoint'i backend inference hattına bağlamak |
| Rapor | Önceki eğitim metrikleri korunacak, web demo ekran görüntüleri eklenecek |

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

Önceki tam eğitim RTX 4090 Laptop üzerinde yaklaşık 21.5 saatte tamamlandı. Bu koşuda 6935 eğitim görseli, 975 validation görseli ve 10 epoch kullanıldı.

Önceki tam eğitimden raporlanan ana sonuçlar:

| Metrik | Değer |
|---|---:|
| Son train loss | 0.1743 |
| Son train IoU | 0.6949 |
| En iyi validation IoU civarı | 0.5798 |
| Önerilen checkpoint | `checkpoints/epoch_4_lora` |

Kaggle T4 üzerinde yeniden üretim/entegrasyon koşusunda ölçülen hız yaklaşık `3.9 sn / batch` seviyesindedir.

| Koşu | Tahmini süre | Kullanım amacı |
|---|---:|---|
| 1000 train + 200 val, 2 epoch | 2.5-3 saat | Hızlı test |
| Full train + full val, 1 epoch | 8.5-9 saat | Teslim için minimum model |
| Full train + full val, 2 epoch | 17-18 saat | Daha iyi checkpoint adayı |
| 10 epoch | 85-90 saat | Kaggle T4 için pratik değil; önceki tam eğitim sonucu kullanılmalı |

Teslim için önerilen hedef: önceki 10 epoch sonucunu ana model sonucu olarak kullanmak; Kaggle'daki 1 epoch koşusunu ise yeniden üretim, checkpoint doğrulama ve web entegrasyonu için kullanmak. Eğer eski `epoch_4_lora` checkpoint dosyası eldeyse web entegrasyonunda öncelikli olarak o kullanılmalıdır.

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

1. Önceki tam eğitimden gelen `checkpoints/epoch_4_lora`
2. Eğer `epoch_4_lora` elde değilse `checkpoints/epoch_5_lora`
3. Eski checkpoint dosyaları bulunamıyorsa güncel Kaggle koşusundan `checkpoints/epoch_1_lora`
4. Epoch tamamlanmadıysa son ara checkpoint, örn. `checkpoints/epoch_1_batch_6500_lora`

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
- Önceki tam eğitim koşusunda 10 epoch tamamlanmıştır.
- Raporlanan en iyi validation IoU yaklaşık `0.5798` seviyesindedir.
- Önerilen eski checkpoint `epoch_4_lora` olarak kaydedilmiştir.
- Güncel Kaggle koşuları, eski sonucu doğrulamak ve web entegrasyonu için kullanılabilir checkpoint üretmek amacıyla yürütülmektedir.
- Yeni koşularda runtime kesintisi riskini azaltmak için her 500 batch'te ara checkpoint alınmaktadır.

Şu ifadeler ancak dosya/çıktı gerçekten eldeyse kullanılmalıdır:

- “Web demosu eğitilmiş checkpoint ile canlı maske üretmektedir.”
- “Web entegrasyonu tamamen tamamlandı.”

## Ekip

Bu proje üç kişilik Software Engineering capstone ekibi tarafından geliştirilmektedir:

- Tarık Deveci
- Osman Yiğit Alver
- Ezgi Nilsu Kiraz
