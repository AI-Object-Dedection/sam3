# Teknik Rehber

Bu dosya projede kullanılan teknik kavramları başlangıç seviyesinde açıklar.
Bir terimi unuttuğunuzda buraya bakabilirsiniz.

---

## 1. Segmentation (Segmentasyon) Nedir?

Bir görseldeki nesnelerin **piksel piksel** nerede olduğunu bulmaktır.

```
Sınıflandırma:  "Bu fotoğrafta çatlak var"     → Sadece var/yok
Tespit:         "Çatlak şurada"                 → Kutu çizer (bounding box)
Segmentasyon:   "Çatlağın tam şekli bu"          → Piksel piksel işaretler (mask)
```

Bizim projemiz **segmentasyon** yapıyor — en detaylı olanı.

### Binary Mask Nedir?

Görüntüyle aynı boyutta bir matris:
- `1` = hasar var
- `0` = hasar yok

```
Orijinal görsel:        Binary mask:
┌──────────────┐        ┌──────────────┐
│  köprü       │        │ 0 0 0 0 0 0  │
│  fotoğrafı   │   →    │ 0 1 1 1 0 0  │  ← çatlak bölgesi
│              │        │ 0 0 1 1 0 0  │
│              │        │ 0 0 0 0 0 0  │
└──────────────┘        └──────────────┘
```

---

## 2. SAM3 (Segment Anything Model 3)

### Ne Yapar?

Meta'nın geliştirdiği büyük bir segmentasyon modeli. Herhangi bir görseldeki nesneleri otomatik olarak segmente edebilir.

### Bileşenleri

```
SAM3 Modeli
├── Image Encoder     → Görseli anlar (özellik çıkarır)
├── Prompt Encoder    → "Neyi segmente et?" sorusunu anlar
└── Mask Decoder      → Segmentation mask üretir
```

### Kodda Nasıl Kullanılıyor?

```python
from transformers import Sam3Model, Sam3Processor

# Processor: Görseli modelin anlayacağı formata çevirir
processor = Sam3Processor.from_pretrained("facebook/sam3")

# Model: Asıl segmentasyon işini yapar
model = Sam3Model.from_pretrained("facebook/sam3")
```

**Processor** = tercüman (görseli sayılara çevirir)
**Model** = beyin (sayılardan mask üretir)

### İlgili Dosyalar

- `src/model.py` → Model ve processor yükleme
- `src/config.py` → `MODEL_NAME = "facebook/sam3"`

---

## 3. DACL10K Veri Seti

### Genel Bilgi

| Özellik | Değer |
|---------|-------|
| Tam adı | Damage Classification 10K |
| Görsel sayısı | ~9.920 |
| Eğitim seti | 6.935 görsel |
| Doğrulama seti | 975 görsel |
| Sınıf sayısı | 19 (13 hasar + 6 bileşen) |
| Annotation formatı | LabelMe JSON |
| Lisans | CC BY-NC 4.0 (ticari olmayan kullanım) |

### 19 Sınıf

#### Hasar Türleri (13)

| Kısaltma | Tam Adı | Açıklama |
|----------|---------|----------|
| Crack | Crack | Çatlak |
| ACrack | Alligator Crack | Timsah derisi gibi ağ şeklinde çatlak |
| Spalling | Spalling | Beton yüzeyinden parça kopması |
| Efflorescence | Efflorescence | Beyaz tuz çiçeklenmesi |
| ExposedRebars | Exposed Rebars | Açığa çıkmış demir donatı |
| Cavity | Cavity | İçi boş oyuk |
| Restformwork | Restformwork | Kalıp izi kalıntısı |
| Rockpocket | Rockpocket | Taş cebi (agrega görünür) |
| Hollowareas | Hollow Areas | Boşluklu alanlar |
| Rust | Rust | Pas |
| Weathering | Weathering | Hava koşullarından aşınma |
| Graffiti | Graffiti | Sprey boya yazıları |
| Wetspot | Wetspot | Islak leke / su sızıntısı |

#### Köprü Bileşenleri (6)

| Kısaltma | Tam Adı | Açıklama |
|----------|---------|----------|
| Bearing | Bearing | Mesnet (köprüyü taşıyan parça) |
| Drainage | Drainage | Su tahliye sistemi |
| EJoint | Expansion Joint | Genleşme derzi |
| JTape | Joint Tape | Derz bandı |
| PEquipment | Protective Equipment | Koruyucu ekipman |
| WConccor | Washouts/Concrete Corrosion | Beton korozyonu |

### Annotation (İşaretleme) Formatı

Datasetninja.com versiyonu **Supervisely formatında** JSON dosyaları kullanır.
Her görsel için ayrı bir `.json` dosyası var. Yapısı:

```json
{
  "description": "",
  "size": {
    "height": 960,
    "width": 1280
  },
  "objects": [
    {
      "classTitle": "Crack",
      "points": {
        "exterior": [[100, 200], [150, 250], [120, 300]],
        "interior": []
      }
    },
    {
      "classTitle": "Rust",
      "points": {
        "exterior": [[400, 100], [500, 100], [500, 200], [400, 200]],
        "interior": []
      }
    }
  ]
}
```

**Önemli:** Bir görselde birden fazla hasar olabilir (multi-label).

### Klasör Yapısı

```
data/dacl10k/
├── images/
│   ├── train/           # 6.935 adet .jpg dosyası
│   └── validation/      # 975 adet .jpg dosyası
└── annotations/
    ├── train/           # 6.935 adet .json dosyası
    └── validation/      # 975 adet .json dosyası
```

### Veri Setini Kurma

Datasetninja.com'dan indirmek için:

```bash
# 1. Kurulum aracını yükle (sadece bir kez)
pip install dataset-tools

# 2. İndir ve düzenle
python scripts/setup_dataset.py
```

`setup_dataset.py` scripti:
1. Datasetninja'dan ~4-5 GB veri indirir
2. Dosyaları doğru klasör yapısına taşır
3. Annotation dosya adlarını düzenler (`resim.jpg.json` → `resim.json`)

### İlgili Dosyalar

- `src/dataset.py` → Veri okuma ve mask oluşturma
- `src/config.py` → Veri seti yolları
- `scripts/inspect_dataset.py` → Veri setini inceleme aracı

---

## 4. LoRA (Low-Rank Adaptation)

### Problem

SAM3 gibi büyük modellerin milyonlarca parametresi var. Hepsini eğitmek:
- Çok fazla GPU belleği gerektirir (belki 40GB+)
- Saatler/günler sürer
- Orijinal modelin bilgisini bozma riski taşır

### Çözüm: LoRA

Modelin **tamamını** eğitmek yerine, küçük **ek katmanlar** (adaptörler) ekleyip sadece onları eğitmek.

```
Normal Fine-tuning:
┌──────────────────────────────┐
│  Tüm model parametreleri     │  ← HEPSİ güncellenir
│  (100 milyon parametre)      │     Çok bellek, çok zaman
└──────────────────────────────┘

LoRA Fine-tuning:
┌──────────────────────────────┐
│  Orijinal parametreler       │  ← DONDURULMUŞ (değişmez)
│  (100 milyon parametre)      │
│  ┌────────────────────┐      │
│  │  LoRA adaptörleri  │      │  ← SADECE BUNLAR güncellenir
│  │  (1-2 milyon)      │      │     Az bellek, hızlı
│  └────────────────────┘      │
└──────────────────────────────┘
```

### LoRA Ayarları (config.py)

| Ayar | Değer | Ne Anlama Geliyor? |
|------|-------|--------------------|
| `LORA_RANK` | 8 | Ek katmanların boyutu. Düşük = daha az parametre, hızlı ama sınırlı. Yüksek = daha çok parametre, yavaş ama güçlü. |
| `LORA_ALPHA` | 16 | LoRA'nın etkisini ölçekler. Genellikle rank'ın 2 katı iyi çalışır. |
| `LORA_DROPOUT` | 0.1 | Eğitim sırasında rastgele %10 bağlantıyı kapatır. Aşırı öğrenmeyi (overfitting) önler. |
| `target_modules` | `["q_proj", "v_proj"]` | LoRA'nın uygulanacağı katmanlar. Attention mekanizmasının sorgu ve değer katmanları. |

### İlgili Dosyalar

- `src/lora.py` → LoRA konfigürasyonu ve uygulama
- `src/config.py` → LoRA ayarları

---

## 5. Eğitim Süreci

### Temel Kavramlar

| Terim | Açıklama | Benzetme |
|-------|----------|----------|
| **Epoch** | Modelin tüm eğitim verisini 1 kez görmesi | Ders kitabını baştan sona 1 kez okumak |
| **Batch** | Bir seferde işlenen görsel grubu | Sayfaları 2'şer 2'şer okumak (batch_size=2) |
| **Loss** | Modelin hatasının sayısal değeri | Sınav notu (düşük = daha iyi) |
| **Learning Rate** | Öğrenme hızı | Çok hızlı = dikkatsiz öğrenme, çok yavaş = hiç öğrenememe |
| **Optimizer** | Parametreleri güncelleyen algoritma | Öğretmen (hatalardan nasıl ders çıkarılacağını söyler) |
| **Gradient** | Hatanın hangi yöne gittiğini gösteren bilgi | Pusula (hangi yöne düzeltme yapılmalı) |
| **Checkpoint** | Eğitim sırasında modelin kaydedilmiş hali | Oyunda save point |

### Eğitim Döngüsü

```
Her epoch için:
  Her batch için:
    1. Görselleri modele ver          (forward pass)
    2. Tahmin ile doğruyu karşılaştır (loss hesapla)
    3. Hatayı geri yay                (backward pass)
    4. Parametreleri güncelle         (optimizer.step)

  Epoch sonunda:
    5. Validation setinde test et
    6. Modeli kaydet (checkpoint)
```

### İlgili Dosyalar

- `src/train.py` → Eğitim döngüsü
- `src/evaluate.py` → Değerlendirme (IoU hesaplama)
- `scripts/run_train.py` → Eğitimi başlatan script

---

## 6. Değerlendirme Metrikleri

### IoU (Intersection over Union)

Modelin tahmin ettiği mask ile gerçek mask'ın ne kadar örtüştüğünü ölçer.

```
Tahmin edilen mask:     Gerçek mask:
┌──────────┐            ┌──────────┐
│    ████  │            │   █████  │
│    ████  │            │   █████  │
│          │            │          │
└──────────┘            └──────────┘

        Kesişim (ikisi de 1):  ███
        Birleşim (en az biri 1): ██████

IoU = Kesişim / Birleşim = 3/6 = 0.50
```

| IoU Değeri | Anlam |
|-----------|-------|
| 1.0 | Mükemmel (tahmin = gerçek) |
| 0.7+ | İyi |
| 0.5+ | Orta |
| 0.3 altı | Zayıf |

### İlgili Dosyalar

- `src/evaluate.py` → `calculate_iou()` fonksiyonu

---

## 7. Sözlük

Projede karşılaşabileceğiniz diğer terimler:

| Terim | Açıklama |
|-------|----------|
| **Tensor** | Çok boyutlu sayı dizisi. Görseller modelde tensor olarak işlenir. |
| **GPU/CUDA** | Grafik kartı ile hızlı hesaplama. `torch.cuda.is_available()` ile kontrol edilir. |
| **Pretrained** | Önceden eğitilmiş. SAM3 zaten büyük veri üzerinde eğitilmiş. |
| **Fine-tuning** | Önceden eğitilmiş modeli yeni bir görev için ince ayar yapmak. |
| **Frozen** | Donmuş parametreler. LoRA'da orijinal parametreler frozen kalır. |
| **DataLoader** | PyTorch'un veriyi batch'ler halinde modele veren aracı. |
| **Forward pass** | Veriyi modelden geçirip tahmin alma. |
| **Backward pass** | Hatayı geri yayarak öğrenme bilgisi (gradient) hesaplama. |
| **Overfitting** | Modelin eğitim verisini ezberlemesi ama yeni verilerde başarısız olması. |
| **Inference** | Eğitilmiş modelle tahmin yapma (eğitim yok, sadece kullanma). |
