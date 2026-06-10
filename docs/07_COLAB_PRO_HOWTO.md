# Google Colab Pro+ ile Tam Eğitim (How-To)

Bu rehber, Kaggle'da yaşanan iki sorunu çözer:
- **Checkpoint kaybolması** — checkpointler artık Drive'a kaydedilir
- **Bağlantı kopunca eğitimin durması** — eğitim arka plan process olarak çalışır

Notebook: `colab_faz4_egitim.ipynb`

## 1) Veri setini Drive'a yükle

Drive'da şu yapıyı oluştur (bir kerelik):

```
MyDrive/sam3/dacl10k/
├── images/{train,validation}/*.jpg
└── annotations/{train,validation}/*.json
```

## 2) Notebook'u aç ve sırayla çalıştır

1. Runtime > Change runtime type > GPU seç
2. Adım 1: Drive'ı bağla (izin ver)
3. Adım 2: `REPO_URL` değişkenine GitHub repo adresini yaz — repo Drive'a klonlanır
4. Adım 3: Secrets'a `HF_TOKEN` ekle, login yap
5. Adım 4: veri seti kontrolü
6. Adım 5: eğitimi arka planda başlat — bu hücre bitince process **arka planda çalışmaya devam eder**

## 3) Eğitimi takip et

- Adım 6 hücresini istediğin zaman tekrar çalıştırarak son logları gör
- Adım 7 ile Drive'daki checkpoint klasörlerini listele

## 4) Neden artık güvenli?

- `SAM3_DATA_DIR` ve `SAM3_CHECKPOINT_DIR` ortam değişkenleri (`src/config.py`) sayesinde
  veri ve checkpointler doğrudan Drive'a okunur/yazılır — sekme kapansa bile kaybolmaz
- `subprocess.Popen` ile başlatılan eğitim, Colab Pro+'ın arka plan çalıştırma özelliğiyle
  bağlantı kopsa bile devam eder
- `CHECKPOINT_EVERY_STEPS` (config.py) ile epoch bitmeden de ara checkpoint alınır

## Sık hata ve hızlı çözüm

1. `Veri seti bulunamadı` → Drive'da `MyDrive/sam3/dacl10k/` yapısını kontrol et
2. `HF_TOKEN secret bulunamadı` → Secrets'a `HF_TOKEN` ekle
3. `CUDA out of memory` → `src/config.py` içinde `BATCH_SIZE` zaten 1, gerekirse `IMAGE_SIZE` düşür
4. Eğitim durmuş gibi görünüyor → Adım 6 ile logu kontrol et, GPU kotası bitmiş olabilir
   (Runtime > Manage sessions'tan kontrol et)
