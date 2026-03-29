# Geliştirme Adımları

Bu dosya projenin adım adım nasıl geliştirileceğini tanımlar.
Her faz için: ne yapılacak, hangi dosyalar değişecek, nasıl test edilecek.

**Kural:** Bir faz tamamlanmadan sonrakine geçmeyin.

---

## Genel Bakış

| Faz | İsim | Durum | Açıklama |
|-----|------|-------|----------|
| 1 | Proje İskeleti | ✅ Tamamlandı | Klasör yapısı, boş dosyalar, config |
| 2 | Veri Seti Entegrasyonu | ✅ Tamamlandı | DACL10K numpy okuma, doğrulama |
| 3 | Model + LoRA Kurulumu | ✅ Tamamlandı | SAM3 yükleme, LoRA uygulama, doğrulama |
| 4 | Eğitim + Değerlendirme | ⬜ Bekliyor | Training loop, loss, IoU |
| 5 | Inference + Demo | ⬜ Bekliyor | Tek görsel tahmini, görselleştirme |

---

## Faz 1: Proje İskeleti ✅

**Durum:** Tamamlandı

**Yapılan işler:**
- [x] Klasör yapısı oluşturuldu
- [x] Tüm Python dosyaları iskelet olarak yazıldı
- [x] `config.py` merkezi ayarlarla dolduruldu
- [x] `requirements.txt` hazırlandı
- [x] `README.md` yazıldı
- [x] `.gitignore` oluşturuldu
- [x] Git repository başlatıldı

---

## Faz 2: Veri Seti Entegrasyonu ✅

**Durum:** Tamamlandı

**Amaç:** DACL10K veri setini projeye entegre etmek ve doğru okunduğunu doğrulamak.

### Yapılacak İşler

- [x] `config.py` — yolları gerçek DACL10K yapısına güncelle (numpy formatı)
- [x] `dataset.py` — numpy .npy formatına uyumla
- [x] `inspect_dataset.py` — numpy bazlı inceleme aracına güncelle
- [x] DACL10K veri setini Kaggle'dan indir ve `data/dacl10k/` altına bağla (junction)
- [x] `python scripts/inspect_dataset.py` çalıştırarak veriyi doğrula
- [x] Dataset'ten örnek görsel + mask çıktısını görsel olarak kontrol et (`visualize_sample.py`)

### Veri Formatı: Datasetninja.com (Supervisely)

Datasetninja.com versiyonu kullanılıyor:
- Görseller: `.jpg` dosyaları (değişken boyut, RGB)
- Annotationlar: `.json` dosyaları (Supervisely formatı, polygon şekilleri)
- Mask'lar runtime'da JSON polygon'lardan oluşturuluyor (PIL.ImageDraw ile)

### Değişen Dosyalar

| Dosya | Değişiklik |
|-------|-----------|
| `src/config.py` | JSON+JPG formatına uygun yol tanımları ✅ |
| `src/dataset.py` | Supervisely JSON okuma + polygon→mask dönüşümü ✅ |
| `scripts/inspect_dataset.py` | JSON+JPG bazlı inceleme ✅ |
| `scripts/setup_dataset.py` | Yeni: datasetninja indirme ve kurulum scripti ✅ |

### Doğrulama Kontrol Listesi

```
✅ python scripts/setup_dataset.py ile veri seti kuruldu
✅ python scripts/inspect_dataset.py hatasız çalışıyor
✅ Annotation sayısı doğru (train: 6935, val: 975)
✅ 19 sınıfın hepsi veride mevcut
✅ Polygon → mask dönüşümü doğru çalışıyor
```

---

## Faz 3: Model + LoRA Kurulumu 🔄

**Durum:** Tamamlandı

**Amaç:** SAM3 modelini yüklemek, LoRA uygulamak ve modelin çalıştığını doğrulamak.

### Yapılacak İşler

- [x] `facebook/sam3` modelini HuggingFace'ten indir ve test et
- [x] `AutoModel` ve `AutoProcessor` sınıflarının çalıştığını doğrula
- [x] `lora.py`'deki `apply_lora()` fonksiyonunu test et
- [x] Eğitilebilir parametre oranını kontrol et (%0.26 — q_proj + v_proj)
- [x] Modele örnek bir görsel verip çıktı alınabildiğini doğrula

### Değişecek Dosyalar

| Dosya | Olası Değişiklik |
|-------|-----------------|
| `src/model.py` | API değişiklikleri varsa güncelleme |
| `src/lora.py` | `target_modules` ayarı (modelin gerçek katman adlarına göre) |
| `src/config.py` | Gerekirse model ayarları |

### Doğrulama Kontrol Listesi

```
✅ model = load_model() hatasız çalışıyor
✅ processor = load_processor() hatasız çalışıyor
✅ model = apply_lora(model) hatasız çalışıyor
✅ print_trainable_params(model) → %0.26 (q_proj + v_proj)
✅ Vision encoder örnek görsel ile test edildi
```

### Öğrenilen Bilgiler

- `Sam3VideoModel` yapısı: `detector_model` + `tracker_model` + `tracker_neck`
- Vision encoder yolu: `model.base_model.model.detector_model.vision_encoder`
- Model forward pass için `inference_session` gerekiyor (video modeli)
- LoRA eğitilebilir oran: %0.26 (sadece q_proj + v_proj) — Faz 4'te gerekirse artırılabilir

### AI Agent'a Not

Bu faz kritiktir. Olası sorunlar:
- `Sam3Model` / `Sam3Processor` sınıf adları transformers sürümüne göre farklı olabilir
- `target_modules` modelin gerçek katman adlarıyla eşleşmeli — `model.named_modules()` ile kontrol et
- GPU bellek hatası alınırsa batch size veya model boyutu küçültülebilir
- Modelin çıktı yapısını anlamak önemli (mask decoder'ın ne döndürdüğü)

---

## Faz 4: Eğitim + Değerlendirme ⬜

**Durum:** Bekliyor (Faz 3 tamamlanınca başlayacak)

**Amaç:** Modeli DACL10K üzerinde eğitmek ve performansını ölçmek.

### Yapılacak İşler

- [ ] `train.py` — forward pass implementasyonu
- [ ] `train.py` — loss fonksiyonu seçimi ve implementasyonu
- [ ] `evaluate.py` — gerçek tahmin kodu
- [ ] `scripts/run_train.py` — DataLoader oluşturma
- [ ] `scripts/run_eval.py` — checkpoint yükleme + DataLoader
- [ ] Küçük veri alt kümesiyle (10-20 görsel) test eğitimi
- [ ] Loss'un düşüp düşmediğini kontrol et
- [ ] Validation IoU'nun mantıklı olup olmadığını kontrol et
- [ ] Tam eğitim (10 epoch)

### Değişecek Dosyalar

| Dosya | Değişiklik |
|-------|-----------|
| `src/train.py` | Forward pass + loss hesaplama (TODO'lar) |
| `src/evaluate.py` | Gerçek tahmin + IoU hesaplama (TODO'lar) |
| `scripts/run_train.py` | DataLoader oluşturma (TODO) |
| `scripts/run_eval.py` | Checkpoint yükleme + DataLoader (TODO) |

### Doğrulama Kontrol Listesi

```
□ 1 batch forward pass hatasız çalışıyor mu?
□ Loss değeri makul bir sayı mı? (NaN veya Inf değil)
□ Loss epoch'lar boyunca düşüyor mu?
□ Validation IoU 0'dan büyük mü?
□ Checkpoint dosyaları kaydediliyor mu?
□ Checkpoint'ten model geri yüklenebiliyor mu?
```

### AI Agent'a Not

Bu fazda dikkat edilecekler:
- **Loss fonksiyonu:** Segmentasyon için genellikle BCE (Binary Cross Entropy) veya Dice Loss kullanılır. BCE ile başlamak daha basit.
- **DataLoader:** `batch_size=2` başlangıç için yeterli, GPU bellek hatasında 1'e düşür
- **Overfitting kontrolü:** Train loss düşerken val loss artıyorsa overfitting var
- **Gradient accumulation:** Bellek yetmezse küçük batch + gradient accumulation düşünülebilir
- **Mixed precision:** `torch.cuda.amp` ile bellek tasarrufu sağlanabilir (opsiyonel)

---

## Faz 5: Inference + Demo ⬜

**Durum:** Bekliyor (Faz 4 tamamlanınca başlayacak)

**Amaç:** Eğitilmiş modeli kullanarak tek görsel üzerinde tahmin yapmak ve sonuçları görselleştirmek.

### Yapılacak İşler

- [ ] `infer.py` — gerçek model forward pass
- [ ] `scripts/run_infer.py` — checkpoint yükleme
- [ ] Tahmin sonucunu görselleştirme (orijinal görsel + mask overlay)
- [ ] Birkaç örnek görsel üzerinde demo çalıştırma
- [ ] Sonuçları `outputs/` klasörüne kaydetme

### Değişecek Dosyalar

| Dosya | Değişiklik |
|-------|-----------|
| `src/infer.py` | Gerçek inference implementasyonu (TODO) |
| `scripts/run_infer.py` | Checkpoint yükleme (TODO) |
| `src/utils.py` | Görselleştirme fonksiyonu eklenebilir |

### Doğrulama Kontrol Listesi

```
□ Tek bir görselde tahmin alınabiliyor mu?
□ Tahmin edilen mask mantıklı görünüyor mu?
□ Sonuçlar outputs/ klasörüne kaydediliyor mu?
□ Orijinal görsel + mask overlay görselleştirmesi yapılabiliyor mu?
```

---

## Faz Durumlarını Güncelleme

Bir fazı tamamladığınızda:

1. Bu dosyadaki ilgili fazın durumunu `✅ Tamamlandı` olarak değiştirin
2. `CLAUDE.md`'deki faz tablosunu güncelleyin
3. Commit atın: `[faz-X] faz açıklaması`

Bu sayede herhangi bir AI agent projeye baktığında tam olarak nerede olduğunuzu bilir.
