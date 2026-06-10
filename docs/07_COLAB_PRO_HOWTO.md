# Google Colab Pro+ ile Tam Eğitim (How-To)

Bu rehber üç sorunu çözer:
- **Checkpoint kaybolması** — checkpointler düzenli olarak Drive'a yedeklenir
- **Bağlantı kopunca eğitimin durması** — eğitim arka plan process olarak çalışır
- **Drive mount'unun çökmesi** (`Transport endpoint is not connected`) — kod ve veri
  yerel diske alındığı için eğitim Drive'a hiç dokunmaz

Notebook: `colab_faz4_egitim.ipynb`

## Mimari (neden artık çökmüyor)

Eskiden repo ve veri seti **Drive üzerinden** çalışıyordu. Eğitim binlerce küçük
dosyayı Drive'dan okuyunca Drive FUSE mount'u çöküyordu (`Transport endpoint is
not connected`) ve `%cd` bile hata veriyordu.

Yeni mimaride:

| Şey | Konum | Neden |
|-----|-------|-------|
| Kod (repo) | Yerel `/content/SAM3` | `cwd` hatası olmaz, hızlı |
| Veri seti | Yerel `/content/dacl10k` | Eğitim Drive'dan okumaz → mount çökmez |
| Checkpoint | Yerel `/content/checkpoints` | Yazma her zaman güvenli |
| Log | Yerel `/content/logs` | Yazma her zaman güvenli |
| **Yedek** | Drive `MyDrive/sam3/` | Yedekleme döngüsü her 120 sn kopyalar → kalıcı |

Veri setinin kalıcı kopyası Drive'da `MyDrive/sam3/dacl10k/` altında durur; her
oturum başında bir kerelik yerel diske kopyalanır. Checkpoint ve loglar yerel
diske yazılır, ayrı bir arka plan döngüsü bunları her 120 saniyede Drive'a
kopyalar. Böylece **eğitim süreci Drive'a hiç dokunmaz** — Drive kopsa bile
eğitim durmaz, ilerleme yine de Drive'da birikir.

## 1) Veri setini Drive'a yükle (bir kerelik)

Drive'da şu yapıyı oluştur:

```
MyDrive/sam3/dacl10k/
├── images/{train,validation}/*.jpg
└── annotations/{train,validation}/*.json
```

> Veri seti Drive'da yoksa, Adım 4 onu HuggingFace'ten indirir ve Drive'a yedekler.

## 2) Notebook'u aç ve sırayla çalıştır

1. Runtime > Change runtime type > A100/L4 GPU seç
2. Adım 0: GPU kontrolü
3. Adım 1: Drive'ı bağla (izin ver), yollar tanımlanır
4. Adım 2: repo yerel diske klonlanır, paketler kurulur (`torchao` kaldırılır)
5. Adım 3: Secrets'a `HF_TOKEN` ekle (+ "Notebook access" aç), login yap
6. Adım 4: veri seti yerel diske hazırlanır (Drive'dan kopya / HF'ten indirme)
7. Adım 5: veri seti kontrolü
8. Adım 6: eğitim + Drive yedekleme döngüsü arka planda başlar — bu hücre bitince
   process'ler **arka planda çalışmaya devam eder**

## 3) Eğitimi takip et

- Adım 7 hücresini istediğin zaman tekrar çalıştırarak son logları gör
- Adım 8 ile checkpoint klasörlerini listele

## 4) Neden artık güvenli?

- `SAM3_DATA_DIR` / `SAM3_CHECKPOINT_DIR` ortam değişkenleri (`src/config.py`) ile
  eğitim **yerel diske** yönlendirilir → Drive mount'u çökmez
- Yedekleme döngüsü (rsync) checkpoint+log'u her 120 sn Drive'a kopyalar → kalıcı
- `subprocess.Popen` ile başlatılan eğitim, Colab Pro+'ın arka plan çalıştırma
  özelliğiyle bağlantı kopsa bile devam eder
- `CHECKPOINT_EVERY_STEPS` (config.py) ile epoch bitmeden de ara checkpoint alınır

## Sık hata ve hızlı çözüm

1. `Transport endpoint is not connected` → Runtime > Restart session, Adım 1'den
   başla. (Yeni mimaride eğitim sırasında bu hata artık oluşmaz.)
2. `ImportError: incompatible version of torchao` → Adım 2 `torchao`'yu kaldırır;
   bu hatayı görürsen Adım 2'yi tekrar çalıştır
3. `Secret HF_TOKEN does not exist` → Secrets'a `HF_TOKEN` ekle, "Notebook access" aç
4. `Veri seti bulunamadı` → Adım 4'ü çalıştır veya Drive'da `MyDrive/sam3/dacl10k/`
   yapısını kontrol et
5. `CUDA out of memory` → `src/config.py` içinde `BATCH_SIZE` zaten 1, gerekirse
   `IMAGE_SIZE` düşür
6. Eğitim durmuş gibi görünüyor → Adım 7 ile logu kontrol et, GPU kotası bitmiş
   olabilir (Runtime > Manage sessions)
