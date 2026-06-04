# Kaggle'da En Kolay Sekilde SAM3 DDP Egitimi (How-To)

Bu rehberin amaci: en az ugrastiran yolla Kaggle'da egitimi baslatmak.

Bu projede DDP script hazir:
- `scripts/run_train_ddp.py`

## Hangi yontem en kolay?

En kolay yontem: **Kaggle Notebook acip kodu direkt calistirmak**.
CLI zorunlu degil.

---

## Yontem A (Onerilen): Kaggle Notebook icinden calistir

## 1) Kaggle Notebook olustur

Notebook ayarlari:
- Accelerator: **GPU**
- GPU count: **2** (gorunuyorsa)
- Internet: **ON**

## 2) Projeyi Notebook ortamina al

En kolay iki secenek:
1. GitHub repo ise notebookta clone et
2. Repo zip dosyasini yukleyip `/kaggle/working` altina ac

Ornek (GitHub clone):

```bash
%cd /kaggle/working
!git clone <REPO_URL> SAM3
%cd /kaggle/working/SAM3
```

## 3) Paketleri kur

```bash
%cd /kaggle/working/SAM3
!pip install -q -r requirements.txt
```

## 4) HuggingFace token ayarla

1. Kaggle -> Add-ons -> Secrets
2. Yeni secret ekle: `HF_TOKEN`
3. Notebookta login yap:

```python
import os
from huggingface_hub import login

token = os.environ.get("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN secret bulunamadi")

login(token=token)
print("HF login tamam")
```

## 5) GPU sayisini kontrol et

```bash
!nvidia-smi
```

```python
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU sayisi:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
```

Beklenen: `GPU sayisi: 2`

## 6) DDP egitimini baslat

```bash
%cd /kaggle/working/SAM3
!torchrun --nproc_per_node=2 scripts/run_train_ddp.py
```

Eger GPU sayisi 1 ise:

```bash
!python scripts/run_train.py
```

## 7) Ciktilari kaydet

DDP egitim checkpointleri:
- `checkpoints/ddp_epoch_X_lora`
- `checkpoints/ddp_best_lora`

Notebook bitince `Save Version` ile ciktilari sakla.

---

## Yontem B (Opsiyonel): Kaggle CLI ile notebook push

Bu yontem daha teknik ama otomasyon icin iyi.

## 1) Bilgisayarda Kaggle API ayarla (Windows)

`kaggle.json` dosyasini su klasore koy:

`C:/Users/Lenovo/.kaggle/kaggle.json`

## 2) Push klasorunu hazirla

Klasorde en az su 2 dosya olmali:
- `.ipynb`
- `kernel-metadata.json`

Ornek metadata:

```json
{
  "id": "kullaniciadi/sam3-ddp-train",
  "title": "SAM3 DDP Train",
  "code_file": "sam3_kaggle_train.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true
}
```

## 3) Push et

```bash
kaggle kernels push -p <FOLDER_PATH>
```

---

## Sik hata ve hizli cozum

1. `HF_TOKEN secret bulunamadi`
- Secrets'a `HF_TOKEN` ekle.

2. `GPU sayisi 1`
- Notebook ayarinda 2 GPU secenegini tekrar kontrol et.

3. `Model indirilemiyor`
- Internet ON oldugunu kontrol et.
- HF login adimini tekrar calistir.

4. `CUDA out of memory`
- `src/config.py` icinde batch size dusur.
- Gerekirse augmentation veya image islemlerini hafiflet.

---

## Kisa ozet

En basit akisin ozeti:
1. Kaggle notebook ac
2. Repoyu `/kaggle/working/SAM3` altina al
3. `pip install -r requirements.txt`
4. `HF_TOKEN` ile login
5. `torchrun --nproc_per_node=2 scripts/run_train_ddp.py`

Bu kadar.