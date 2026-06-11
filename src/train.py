"""
train.py — Eğitim Döngüsü

Bu dosya modelin eğitim sürecini yönetir.

Eğitim döngüsü (training loop) nedir?
Model, veri setini tekrar tekrar görür ve her seferinde:
1. Tahmin yapar  (forward pass)
2. Tahmini doğruyla karşılaştırır  (loss hesaplar)
3. Hatalarından öğrenir  (backward pass + optimizer)

Bu döngüye "epoch" denir. Örneğin 3 epoch = veri setini 3 kez görmek.

SAM3 akışı (tam pipeline):
    Görsel + "damage" metni
    → Vision Encoder   (görseli analiz eder)
    → Text Encoder     (metni analiz eder)
    → DETR Encoder     (görsel + metin birleşir)
    → DETR Decoder     (nesneleri bulur)
    → Mask Decoder     (piksel piksel mask üretir)

Çıktı olarak outputs.semantic_seg kullanıyoruz: (B, 1, 288, 288) boyutlu ham logits.
"""

import random
import math
import os
import glob
import shutil

import torch

from src.config import Config
from src.dataset import DACL10K_CLASSES
from src.evaluate import calculate_iou
from src.losses import build_loss_fn
from src.utils import log, ensure_dir, load_training_state, save_training_state


def _prune_batch_checkpoints(checkpoint_dir, epoch_num, keep=None):
    """
    Bir epoch'a ait ara (batch) checkpoint klasörlerini siler.

    Neden? Ara checkpoint'ler (`epoch_N_batch_M_lora`) her 500 adımda birikir.
    Hepsi Drive'a kopyalanırsa hem yer dolar hem yedekleme yavaşlar. En son
    ihtiyacımız olan checkpoint kaydedildikten sonra eskileri temizleriz.

    Args:
        checkpoint_dir: Checkpoint klasörü
        epoch_num: Hangi epoch'un ara checkpoint'leri (1-tabanlı, ör: 3)
        keep: Silinmeyecek klasörün tam yolu (en yenisini koru), yoksa hepsini sil
    """
    desen = os.path.join(checkpoint_dir, f"epoch_{epoch_num}_batch_*_lora")
    for yol in glob.glob(desen):
        if keep is not None and os.path.abspath(yol) == os.path.abspath(keep):
            continue
        try:
            shutil.rmtree(yol)
        except OSError:
            pass  # Silinemezse sorun değil, sadece yer kaplar


def _create_scheduler(optimizer, steps_per_epoch):
    """Config ayarlarına göre LR scheduler oluşturur."""
    scheduler_name = Config.LR_SCHEDULER.lower()
    total_steps = max(steps_per_epoch * Config.NUM_EPOCHS, 1)

    if scheduler_name == "none":
        return None

    if scheduler_name == "cosine_warmup":
        warmup_steps = int(total_steps * Config.WARMUP_RATIO)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))

            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_ratio = Config.MIN_LR_RATIO
            return min_ratio + (1.0 - min_ratio) * cosine_decay

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if scheduler_name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=Config.LEARNING_RATE,
            total_steps=total_steps,
            pct_start=Config.WARMUP_RATIO,
            div_factor=10.0,
            final_div_factor=20.0,
        )

    raise ValueError("Config.LR_SCHEDULER gecersiz. Beklenen: none, cosine_warmup, onecycle")


def train_one_epoch(model, dataloader, optimizer, scheduler, loss_fn, scaler, epoch,
                    skip_steps=0, on_checkpoint=None):
    """
    Modeli tek bir epoch boyunca eğitir.

    Args:
        model:       Eğitilecek model (LoRA uygulanmış)
        dataloader:  Eğitim verisi (batch'ler halinde)
        optimizer:   Parametre güncelleyici (AdamW)
        loss_fn:     Kayıp fonksiyonu (BCEWithLogitsLoss)
        scaler:      Mixed precision için GradScaler
        epoch:       Kaçıncı epoch olduğu (log için)
        skip_steps:  Bu epoch'ta atlanacak batch sayısı. Eğitim epoch ortasında
                     kesilip devam ettiğinde, daha önce işlenen batch'leri tekrar
                     işlememek için baştan bu kadarı atlanır (GPU'ya hiç verilmez).
        on_checkpoint: Ara checkpoint alındığında çağrılan fonksiyon.
                     İmza: on_checkpoint(resume_step, latest_ckpt_adi). Eğitim
                     durumunu (kaçıncı adımda olduğumuzu) Drive'a kaydetmek için.

    Returns:
        tuple: (ortalama_loss, ortalama_iou)
    """
    model.train()  # Modeli eğitim moduna al
    total_loss = 0.0
    total_iou = 0.0
    trained_batches = 0          # Gerçekten eğitilen batch sayısı (atlananlar hariç)
    prev_batch_ckpt = None       # En son alınan ara checkpoint (eskisini silmek için)

    if skip_steps > 0:
        log(f"Devam: bu epoch'ta ilk {skip_steps} batch atlanıyor (zaten işlenmişti).")

    for batch_idx, batch in enumerate(dataloader):
        # Devam ederken: daha önce işlenmiş batch'leri atla (GPU işlemi yapma)
        if batch_idx < skip_steps:
            continue

        # Batch içindeki tensor'ları GPU'ya taşı
        pixel_values = batch["pixel_values"].to(Config.DEVICE)
        input_ids = batch["input_ids"].to(Config.DEVICE)
        attention_mask = batch["attention_mask"].to(Config.DEVICE)
        gt_mask = batch["ground_truth_mask"].to(Config.DEVICE)  # (B, 288, 288)

        # ---- İleri Geçiş (Forward Pass) ----
        # Mixed precision: bazı hesaplamalar 16-bit ile yapılır → daha az bellek
        with torch.amp.autocast("cuda", enabled=(Config.DEVICE == "cuda")):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # semantic_seg: (B, 1, 288, 288) → squeeze → (B, 288, 288)
            # Ham logits (sigmoid uygulanmamış) — loss_fn bunu bekliyor
            pred_mask = outputs.semantic_seg.squeeze(1)

            # ---- Loss Hesaplama ----
            # BCEWithLogitsLoss: her piksel için "hasar var mı?" sorusunun cevabını değerlendirir
            # Otomatik olarak sigmoid uygular, ayrıca yapmanıza gerek yok
            loss = loss_fn(pred_mask, gt_mask)

        # ---- Geri Yayılım (Backward Pass) ----
        optimizer.zero_grad()          # Önceki gradientleri temizle
        scaler.scale(loss).backward()  # Gradientleri hesapla (mixed precision ile)
        scaler.step(optimizer)         # Parametreleri güncelle
        scaler.update()                # Scaler'ı güncelle
        if scheduler is not None:
            scheduler.step()

        # İstatistikleri biriktir
        iou = calculate_iou(pred_mask.detach(), gt_mask)
        total_loss += loss.item()
        total_iou += iou
        trained_batches += 1

        # Her 10 batch'te bir durum bilgisi yazdır
        if (batch_idx + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            log(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} "
                f"| Loss: {loss.item():.4f} | IoU: {iou:.4f} | LR: {current_lr:.6e}")

        if (
            Config.CHECKPOINT_EVERY_STEPS > 0
            and (batch_idx + 1) % Config.CHECKPOINT_EVERY_STEPS == 0
        ):
            ckpt_adi = f"epoch_{epoch+1}_batch_{batch_idx+1}_lora"
            ara_kayit_yolu = os.path.join(Config.CHECKPOINT_DIR, ckpt_adi)
            model.save_pretrained(ara_kayit_yolu)
            log(f"Ara checkpoint kaydedildi: {ara_kayit_yolu}")

            # Eğitim durumunu kaydet — epoch ortasında kesilirse buradan devam edilir.
            # resume_step = işlenen toplam batch (atlananlar + bu turdakiler).
            if on_checkpoint is not None:
                on_checkpoint(batch_idx + 1, ckpt_adi)

            # Bir önceki ara checkpoint'i sil — sadece en yenisini tut (yer/yedekleme)
            if prev_batch_ckpt is not None:
                try:
                    shutil.rmtree(prev_batch_ckpt)
                except OSError:
                    pass
            prev_batch_ckpt = ara_kayit_yolu

    # Ortalamalar: yalnızca gerçekten eğitilen batch'lere böl (atlananlar hariç)
    avg_loss = total_loss / max(trained_batches, 1)
    avg_iou = total_iou / max(trained_batches, 1)
    return avg_loss, avg_iou


def train(model, train_dataloader, val_dataloader=None):
    """
    Tüm eğitim sürecini yönetir.

    Args:
        model:             Eğitilecek model (LoRA uygulanmış)
        train_dataloader:  Eğitim verisi
        val_dataloader:    Doğrulama verisi (opsiyonel)
    """
    from src.evaluate import evaluate

    log("Eğitim başlıyor...")
    log(f"Epoch sayısı   : {Config.NUM_EPOCHS}")
    log(f"Learning rate  : {Config.LEARNING_RATE}")
    log(f"Cihaz          : {Config.DEVICE}")
    log(f"Text prompt    : '{Config.TEXT_PROMPT}'")
    log(f"Mask boyutu    : {Config.MASK_OUTPUT_SIZE}x{Config.MASK_OUTPUT_SIZE}")
    log(f"Loss tipi      : {Config.LOSS_TYPE}")
    log(f"LR scheduler   : {Config.LR_SCHEDULER}")
    log(f"Early stopping : {Config.EARLY_STOPPING}")

    # Checkpoint klasörünü oluştur
    ensure_dir(Config.CHECKPOINT_DIR)

    # Kayıp fonksiyonu: Config'e göre dinamik seçilir
    loss_fn = build_loss_fn()

    # Optimizer: sadece LoRA parametrelerini eğit (frozen parametreler hariç)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE,
    )
    scheduler = _create_scheduler(optimizer, len(train_dataloader))

    # Mixed precision scaler: GPU belleğinden tasarruf sağlar
    # CPU'da eğitiliyorsa scaler devre dışı
    use_amp = (Config.DEVICE == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    no_improvement_epochs = 0
    start_epoch = 0
    resume_step = 0   # Başlanacak epoch'ta atlanacak batch sayısı (epoch ortası devam)

    # ---- Kaldığı yerden devam (resume) ----
    # Önceki bir eğitim varsa durumunu oku ve kaldığı yerden devam et.
    # (Model ağırlıkları zaten load_or_apply_lora ile yüklendi; burada
    #  epoch sayacı, en iyi loss VE epoch içindeki adım bilgisini geri yüklüyoruz.)
    state = load_training_state(Config.CHECKPOINT_DIR)
    if state is not None:
        # Yeni format: completed_epochs + resume_step. Eski format: last_epoch.
        start_epoch = state.get("completed_epochs", state.get("last_epoch", 0))
        resume_step = state.get("resume_step", 0)
        best_val_loss = state.get("best_val_loss", float("inf"))
        no_improvement_epochs = state.get("no_improvement_epochs", 0)
        log(f"Devam ediliyor: epoch {start_epoch}/{Config.NUM_EPOCHS}, "
            f"epoch içi adım {resume_step} (en iyi val_loss: {best_val_loss:.4f})")

        # LR scheduler'ı kaldığımız noktaya getir (tamamlanan epoch'lar + epoch içi adımlar)
        if scheduler is not None:
            gecen_adim = start_epoch * len(train_dataloader) + resume_step
            for _ in range(gecen_adim):
                scheduler.step()

    if start_epoch >= Config.NUM_EPOCHS:
        log("Eğitim zaten tamamlanmış (tüm epoch'lar bitti). Yeni epoch yok.")
        return

    # Her epoch için eğitim yap
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        log(f"\n{'='*50}")
        log(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} başladı")

        # Yalnızca devam edilen (ilk) epoch'ta batch atla; sonraki epoch'lar baştan.
        skip_steps = resume_step if epoch == start_epoch else 0

        # Ara checkpoint alındığında epoch içi durumu kaydeden fonksiyon.
        # epoch/best_val_loss bu döngü turunda sabit; mid-epoch state için yeterli.
        def _ara_durum_kaydet(adim, ckpt_adi, _epoch=epoch):
            save_training_state(Config.CHECKPOINT_DIR, {
                "completed_epochs": _epoch,        # bu epoch henüz bitmedi
                "resume_step": adim,               # bu epoch'ta işlenen batch sayısı
                "latest_ckpt": ckpt_adi,           # yüklenecek en son checkpoint
                "last_epoch": _epoch,              # eski format/notebook gösterimi için
                "best_val_loss": best_val_loss,
                "no_improvement_epochs": no_improvement_epochs,
            })

        # Eğitim
        train_loss, train_iou = train_one_epoch(
            model, train_dataloader, optimizer, scheduler, loss_fn, scaler, epoch,
            skip_steps=skip_steps, on_checkpoint=_ara_durum_kaydet,
        )
        log(f"Epoch {epoch+1} [train] | Loss: {train_loss:.4f} | IoU: {train_iou:.4f}")

        # Validation (eğer val_dataloader verilmişse)
        if val_dataloader is not None:
            results = evaluate(model, val_dataloader, loss_fn)
            log(f"Epoch {epoch+1} [val]   | Loss: {results['mean_loss']:.4f} "
                f"| IoU: {results['mean_iou']:.4f}")

            if results["mean_loss"] < (best_val_loss - Config.EARLY_STOPPING_MIN_DELTA):
                best_val_loss = results["mean_loss"]
                no_improvement_epochs = 0

                best_path = f"{Config.CHECKPOINT_DIR}/best_lora"
                model.save_pretrained(best_path)
                log(f"En iyi model guncellendi: {best_path} | val_loss: {best_val_loss:.4f}")
            else:
                no_improvement_epochs += 1
                log(f"Val loss iyilesmedi. Sayac: {no_improvement_epochs}/{Config.EARLY_STOPPING_PATIENCE}")

                if Config.EARLY_STOPPING and no_improvement_epochs >= Config.EARLY_STOPPING_PATIENCE:
                    log("Early stopping tetiklendi. Egitim durduruluyor.")
                    # Eğitim erken bitti — tamamlandı say (tekrar çalıştırınca devam etmesin)
                    save_training_state(Config.CHECKPOINT_DIR, {
                        "completed_epochs": Config.NUM_EPOCHS,
                        "resume_step": 0,
                        "latest_ckpt": f"epoch_{epoch+1}_lora",
                        "last_epoch": Config.NUM_EPOCHS,
                        "best_val_loss": best_val_loss,
                        "no_improvement_epochs": no_improvement_epochs,
                    })
                    _prune_batch_checkpoints(Config.CHECKPOINT_DIR, epoch + 1)
                    break

        # Checkpoint kaydet — sadece LoRA adapter ağırlıklarını kaydet
        # (base model HuggingFace'ten yüklenebilir, sadece öğrenilen kısım kaydedilir)
        adapter_adi = f"epoch_{epoch+1}_lora"
        adapter_path = os.path.join(Config.CHECKPOINT_DIR, adapter_adi)
        model.save_pretrained(adapter_path)
        log(f"LoRA adapter kaydedildi: {adapter_path}")

        # Eğitim durumunu kaydet — kesilirse buradan devam edilir.
        # epoch tamamlandı: completed_epochs = epoch+1, resume_step = 0 (sonraki epoch baştan).
        save_training_state(Config.CHECKPOINT_DIR, {
            "completed_epochs": epoch + 1,
            "resume_step": 0,
            "latest_ckpt": adapter_adi,
            "last_epoch": epoch + 1,
            "best_val_loss": best_val_loss,
            "no_improvement_epochs": no_improvement_epochs,
        })

        # Bu epoch'un ara checkpoint'leri artık gereksiz (epoch sonu checkpoint var)
        _prune_batch_checkpoints(Config.CHECKPOINT_DIR, epoch + 1)

    log(f"\n{'='*50}")
    log("Eğitim tamamlandı!")


def multiclass_train_one_epoch(model, dataloader, optimizer, scheduler, loss_fn, scaler, processor, epoch):
    """
    Multi-class eğitim: her batch'te rastgele bir sınıf seçilir,
    o sınıfın adı text prompt olarak, o sınıfın maskı ground truth olarak kullanılır.

    Örnek: Batch geldi → görselde "Crack" var → "Crack" prompt ile eğit.
    Bir sonraki batch'te "Rust" seç → "Rust" prompt ile eğit.
    Bu sayede model sınıf adlarını öğrenir.

    Args:
        model, dataloader, optimizer, loss_fn, scaler: Standart eğitim parametreleri
        processor: Tokenizer için gerekli
        epoch: Log için epoch numarası

    Returns:
        tuple: (ortalama_loss, ortalama_iou)
    """
    # Tüm sınıf adlarını bir kez tokenize et
    sinif_tokenlar = {}
    for sinif_adi in DACL10K_CLASSES:
        token = processor.tokenizer(
            sinif_adi, return_tensors="pt", padding=True, truncation=True
        )
        sinif_tokenlar[sinif_adi] = {
            "input_ids"     : token["input_ids"].squeeze(0),
            "attention_mask": token["attention_mask"].squeeze(0),
        }

    model.train()
    total_loss = 0.0
    total_iou = 0.0

    for batch_idx, batch in enumerate(dataloader):
        pixel_values      = batch["pixel_values"].to(Config.DEVICE)
        multi_class_mask  = batch["multi_class_mask"].to(Config.DEVICE)  # (B, 19, 288, 288)

        # Bu batch'te hangi sınıflar aktif? (masklarinda piksel var)
        # batch_size=1 olduğu için ilk öğeye bakıyoruz
        aktif_siniflar = [
            i for i in range(len(DACL10K_CLASSES))
            if multi_class_mask[0, i].sum() > 0
        ]

        if not aktif_siniflar:
            # Bu görselde hiç annotation yok — binary mask ile devam et
            aktif_siniflar = list(range(len(DACL10K_CLASSES)))
            gt_mask = batch["ground_truth_mask"].to(Config.DEVICE)
            sinif_adi = "damage"
            input_ids     = batch["input_ids"].to(Config.DEVICE)
            attention_mask = batch["attention_mask"].to(Config.DEVICE)
        else:
            # Aktif sınıflardan birini rastgele seç
            sinif_idx  = random.choice(aktif_siniflar)
            sinif_adi  = DACL10K_CLASSES[sinif_idx]
            gt_mask    = multi_class_mask[:, sinif_idx, :, :]  # (B, 288, 288)
            tokenlar   = sinif_tokenlar[sinif_adi]
            input_ids      = tokenlar["input_ids"].unsqueeze(0).to(Config.DEVICE)
            attention_mask = tokenlar["attention_mask"].unsqueeze(0).to(Config.DEVICE)

        with torch.amp.autocast("cuda", enabled=(Config.DEVICE == "cuda")):
            outputs   = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            pred_mask = outputs.semantic_seg.squeeze(1)
            loss      = loss_fn(pred_mask, gt_mask)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        iou        = calculate_iou(pred_mask.detach(), gt_mask)
        total_loss += loss.item()
        total_iou  += iou

        if (batch_idx + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            log(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} "
                f"| Sinif: {sinif_adi:<15} | Loss: {loss.item():.4f} | IoU: {iou:.4f} "
                f"| LR: {current_lr:.6e}")

        if (
            Config.CHECKPOINT_EVERY_STEPS > 0
            and (batch_idx + 1) % Config.CHECKPOINT_EVERY_STEPS == 0
        ):
            ara_kayit_yolu = (
                f"{Config.CHECKPOINT_DIR}/mc_epoch_{epoch+1}_batch_{batch_idx+1}_lora"
            )
            model.save_pretrained(ara_kayit_yolu)
            log(f"Ara checkpoint kaydedildi: {ara_kayit_yolu}")

    avg_loss = total_loss / max(len(dataloader), 1)
    avg_iou  = total_iou  / max(len(dataloader), 1)
    return avg_loss, avg_iou


def train_multiclass(model, train_dataloader, val_dataloader, processor):
    """
    Multi-class eğitim sürecini yönetir.
    Her batch'te rastgele bir hasar sınıfı seçilerek eğitim yapılır.

    Args:
        model:             LoRA uygulanmış model
        train_dataloader:  Eğitim verisi
        val_dataloader:    Validation verisi
        processor:         Tokenizer için SAM3 Processor
    """
    from src.evaluate import evaluate

    log("Multi-class eğitim başlıyor...")
    log(f"Epoch sayısı  : {Config.NUM_EPOCHS}")
    log(f"Sinif sayisi  : {len(DACL10K_CLASSES)}")
    log(f"Cihaz         : {Config.DEVICE}")
    log(f"Loss tipi     : {Config.LOSS_TYPE}")
    log(f"LR scheduler  : {Config.LR_SCHEDULER}")
    log(f"Early stopping: {Config.EARLY_STOPPING}")

    ensure_dir(Config.CHECKPOINT_DIR)

    loss_fn   = build_loss_fn()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE,
    )
    scheduler = _create_scheduler(optimizer, len(train_dataloader))
    scaler = torch.amp.GradScaler("cuda", enabled=(Config.DEVICE == "cuda"))

    best_val_loss = float("inf")
    no_improvement_epochs = 0

    for epoch in range(Config.NUM_EPOCHS):
        log(f"\n{'='*50}")
        log(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} başladı")

        train_loss, train_iou = multiclass_train_one_epoch(
            model, train_dataloader, optimizer, scheduler, loss_fn, scaler, processor, epoch
        )
        log(f"Epoch {epoch+1} [train] | Loss: {train_loss:.4f} | IoU: {train_iou:.4f}")

        if val_dataloader is not None:
            results = evaluate(model, val_dataloader, loss_fn)
            log(f"Epoch {epoch+1} [val]   | Loss: {results['mean_loss']:.4f} "
                f"| IoU: {results['mean_iou']:.4f}")

            if results["mean_loss"] < (best_val_loss - Config.EARLY_STOPPING_MIN_DELTA):
                best_val_loss = results["mean_loss"]
                no_improvement_epochs = 0

                best_path = f"{Config.CHECKPOINT_DIR}/mc_best_lora"
                model.save_pretrained(best_path)
                log(f"En iyi model guncellendi: {best_path} | val_loss: {best_val_loss:.4f}")
            else:
                no_improvement_epochs += 1
                log(f"Val loss iyilesmedi. Sayac: {no_improvement_epochs}/{Config.EARLY_STOPPING_PATIENCE}")

                if Config.EARLY_STOPPING and no_improvement_epochs >= Config.EARLY_STOPPING_PATIENCE:
                    log("Early stopping tetiklendi. Multi-class egitim durduruluyor.")
                    break

        adapter_path = f"{Config.CHECKPOINT_DIR}/mc_epoch_{epoch+1}_lora"
        model.save_pretrained(adapter_path)
        log(f"LoRA adapter kaydedildi: {adapter_path}")

    log(f"\n{'='*50}")
    log("Multi-class eğitim tamamlandı!")
