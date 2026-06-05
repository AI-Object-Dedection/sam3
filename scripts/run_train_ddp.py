"""
run_train_ddp.py — DDP ile 2 GPU Eğitim (Kaggle)

Bu script, eğitim sürecini Distributed Data Parallel (DDP) ile çalıştırır.
Kaggle'da 2x T4 GPU kullanırken önerilen yöntem budur.

Kullanım:
    torchrun --nproc_per_node=2 scripts/run_train_ddp.py
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import Sam3Model

from src.config import Config
from src.dataset import DACL10KDataset
from src.evaluate import calculate_iou
from src.lora import apply_lora
from src.losses import build_loss_fn
from src.model import load_processor
from src.utils import ensure_dir, log, set_seed


def create_scheduler(optimizer, steps_per_epoch):
    """Config'e göre scheduler üretir."""
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
            return Config.MIN_LR_RATIO + (1.0 - Config.MIN_LR_RATIO) * cosine_decay

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


def reduce_mean(value, device):
    """Tüm process'lerdeki skoru ortalar."""
    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / dist.get_world_size()
    return tensor.item()


def sync_epoch_stats(total_loss, total_iou, total_steps, device):
    """Epoch istatistiklerini tüm rank'lerden toplayıp global ortalama üretir."""
    stats = torch.tensor([total_loss, total_iou, float(total_steps)], dtype=torch.float32, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    global_steps = max(int(stats[2].item()), 1)
    mean_loss = stats[0].item() / global_steps
    mean_iou = stats[1].item() / global_steps
    return mean_loss, mean_iou


def evaluate_ddp(model, dataloader, loss_fn, device):
    """Validation metriklerini DDP'de global olarak hesaplar."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            gt_mask = batch["ground_truth_mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=True):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                pred_mask = outputs.semantic_seg.squeeze(1)
                loss = loss_fn(pred_mask, gt_mask)

            iou = calculate_iou(pred_mask, gt_mask)
            total_loss += loss.item()
            total_iou += iou
            total_steps += 1

    return sync_epoch_stats(total_loss, total_iou, total_steps, device)


def train_one_epoch_ddp(model, dataloader, optimizer, scheduler, loss_fn, scaler, device, epoch, rank):
    """Tek epoch DDP eğitim."""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_steps = 0

    for batch_idx, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        gt_mask = batch["ground_truth_mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=True):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            pred_mask = outputs.semantic_seg.squeeze(1)
            loss = loss_fn(pred_mask, gt_mask)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        iou = calculate_iou(pred_mask.detach(), gt_mask)
        total_loss += loss.item()
        total_iou += iou
        total_steps += 1

        if rank == 0 and (batch_idx + 1) % 20 == 0:
            lr = optimizer.param_groups[0]["lr"]
            log(
                f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} "
                f"| Loss: {loss.item():.4f} | IoU: {iou:.4f} | LR: {lr:.6e}"
            )

    return sync_epoch_stats(total_loss, total_iou, total_steps, device)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("DDP egitimi icin CUDA gerekli.")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend=Config.DDP_BACKEND)

    set_seed(42 + rank)

    if rank == 0:
        log("=" * 60)
        log("SAM3 Fine-Tuning — DDP Egitim Basliyor")
        log(f"GPU sayisi (world_size): {world_size}")
        log(f"Loss: {Config.LOSS_TYPE} | Scheduler: {Config.LR_SCHEDULER}")
        log(f"Augmentasyon: {Config.USE_AUGMENTATION}")
        log("=" * 60)

    processor = load_processor()

    # Her process modeli kendi GPU'suna kurar
    base_model = Sam3Model.from_pretrained(
        Config.MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=os.path.isdir(Config.MODEL_NAME),
    )
    base_model = base_model.to(device)
    model = apply_lora(base_model)

    # DDP wrapper
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=Config.DDP_FIND_UNUSED_PARAMETERS,
    )

    train_dataset = DACL10KDataset(
        images_dir=Config.TRAIN_IMAGES,
        annotations_dir=Config.TRAIN_MASKS,
        processor=processor,
        max_samples=Config.MAX_TRAIN_SAMPLES,
        is_train=True,
        use_augmentation=Config.USE_AUGMENTATION,
    )

    val_dataset = DACL10KDataset(
        images_dir=Config.VAL_IMAGES,
        annotations_dir=Config.VAL_MASKS,
        processor=processor,
        max_samples=Config.MAX_VAL_SAMPLES,
        is_train=False,
        use_augmentation=False,
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=Config.DDP_NUM_WORKERS,
        persistent_workers=(Config.DDP_NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=Config.DDP_NUM_WORKERS,
        persistent_workers=(Config.DDP_NUM_WORKERS > 0),
    )

    loss_fn = build_loss_fn()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LEARNING_RATE,
    )
    scheduler = create_scheduler(optimizer, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    if rank == 0:
        ensure_dir(Config.CHECKPOINT_DIR)

    dist.barrier()

    best_val_loss = float("inf")
    no_improvement_epochs = 0

    for epoch in range(Config.NUM_EPOCHS):
        train_sampler.set_epoch(epoch)

        if rank == 0:
            log(f"\n{'='*50}")
            log(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} basladi")

        train_loss, train_iou = train_one_epoch_ddp(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            epoch=epoch,
            rank=rank,
        )

        val_loss, val_iou = evaluate_ddp(model, val_loader, loss_fn, device)

        if rank == 0:
            log(f"Epoch {epoch+1} [train] | Loss: {train_loss:.4f} | IoU: {train_iou:.4f}")
            log(f"Epoch {epoch+1} [val]   | Loss: {val_loss:.4f} | IoU: {val_iou:.4f}")

            epoch_path = f"{Config.CHECKPOINT_DIR}/ddp_epoch_{epoch+1}_lora"
            model.module.save_pretrained(epoch_path)
            log(f"LoRA adapter kaydedildi: {epoch_path}")

            if val_loss < (best_val_loss - Config.EARLY_STOPPING_MIN_DELTA):
                best_val_loss = val_loss
                no_improvement_epochs = 0

                best_path = f"{Config.CHECKPOINT_DIR}/ddp_best_lora"
                model.module.save_pretrained(best_path)
                log(f"En iyi model guncellendi: {best_path} | val_loss: {best_val_loss:.4f}")
            else:
                no_improvement_epochs += 1
                log(f"Val loss iyilesmedi. Sayac: {no_improvement_epochs}/{Config.EARLY_STOPPING_PATIENCE}")

        stop_flag = torch.tensor(0, device=device, dtype=torch.int32)
        if rank == 0 and Config.EARLY_STOPPING and no_improvement_epochs >= Config.EARLY_STOPPING_PATIENCE:
            stop_flag.fill_(1)
            log("Early stopping tetiklendi. DDP egitimi durduruluyor.")

        dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            break

    if rank == 0:
        log(f"\n{'='*50}")
        log("DDP egitimi tamamlandi!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
