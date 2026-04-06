# Faz 4 — Tam Eğitim Raporu

**Tarih:** 2026-04-05 (gece) — 2026-04-06 (akşam)
**Süre:** 21:25 — 19:08 (~21.5 saat)

---

## Özet

SAM3 modeli DACL10K köprü hasarı veri seti üzerinde LoRA ile **tam eğitim** başarıyla tamamlandı.
6935 eğitim görseli, 975 validation görseli, 10 epoch.

---

## Eğitim Konfigürasyonu

| Parametre | Değer |
|-----------|-------|
| Model | `facebook/sam3` (842M parametre) |
| Eğitilebilir parametre | 2.138.112 (%0.25 — LoRA) |
| Eğitim görseli | 6.935 (tüm dataset) |
| Validation görseli | 975 (tüm dataset) |
| Epoch sayısı | 10 |
| Batch size | 1 |
| Learning rate | 0.0001 |
| Loss fonksiyonu | BCEWithLogitsLoss |
| Text prompt | "damage" |
| Mask boyutu | 288×288 |
| GPU | NVIDIA RTX 4090 Laptop (17.2 GB VRAM) |

---

## Sonuçlar

### Eğitim (Train) Metrikleri

| Epoch | Loss | IoU |
|-------|------|-----|
| 1 | 0.4287 | 0.4874 |
| 2 | 0.3480 | 0.5565 |
| 3 | 0.3078 | 0.5844 |
| 4 | 0.2773 | 0.6040 |
| 5 | 0.2502 | 0.6282 |
| 6 | 0.2283 | 0.6451 |
| 7 | 0.2058 | 0.6621 |
| 8 | 0.1933 | 0.6759 |
| 9 | 0.1797 | 0.6874 |
| **10** | **0.1743** | **0.6949** |

### Validation Metrikleri

| Epoch | Loss | IoU |
|-------|------|-----|
| 1 | 0.3653 | 0.5494 |
| 2 | 0.3516 | 0.5684 |
| 3 | 0.3529 | 0.5673 |
| 4 | 0.3658 | 0.5798 |
| **5** | 0.3698 | **0.5708** ← en iyi val IoU civarı |
| 6 | 0.3735 | 0.5624 |
| 7 | 0.3874 | 0.5472 |
| 8 | 0.4017 | 0.5726 |
| 9 | 0.4244 | 0.5692 |
| 10 | 0.4320 | 0.5614 |

---

## Değerlendirme

### Olumlu Bulgular

- **Train loss sürekli düştü:** 0.43 → 0.17 — model her epoch'ta öğrendi
- **Train IoU sürekli arttı:** 0.49 → 0.69 — hasar tespiti giderek iyileşti
- **Val IoU ~0.55-0.58 bandında sabit:** Model gerçek görsellerden daha önce görmediği köprülerdeki hasarları biliyor

### Dikkat Edilecek Nokta — Hafif Overfitting

Epoch 5'ten sonra val loss artmaya başladı (0.37 → 0.43) ama val IoU stabil kaldı.
Bu hafif bir overfitting işareti. En iyi model **epoch 4 veya 5**.

```
En iyi checkpoint: checkpoints/epoch_4_lora  (val IoU: 0.5798)
```

### IoU Değerlendirme Skalası

| IoU Değeri | Anlam |
|-----------|-------|
| 1.0 | Mükemmel |
| 0.7+ | İyi |
| **0.58** | **Orta-İyi — ilk fine-tuning için başarılı** |
| 0.3 altı | Zayıf |

---

## Checkpoint Konumu

```
checkpoints/
├── epoch_1_lora/
├── epoch_2_lora/
├── epoch_3_lora/
├── epoch_4_lora/   ← Önerilen (en iyi val IoU)
├── epoch_5_lora/
├── epoch_6_lora/
├── epoch_7_lora/
├── epoch_8_lora/
├── epoch_9_lora/
└── epoch_10_lora/
```

---

## Sonraki Adım — Faz 5: Inference + Demo

Eğitim tamamlandı. Sırada:
1. `epoch_4_lora` checkpoint'ini yükle
2. Gerçek bir köprü fotoğrafı ver
3. Modelin hasar maskini görselleştir

Komut: `python scripts/run_infer.py`
