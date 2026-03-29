"""
model.py — SAM3 Model Yükleme

Bu dosya SAM3 modelini ve processor'ı yükler.

SAM3 nedir?
Meta'nın geliştirdiği "Segment Anything Model 3" — görsellerdeki nesneleri
metin ipuçlarıyla segmente edebilen bir modeldir.

ÖNEMLİ — SAM3 nasıl çalışır?
SAM3'e bir görsel VE bir metin veriyoruz. Metin, ne aradığımızı söyler.
Örnek:
    processor(images=image, text="crack", return_tensors="pt")
    → model, görseldeki çatlakları bulur ve mask döndürür.

Bizim projemizde metin olarak DACL10K sınıf adlarını kullanacağız:
    "Crack", "Rust", "Spalling" gibi...

Model çıktısı (outputs):
    - outputs.masks  → bulunan nesnelerin segmentation maskleri
    - outputs.scores → her mask için güven skoru (0-1 arası)
    - outputs.boxes  → bulunan nesnelerin sınırlayıcı kutuları

Sonuçları görsel boyutuna getirmek için:
    processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist()
    )

Processor nedir?
Model görseli doğrudan anlayamaz. Processor, görseli modelin anlayabileceği
sayısal formata (tensor) dönüştürür. Yani bir "tercüman" gibidir.

Bu dosyada iki fonksiyon var:
- load_model()     → modeli yükler ve doğru cihaza (GPU/CPU) taşır
- load_processor() → görselleri modele uygun hale getiren processor'ı yükler
"""

from transformers import Sam3Model, Sam3Processor

from src.config import Config
from src.utils import log


def load_model():
    """
    SAM3 modelini HuggingFace'ten yükler.

    Kullanım için HuggingFace hesabına giriş yapmış olman gerekir:
        huggingface-cli login

    Returns:
        model: Yüklenmiş SAM3 modeli (doğru cihaza yerleştirilmiş)
    """
    log(f"Model yükleniyor: {Config.MODEL_NAME}")

    # Modeli HuggingFace'ten indir ve yükle
    model = Sam3Model.from_pretrained(Config.MODEL_NAME)

    # Modeli doğru cihaza taşı (GPU varsa GPU, yoksa CPU)
    model = model.to(Config.DEVICE)

    log(f"Model yüklendi. Cihaz: {Config.DEVICE}")
    return model


def load_processor():
    """
    SAM3 Processor'ı yükler.

    Processor ne yapar?
    - Görselleri modelin beklediği boyuta getirir
    - Piksel değerlerini normalize eder
    - Metin ipuçlarını (ör: "crack") sayısal formata çevirir
    - Tensor formatına çevirir

    Returns:
        processor: SAM3 Processor
    """
    log(f"Processor yükleniyor: {Config.MODEL_NAME}")

    processor = Sam3Processor.from_pretrained(Config.MODEL_NAME)

    log("Processor yüklendi.")
    return processor
