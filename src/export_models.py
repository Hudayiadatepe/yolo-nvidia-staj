"""Hafta 2: YOLOv8n modelini ONNX ve TensorRT (FP32/FP16) formatlarina cevirir.

Kullanim:
    python src/export_models.py

Ne yapar:
  - models/yolov8n.pt -> models/yolov8n.onnx
  - models/yolov8n.pt -> models/yolov8n.engine (TensorRT, FP32)
  - models/yolov8n.pt -> models/yolov8n_fp16.engine (TensorRT, FP16)

Not: TensorRT export, GPU'ya ve o an kurulu TensorRT surumune ozel bir "engine"
dosyasi uretir - baska bir makineye tasinamaz, her makinede ayri export gerekir.
"""
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_NAME = "yolov8n.pt"
IMG_SIZE = 640


def main():
    # NOT: Proje yolu Turkce karakter icerdigi icin ("Yeni klasor" -> "ö"),
    # TensorRT'nin C++ katmani dosyayi bulamiyor (Windows'a ozgu bir sorun).
    # Bu yuzden export islemini ASCII-uyumlu gecici bir klasorde yapip
    # sonuclari projeye geri tasiyoruz.
    tmp_dir = Path(tempfile.gettempdir()) / "yolo_export_tmp"
    tmp_dir.mkdir(exist_ok=True)
    tmp_weights = tmp_dir / MODEL_NAME
    shutil.copy(MODELS_DIR / MODEL_NAME, tmp_weights)
    print(f"Gecici calisma klasoru (ASCII-uyumlu): {tmp_dir}")

    print("\n[1/3] ONNX'e donusturuluyor...")
    model = YOLO(str(tmp_weights))
    onnx_path = Path(model.export(format="onnx", imgsz=IMG_SIZE))
    onnx_dest = MODELS_DIR / "yolov8n.onnx"
    shutil.move(str(onnx_path), str(onnx_dest))
    print(f"  -> {onnx_dest}")

    # TensorRT export her zaman ayni dosya adini (yolov8n.engine) uretir,
    # bu yuzden FP32 ve FP16 icin ayri isimlere tasiyoruz.
    print("\n[2/3] TensorRT FP32 engine olusturuluyor (bu birkac dakika surebilir)...")
    model = YOLO(str(tmp_weights))
    engine_fp32 = Path(model.export(format="engine", imgsz=IMG_SIZE, half=False))
    engine_fp32_dest = MODELS_DIR / "yolov8n_fp32.engine"
    shutil.move(str(engine_fp32), str(engine_fp32_dest))
    print(f"  -> {engine_fp32_dest}")

    print("\n[3/3] TensorRT FP16 engine olusturuluyor (bu birkac dakika surebilir)...")
    model = YOLO(str(tmp_weights))
    engine_fp16 = Path(model.export(format="engine", imgsz=IMG_SIZE, half=True))
    engine_fp16_dest = MODELS_DIR / "yolov8n_fp16.engine"
    shutil.move(str(engine_fp16), str(engine_fp16_dest))
    print(f"  -> {engine_fp16_dest}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\nTum donusumler tamamlandi.")


if __name__ == "__main__":
    main()
