"""Prepare YOLOv8n artifacts for a fresh clone.

Examples:
    python src/prepare_models.py --base
    python src/prepare_models.py --engines
    python src/prepare_models.py --download-coco --int8
    python src/prepare_models.py --status

TensorRT engine files are device specific. Always build them on the NVIDIA GPU
that will run the application.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
WEIGHTS = MODELS_DIR / "yolov8n.pt"
ONNX_FILE = MODELS_DIR / "yolov8n.onnx"


def run_script(name: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "src" / name)], cwd=ROOT, check=True)


def ensure_base_weights() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if WEIGHTS.exists():
        print(f"OK: {WEIGHTS.relative_to(ROOT)} zaten mevcut.")
        return

    print("YOLOv8n agirliklari Ultralytics kaynagindan indiriliyor...")
    with tempfile.TemporaryDirectory(prefix="tensorvision_weights_") as temp_dir:
        temp_path = Path(temp_dir)
        model = YOLO(str(temp_path / "yolov8n.pt"))
        downloaded = Path(model.ckpt_path)
        if not downloaded.exists():
            raise RuntimeError("Ultralytics model agirligini indirdi ancak dosya bulunamadi.")
        shutil.copy2(downloaded, WEIGHTS)
    print(f"Hazir: {WEIGHTS.relative_to(ROOT)}")


def ensure_onnx() -> None:
    ensure_base_weights()
    if ONNX_FILE.exists():
        print(f"OK: {ONNX_FILE.relative_to(ROOT)} zaten mevcut.")
        return

    print("YOLOv8n ONNX bicimine donusturuluyor...")
    with tempfile.TemporaryDirectory(prefix="tensorvision_onnx_") as temp_dir:
        temp_path = Path(temp_dir)
        temp_weights = temp_path / "yolov8n.pt"
        shutil.copy2(WEIGHTS, temp_weights)
        exported = Path(YOLO(str(temp_weights)).export(format="onnx", imgsz=640))
        shutil.copy2(exported, ONNX_FILE)
    print(f"Hazir: {ONNX_FILE.relative_to(ROOT)}")


def build_fp_engines() -> None:
    ensure_onnx()
    run_script("build_fp32_engine.py")
    run_script("build_fp16_engine.py")


def build_int8(download_coco: bool) -> None:
    ensure_base_weights()
    coco_images = ROOT / "data" / "coco_val" / "images" / "val2017"
    if not coco_images.exists() and download_coco:
        run_script("download_coco_val.py")
    if not coco_images.exists():
        raise RuntimeError(
            "INT8 kalibrasyonu icin COCO val2017 bulunamadi. "
            "Komutu --download-coco --int8 secenekleriyle calistirin."
        )
    run_script("export_int8.py")


def print_status() -> None:
    artifacts = (
        "yolov8n.pt",
        "yolov8n.onnx",
        "yolov8n_fp32.engine",
        "yolov8n_fp16.engine",
        "yolov8n_int8.engine",
    )
    for name in artifacts:
        path = MODELS_DIR / name
        state = f"{path.stat().st_size / 1024 / 1024:.2f} MB" if path.exists() else "YOK"
        print(f"{name:24} {state}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", action="store_true", help="Yalnizca PyTorch agirligini hazirla.")
    parser.add_argument("--engines", action="store_true", help="FP32 ve FP16 TensorRT engine'lerini olustur.")
    parser.add_argument("--int8", action="store_true", help="INT8 TensorRT engine'ini olustur.")
    parser.add_argument("--download-coco", action="store_true", help="INT8 icin COCO val2017'yi indir (~1 GB).")
    parser.add_argument("--status", action="store_true", help="Model dosyalarinin durumunu goster.")
    args = parser.parse_args()

    if args.status:
        print_status()
        return
    if not any((args.base, args.engines, args.int8)):
        args.base = True
    if args.base:
        ensure_base_weights()
    if args.engines:
        build_fp_engines()
    if args.int8:
        build_int8(args.download_coco)
    print_status()


if __name__ == "__main__":
    main()
