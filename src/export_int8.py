"""Hafta 3: YOLOv8n modelini INT8 TensorRT engine'ine cevirir (kalibrasyon veri setiyle).

On kosul: src/download_coco_val.py calistirilmis olmali (kalibrasyon icin COCO val2017
gorselleri kullaniliyor).

Kullanim:
    python src/export_int8.py
"""
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_YAML = ROOT / "data" / "coco_val.yaml"
MODEL_NAME = "yolov8n.pt"
IMG_SIZE = 640


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"{DATA_YAML} yok, once download_coco_val.py calistir.")

    # Turkce karakterli yol sorunu (bkz. export_models.py) yine gecerli - ASCII
    # uyumlu gecici klasorde calisip sonucu projeye geri tasiyoruz.
    tmp_dir = Path(tempfile.gettempdir()) / "yolo_export_tmp"
    tmp_dir.mkdir(exist_ok=True)
    tmp_weights = tmp_dir / MODEL_NAME
    shutil.copy(MODELS_DIR / MODEL_NAME, tmp_weights)

    print("INT8 TensorRT engine olusturuluyor (kalibrasyon icin COCO val2017 kullanilacak, "
          "birkac dakika surebilir)...")
    model = YOLO(str(tmp_weights))
    engine_path = Path(model.export(format="engine", imgsz=IMG_SIZE, int8=True,
                                     data=str(DATA_YAML), fraction=0.2))

    dest = MODELS_DIR / "yolov8n_int8.engine"
    shutil.move(str(engine_path), str(dest))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Tamamlandi -> {dest}")


if __name__ == "__main__":
    main()
