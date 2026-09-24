"""Hafta 2: Test videosu uzerinde YOLOv8 tespitlerini cizip yeni bir video olarak kaydeder.

Video sunumunda/kayidinda gostermek icin - ham video yerine uzerinde tespit
kutucuklari cizilmis hali daha etkileyici olur.

Kullanim:
    python src/annotate_video.py
"""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

MODEL_PATH = MODELS_DIR / "yolov8n.pt"
IMG_SIZE = 640


def main():
    video_dir = DATA_DIR / "test_videos"
    videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    if not videos:
        print("data/test_videos/ altinda video bulunamadi.")
        return
    video_path = videos[0]

    print(f"Video: {video_path.name}")
    model = YOLO(str(MODEL_PATH))

    RESULTS_DIR.mkdir(exist_ok=True)
    model.predict(
        source=str(video_path),
        imgsz=IMG_SIZE,
        device="cuda:0",
        save=True,
        project=str(RESULTS_DIR),
        name="annotated_video",
        exist_ok=True,
        verbose=False,
    )
    print(f"Tamamlandi -> results/annotated_video/ klasoru icinde")


if __name__ == "__main__":
    main()
