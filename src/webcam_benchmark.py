"""Hafta 1: Webcam uzerinde canli YOLOv8 tespiti + uctan uca FPS olcumu.

Kullanim:
    python src/webcam_benchmark.py

Ne yapar:
  - Webcam'i acar, her karede YOLOv8n ile tespit yapar
  - Ekranda tespit kutucuklarini ve anlik FPS'i canli gosterir (video kaydi icin idealdir)
  - 'q' tusuna basana kadar veya suresi dolana kadar calisir
  - Ortalama uctan uca FPS'i (okuma + inference + cizim dahil) hesaplayip kaydeder
"""
import json
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_NAME = "yolov8n.pt"
IMG_SIZE = 640
MAX_SECONDS = 25  # otomatik durma suresi (video kaydi icin makul bir sure)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device}")

    model = YOLO(str(MODELS_DIR / MODEL_NAME))
    model.to(device)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("HATA: Webcam acilamadi.")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    sample_saved = False
    frame_times = []
    start_time = time.perf_counter()

    print("Canli tespit basliyor. Cikmak icin 'q' tusuna bas "
          f"(en fazla {MAX_SECONDS} saniye calisacak)...")

    while True:
        loop_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, imgsz=IMG_SIZE, device=device, verbose=False)
        annotated = results[0].plot()

        loop_end = time.perf_counter()
        instant_fps = 1.0 / (loop_end - loop_start)
        frame_times.append(loop_end - loop_start)

        cv2.putText(annotated, f"FPS: {instant_fps:.1f}", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(annotated, f"Device: {device}", (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Hafta 1 - YOLOv8 Canli Tespit (q ile cik)", annotated)

        if not sample_saved and len(frame_times) > 10:
            cv2.imwrite(str(RESULTS_DIR / "week1_webcam_sample_frame.jpg"), annotated)
            sample_saved = True

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if time.perf_counter() - start_time > MAX_SECONDS:
            print("Sure doldu, durduruluyor.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if frame_times:
        avg_fps = len(frame_times) / sum(frame_times)
        avg_latency_ms = (sum(frame_times) / len(frame_times)) * 1000
        summary = {
            "week": 1,
            "test_type": "webcam_end_to_end",
            "device": device,
            "frame_count": len(frame_times),
            "avg_end_to_end_fps": round(avg_fps, 2),
            "avg_latency_ms": round(avg_latency_ms, 2),
        }
        out_json = RESULTS_DIR / "week1_webcam_benchmark.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nOrtalama uctan uca FPS: {avg_fps:.2f} ({len(frame_times)} kare)")
        print(f"Sonuclar kaydedildi: {out_json.relative_to(ROOT)}")
    else:
        print("Hic kare islenemedi.")


if __name__ == "__main__":
    main()
