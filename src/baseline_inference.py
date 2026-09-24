"""Hafta 1: Baseline YOLOv8 inference + FPS olcumu (PyTorch, optimizasyonsuz).

Kullanim:
    python src/baseline_inference.py

Ne yapar:
  1) YOLOv8n pretrained agirliklarini indirir/yukler (models/yolov8n.pt)
  2) Ornek bir gorsel uzerinde tespit calistirip sonucu results/ altina kaydeder
  3) Saf model inference hizini (FPS, ortalama gecikme) olcer
  4) data/test_videos/ altinda video varsa, uctan uca (okuma+inference+cizim) FPS olcer
  5) Tum sonuclari results/week1_baseline_results.json dosyasina yazar
"""
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"

MODEL_NAME = "yolov8n.pt"
IMG_SIZE = 640
WARMUP_ITERS = 10
BENCH_ITERS = 100


def get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model(device: str) -> YOLO:
    MODELS_DIR.mkdir(exist_ok=True)
    weights_path = MODELS_DIR / MODEL_NAME
    model = YOLO(str(weights_path))
    model.to(device)
    return model


def run_sample_detection(model: YOLO, device: str) -> dict:
    """Ornek gorsel uzerinde tek seferlik dogrulama tespiti."""
    sample_url = "https://ultralytics.com/images/bus.jpg"
    results = model.predict(source=sample_url, imgsz=IMG_SIZE, device=device, verbose=False)
    r = results[0]
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "week1_sample_detection.jpg"
    r.save(filename=str(out_path))
    return {
        "sample_image": sample_url,
        "num_detections": len(r.boxes),
        "output_saved_to": str(out_path.relative_to(ROOT)),
    }


def benchmark_pure_inference(model: YOLO, device: str) -> dict:
    """Sabit boyutlu tensor ile saf model inference hizi (I/O haric)."""
    dummy = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=device)

    # Isinma (warmup)
    for _ in range(WARMUP_ITERS):
        _ = model.predict(source=dummy, imgsz=IMG_SIZE, device=device, verbose=False)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(BENCH_ITERS):
        _ = model.predict(source=dummy, imgsz=IMG_SIZE, device=device, verbose=False)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / BENCH_ITERS) * 1000
    fps = BENCH_ITERS / elapsed

    return {
        "iterations": BENCH_ITERS,
        "imgsz": IMG_SIZE,
        "total_time_s": round(elapsed, 4),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "fps": round(fps, 2),
    }


def benchmark_video_if_present(model: YOLO, device: str) -> dict | None:
    video_dir = DATA_DIR / "test_videos"
    if not video_dir.exists():
        return None
    videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    if not videos:
        return None

    video_path = videos[0]
    start = time.perf_counter()
    frame_count = 0
    results_gen = model.predict(source=str(video_path), imgsz=IMG_SIZE, device=device,
                                 stream=True, verbose=False)
    for _ in results_gen:
        frame_count += 1
    elapsed = time.perf_counter() - start

    return {
        "video_file": video_path.name,
        "frame_count": frame_count,
        "total_time_s": round(elapsed, 4),
        "end_to_end_fps": round(frame_count / elapsed, 2) if elapsed > 0 else None,
    }


def main():
    device = get_device()
    print(f"Cihaz: {device}")
    if device == "cpu":
        print("UYARI: CUDA bulunamadi, CPU uzerinde calisiliyor (yavas olacaktir).")

    print(f"Model yukleniyor: {MODEL_NAME}")
    model = load_model(device)

    print("\n[1/3] Ornek gorsel uzerinde tespit calistiriliyor...")
    sample_result = run_sample_detection(model, device)
    print(f"  -> {sample_result['num_detections']} nesne tespit edildi, "
          f"kayit: {sample_result['output_saved_to']}")

    print("\n[2/3] Saf inference FPS olcumu (dummy tensor, "
          f"{WARMUP_ITERS} warmup + {BENCH_ITERS} olcum)...")
    pure_bench = benchmark_pure_inference(model, device)
    print(f"  -> FPS: {pure_bench['fps']}  |  Ortalama gecikme: {pure_bench['avg_latency_ms']} ms")

    print("\n[3/3] data/test_videos/ altinda video araniyor...")
    video_bench = benchmark_video_if_present(model, device)
    if video_bench:
        print(f"  -> {video_bench['video_file']}: {video_bench['end_to_end_fps']} FPS "
              f"({video_bench['frame_count']} kare)")
    else:
        print("  -> Video bulunamadi, bu adim atlandi. "
              "(data/test_videos/ icine bir .mp4 koyup tekrar calistirabilirsin)")

    summary = {
        "week": 1,
        "model": MODEL_NAME,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else "CPU",
        "precision": "FP32",
        "sample_detection": sample_result,
        "pure_inference_benchmark": pure_bench,
        "video_benchmark": video_bench,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_json = RESULTS_DIR / "week1_baseline_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSonuclar kaydedildi: {out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
