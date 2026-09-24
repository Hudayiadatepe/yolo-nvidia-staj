"""Hafta 2: PyTorch FP32 vs TensorRT FP32 vs TensorRT FP16 FPS/gecikme kiyaslamasi.

Kullanim:
    python src/compare_precisions.py

On kosul: src/export_models.py calistirilmis olmali (ONNX + TensorRT engine'leri
models/ altinda hazir olmali).

Ne yapar:
  - Her model varyanti icin REPEATS kez ayri ayri olcum yapar (saf hiz + gercek
    video uzerinde uctan uca), ortalama ve standart sapma raporlar. Tek olcume
    guvenmek yaniltici olabiliyor (GPU isinmasi, arka plan yuku vb. yuzunden
    calistirmadan calistirmaya %20'ye varan fark gorulebiliyor), bu yuzden
    tekrarli olcum + std sapma her zaman raporlaniyor.
  - Sonuclari kiyaslayan bir tablo + hata cubuklu bar chart uretir
"""
import json
import hashlib
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import ultralytics
from ultralytics import YOLO

try:
    import tensorrt as trt
except ImportError:
    trt = None

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"

IMG_SIZE = 640
WARMUP_ITERS = 10
BENCH_ITERS = 100
REPEATS = 10  # her format kac kez ayri ayri olculecek
VIDEO_LOOPS = 3  # kisa videolarda olcumu daha az gurultulu yapmak icin video birkac kez ust uste izleniyor

# Cikti dosyalarinin onune eklenecek etiket - hangi haftanin sonucu oldugunu belli
# etmek icin (farkli haftalarda farkli VARIANTS listesiyle calistirilinca birbirinin
# ustune yazmasin diye)
OUTPUT_PREFIX = "publication"

VARIANTS = [
    ("PyTorch FP32", MODELS_DIR / "yolov8n.pt"),
    ("TensorRT FP32", MODELS_DIR / "yolov8n_fp32.engine"),
    ("TensorRT FP16", MODELS_DIR / "yolov8n_fp16.engine"),
    ("TensorRT INT8", MODELS_DIR / "yolov8n_int8.engine"),
]


def run_pure_once(model: YOLO, device: str) -> float:
    """Tek bir olcum turu, saf hiz (FPS) dondurur."""
    dummy = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=device)

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
    return BENCH_ITERS / elapsed


def run_video_once(model: YOLO, device: str, video_path: Path) -> float:
    """Tek bir video turu, uctan uca FPS dondurur.

    Video kisa oldugu icin tek geciste olcum cok gurultulu oluyor;
    VIDEO_LOOPS kadar ust uste isleyip kare sayisini artiriyoruz.
    """
    start = time.perf_counter()
    frame_count = 0
    for _ in range(VIDEO_LOOPS):
        for _ in model.predict(source=str(video_path), imgsz=IMG_SIZE, device=device,
                                stream=True, verbose=False):
            frame_count += 1
    elapsed = time.perf_counter() - start
    return frame_count / elapsed if elapsed > 0 else 0.0


def mean_std(values: list[float]) -> dict:
    return {
        "mean": round(statistics.mean(values), 2),
        "std": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
        "runs": [round(v, 2) for v in values],
    }


def find_test_video() -> Path | None:
    video_dir = DATA_DIR / "test_videos"
    if not video_dir.exists():
        return None
    videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    return videos[0] if videos else None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_video(path: Path | None) -> dict | None:
    if path is None:
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return {
            "file": path.name,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": fps,
            "frames": frames,
            "duration_seconds": round(frames / fps, 3) if fps > 0 else None,
        }
    finally:
        cap.release()


def nvidia_driver_version() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.init()  # YOLO() modeli yuklerken cuda context'i henuz hazir olmuyor
    print(f"Cihaz: {device}")
    print(f"Her format {REPEATS} kez ayri ayri olculecek (guvenilirlik icin)\n")

    video_path = find_test_video()
    if video_path:
        print(f"Test videosu bulundu: {video_path.name}\n")
    else:
        print("UYARI: data/test_videos/ altinda video bulunamadi, "
              "sadece dummy tensor testi yapilacak.\n")

    results = {}
    for name, path in VARIANTS:
        if not path.exists():
            print(f"[ATLANDI] {name}: {path.name} bulunamadi. "
                  f"Once src/export_models.py calistirilmali.")
            continue
        print(f"Test ediliyor: {name} ({path.name})...")
        model = YOLO(str(path))

        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(0)

        pure_runs = [run_pure_once(model, device) for _ in range(REPEATS)]
        pure_stats = mean_std(pure_runs)
        print(f"  -> Saf hiz: {pure_stats['mean']} +/- {pure_stats['std']} FPS "
              f"(tekil olcumler: {pure_stats['runs']})")

        bench = {"fps": pure_stats}
        bench["file_size_mb"] = round(path.stat().st_size / (1024 * 1024), 2)
        if device.startswith("cuda"):
            bench["gpu_memory_mb"] = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 1)
            print(f"  -> Dosya boyutu: {bench['file_size_mb']} MB  |  GPU bellek: {bench['gpu_memory_mb']} MB")

        if video_path:
            video_runs = [run_video_once(model, device, video_path) for _ in range(REPEATS)]
            video_stats = mean_std(video_runs)
            bench["video_fps"] = video_stats
            print(f"  -> Video (uctan uca): {video_stats['mean']} +/- {video_stats['std']} FPS "
                  f"(tekil olcumler: {video_stats['runs']})")

        results[name] = bench
        print()

    if not results:
        print("Hicbir model bulunamadi, once export_models.py calistir.")
        return

    RESULTS_DIR.mkdir(exist_ok=True)

    baseline = results.get("PyTorch FP32", {})
    baseline_fps = baseline.get("fps", {}).get("mean")
    baseline_video_fps = baseline.get("video_fps", {}).get("mean")

    print("=" * 80)
    print(f"{'Format':<18}{'Saf FPS (ort+/-std)':>24}{'Hizlanma':>10}"
          f"{'Video FPS (ort+/-std)':>26}{'Video Hizlanma':>16}")
    print("=" * 80)
    for name, bench in results.items():
        fps = bench["fps"]
        fps_str = f"{fps['mean']}+/-{fps['std']}"
        speedup = f"{fps['mean'] / baseline_fps:.2f}x" if baseline_fps else "-"
        video = bench.get("video_fps")
        video_str = f"{video['mean']}+/-{video['std']}" if video else "-"
        video_speedup = (f"{video['mean'] / baseline_video_fps:.2f}x"
                          if video and baseline_video_fps else "-")
        print(f"{name:<18}{fps_str:>24}{speedup:>10}{video_str:>26}{video_speedup:>16}")

    print()
    print(f"{'Format':<18}{'Dosya (MB)':>12}{'GPU Bellek (MB)':>18}")
    for name, bench in results.items():
        print(f"{name:<18}{bench.get('file_size_mb', '-'):>12}{bench.get('gpu_memory_mb', '-'):>18}")

    out_json = RESULTS_DIR / f"{OUTPUT_PREFIX}_precision_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
            "cuda_compute_capability": (
                list(torch.cuda.get_device_capability(0)) if device.startswith("cuda") else None
            ),
            "driver_version": nvidia_driver_version(),
        },
        "software": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "tensorrt": trt.__version__ if trt else None,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
        },
        "protocol": {
            "input_size": IMG_SIZE,
            "warmup_iterations_per_repeat": WARMUP_ITERS,
            "timed_iterations_per_repeat": BENCH_ITERS,
            "independent_repeats": REPEATS,
            "video_loops_per_repeat": VIDEO_LOOPS,
            "pure_fps_scope": "model.predict on a synthetic 1x3x640x640 CUDA tensor",
            "video_fps_scope": "video decode plus model inference; annotation rendering excluded",
        },
        "video": describe_video(video_path),
        "variants": [{"label": label, "model_file": path.name} for label, path in VARIANTS],
        "result_file": out_json.name,
    }
    metadata_json = RESULTS_DIR / f"{OUTPUT_PREFIX}_benchmark_metadata.json"
    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\nSonuclar kaydedildi: {out_json.relative_to(ROOT)}")

    # Bar chart: saf hiz vs video, hata cubuklariyla (std sapma)
    names = list(results.keys())
    pure_means = [results[n]["fps"]["mean"] for n in names]
    pure_stds = [results[n]["fps"]["std"] for n in names]
    video_means = [results[n].get("video_fps", {}).get("mean") for n in names]
    video_stds = [results[n].get("video_fps", {}).get("std") for n in names]
    has_video = all(v is not None for v in video_means)

    x = range(len(names))
    width = 0.35 if has_video else 0.6
    fig, ax = plt.subplots(figsize=(8, 5))
    offset = width / 2 if has_video else 0
    bars1 = ax.bar([i - offset for i in x], pure_means, width, yerr=pure_stds, capsize=4,
                    label="Saf hiz (dummy)", color="#4C72B0")
    label_pad = max(pure_means + video_means) * 0.02
    for bar, val, err in zip(bars1, pure_means, pure_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + label_pad,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    if has_video:
        bars2 = ax.bar([i + offset for i in x], video_means, width, yerr=video_stds, capsize=4,
                        label="Video (uctan uca)", color="#C44E52")
        for bar, val, err in zip(bars2, video_means, video_stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + label_pad,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=8)
        ax.legend()
    ax.margins(y=0.12)

    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("FPS")
    ax.set_title(f"YOLOv8n - Format Kiyaslamasi (FPS, {REPEATS} tekrarin ortalamasi +/- std sapma)")
    fig.tight_layout()
    chart_path = RESULTS_DIR / f"{OUTPUT_PREFIX}_fps_comparison.png"
    fig.savefig(chart_path, dpi=150)
    print(f"Grafik kaydedildi: {chart_path.relative_to(ROOT)}")

    sizes = [results[n].get("file_size_mb", 0) for n in names]
    mems = [results[n].get("gpu_memory_mb", 0) for n in names]
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(names, sizes, color="#8172B2")
    ax1.set_ylabel("MB")
    ax1.set_title("Model dosya boyutu")
    ax2.bar(names, mems, color="#CCB974")
    ax2.set_ylabel("MB")
    ax2.set_title("GPU bellek kullanimi (peak)")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=15)
    fig2.tight_layout()
    chart2_path = RESULTS_DIR / f"{OUTPUT_PREFIX}_size_memory_comparison.png"
    fig2.savefig(chart2_path, dpi=150)
    print(f"Grafik kaydedildi: {chart2_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
