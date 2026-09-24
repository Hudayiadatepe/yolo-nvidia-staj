"""Hafta 3: PyTorch FP32 / TensorRT FP32 / TensorRT FP16 icin mAP (dogruluk) olcumu.

data/coco_val.yaml uzerinden COCO val2017 subset'i ile model.val() calistirir.
On kosul: src/download_coco_val.py calistirilmis olmali.

Kullanim:
    python src/measure_map.py
"""
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DATA_YAML = ROOT / "data" / "coco_val.yaml"

VARIANTS = [
    ("PyTorch FP32", MODELS_DIR / "yolov8n.pt"),
    ("TensorRT FP32", MODELS_DIR / "yolov8n_fp32.engine"),
    ("TensorRT FP16", MODELS_DIR / "yolov8n_fp16.engine"),
    ("TensorRT INT8", MODELS_DIR / "yolov8n_int8.engine"),
]


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"{DATA_YAML} yok, once download_coco_val.py calistir.")

    results = {}
    for name, path in VARIANTS:
        if not path.exists():
            print(f"[ATLANDI] {name}: {path.name} bulunamadi.")
            continue
        print(f"\n=== {name} ===")
        model = YOLO(str(path))
        metrics = model.val(data=str(DATA_YAML), imgsz=640, device="cuda:0", verbose=False)
        results[name] = {
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50-95": round(float(metrics.box.map), 4),
        }
        print(f"  mAP50: {results[name]['mAP50']}  |  mAP50-95: {results[name]['mAP50-95']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_json = RESULTS_DIR / "publication_map_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 45)
    print(f"{'Format':<18}{'mAP50':>12}{'mAP50-95':>15}")
    print("=" * 45)
    for name, r in results.items():
        print(f"{name:<18}{r['mAP50']:>12}{r['mAP50-95']:>15}")
    print(f"\nSonuclar kaydedildi: {out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
