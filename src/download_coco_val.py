"""Hafta 3: COCO val2017 veri setini indirir (sadece val, train2017 HARIC - 19GB'lik
gereksiz indirmeden kaciniyoruz).

Indirilen dosyalar:
  - val2017.zip (~1GB, 5000 gorsel) - resmi COCO sunucusu
  - coco2017labels.zip (~46MB, YOLO-format etiketler) - Ultralytics'in kendi deposu
    (bu zip train+val etiketlerini birlikte iceriyor, sadece val2017 klasorunu aliyoruz)

Sonuc: data/coco_val/images/val2017/*.jpg ve data/coco_val/labels/val2017/*.txt

Kullanim:
    python src/download_coco_val.py
"""
import shutil
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
COCO_DIR = ROOT / "data" / "coco_val"
TMP_DIR = ROOT / "data" / "_coco_tmp"

VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
LABELS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"


def download(url: str, dest: Path):
    if dest.exists():
        print(f"  zaten var, atlaniyor: {dest.name}")
        return
    print(f"  indiriliyor: {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="")
    print()


def main():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    (COCO_DIR / "images").mkdir(parents=True, exist_ok=True)
    (COCO_DIR / "labels").mkdir(parents=True, exist_ok=True)

    if (COCO_DIR / "images" / "val2017").exists() and (COCO_DIR / "labels" / "val2017").exists():
        print("val2017 zaten hazir, indirme atlaniyor.")
        return

    print("[1/2] val2017 gorselleri indiriliyor (~1GB)...")
    val_zip = TMP_DIR / "val2017.zip"
    download(VAL_IMAGES_URL, val_zip)
    if not (COCO_DIR / "images" / "val2017").exists():
        print("  aciliyor...")
        with zipfile.ZipFile(val_zip) as z:
            z.extractall(COCO_DIR / "images")

    print("\n[2/2] YOLO-format etiketler indiriliyor (~46MB)...")
    labels_zip = TMP_DIR / "coco2017labels.zip"
    download(LABELS_URL, labels_zip)
    print("  aciliyor (sadece val2017 klasoru)...")
    with zipfile.ZipFile(labels_zip) as z:
        members = [m for m in z.namelist() if "labels/val2017/" in m]
        z.extractall(TMP_DIR / "labels_extract", members=members)
    extracted_val_labels = TMP_DIR / "labels_extract" / "coco" / "labels" / "val2017"
    if not extracted_val_labels.exists():
        # bazi zip surumlerinde ust klasor farkli olabilir, bul
        candidates = list((TMP_DIR / "labels_extract").rglob("val2017"))
        extracted_val_labels = candidates[0] if candidates else None
    if extracted_val_labels is None:
        raise RuntimeError("val2017 etiket klasoru zip icinde bulunamadi")
    shutil.move(str(extracted_val_labels), str(COCO_DIR / "labels" / "val2017"))

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    n_images = len(list((COCO_DIR / "images" / "val2017").glob("*.jpg")))
    n_labels = len(list((COCO_DIR / "labels" / "val2017").glob("*.txt")))
    print(f"\nTamamlandi: {n_images} gorsel, {n_labels} etiket dosyasi -> {COCO_DIR}")


if __name__ == "__main__":
    main()
