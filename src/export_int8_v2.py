"""Hafta 3 (iyilestirme denemesi): INT8 engine'i 'entropy' kalibrasyon yontemi ve
daha fazla kalibrasyon goruntusuyle tekrar olusturur.

Ultralytics'in kendi export() yolu calibration_method='max' ve 512 goruntu kullanmaya
sabit kodlanmis. modelopt'un varsayilani aslinda 'entropy' (TensorRT'nin klasik
KL-divergence tabanli yontemi, genelde 'max'tan daha az dogruluk kaybi verir).
Bu script modelopt.onnx.quantization.quantize'i dogrudan cagirip bunu deniyor.

Kullanim:
    python src/export_int8_v2.py
"""
import random
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnx
import tensorrt as trt
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
COCO_IMAGES = ROOT / "data" / "coco_val" / "images" / "val2017"
IMG_SIZE = 640
N_CALIB = 1000
CALIBRATION_METHOD = "entropy"


def load_calibration_batch(n: int) -> np.ndarray:
    """COCO val2017'den n gorsel secip modele uygun formata getirir."""
    all_images = sorted(COCO_IMAGES.glob("*.jpg"))
    random.seed(0)
    chosen = random.sample(all_images, min(n, len(all_images)))

    batch = np.zeros((len(chosen), 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i, img_path in enumerate(chosen):
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        batch[i] = img.transpose(2, 0, 1)
    return batch


def main():
    tmp_dir = Path(tempfile.gettempdir()) / "yolo_export_tmp"
    tmp_dir.mkdir(exist_ok=True)
    tmp_weights = tmp_dir / "yolov8n.pt"
    shutil.copy(MODELS_DIR / "yolov8n.pt", tmp_weights)

    print("[1/4] ONNX'e donusturuluyor...")
    model = YOLO(str(tmp_weights))
    onnx_path = Path(model.export(format="onnx", imgsz=IMG_SIZE))

    print(f"[2/4] {N_CALIB} kalibrasyon goruntusu hazirlaniyor...")
    calib_data = load_calibration_batch(N_CALIB)
    print(f"  -> {calib_data.shape[0]} goruntu hazir")

    input_name = onnx.load(str(onnx_path), load_external_data=False).graph.input[0].name

    print(f"[3/4] INT8 kantizasyon (yontem: {CALIBRATION_METHOD})...")
    from modelopt.onnx.quantization import quantize as modelopt_quantize

    int8_onnx_path = onnx_path.with_suffix(".int8v2.onnx")
    modelopt_quantize(
        str(onnx_path),
        quantize_mode="int8",
        calibration_data={input_name: calib_data},
        calibration_method=CALIBRATION_METHOD,
        calibration_eps=["cpu"],
        output_path=str(int8_onnx_path),
        op_types_to_exclude=["Sigmoid"],
    )

    print("[4/4] TensorRT engine olusturuluyor...")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    with open(int8_onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse hatasi")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine olusturulamadi")

    tmp_engine = tmp_dir / "yolov8n_int8_v2.engine"
    with open(tmp_engine, "wb") as f:
        f.write(serialized_engine)

    dest = MODELS_DIR / "yolov8n_int8_v2.engine"
    shutil.move(str(tmp_engine), str(dest))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Tamamlandi -> {dest}")


if __name__ == "__main__":
    main()
