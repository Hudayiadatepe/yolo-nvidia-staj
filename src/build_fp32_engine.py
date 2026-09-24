"""Build a device-specific FP32 TensorRT engine from the exported YOLOv8n ONNX model.

TensorRT plan files are tied to the GPU model and TensorRT runtime. Run this
script on the machine that will execute the benchmark or application.
"""
import shutil
import tempfile
from pathlib import Path

import tensorrt as trt


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
ONNX_FILE = MODELS_DIR / "yolov8n.onnx"
OUT_ENGINE = MODELS_DIR / "yolov8n_fp32.engine"


def main():
    if not ONNX_FILE.exists():
        raise FileNotFoundError(f"{ONNX_FILE} bulunamadi. Once export_models.py calistir.")

    # TensorRT'nin Windows C++ katmani Turkce karakterli yollarda sorun
    # yasayabildigi icin yapim islemini ASCII uyumlu gecici dizinde yurutuyoruz.
    tmp_dir = Path(tempfile.gettempdir()) / "yolo_export_tmp"
    tmp_dir.mkdir(exist_ok=True)
    tmp_onnx = tmp_dir / "yolov8n.onnx"
    tmp_engine = tmp_dir / "yolov8n_fp32.engine"
    shutil.copy(ONNX_FILE, tmp_onnx)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    print(f"ONNX dosyasi okunuyor: {tmp_onnx}")
    with tmp_onnx.open("rb") as handle:
        if not parser.parse(handle.read()):
            for index in range(parser.num_errors):
                print(parser.get_error(index))
            raise RuntimeError("ONNX parse hatasi")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    print("FP32 TensorRT engine olusturuluyor...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine olusturulamadi")

    tmp_engine.write_bytes(serialized_engine)
    shutil.move(str(tmp_engine), str(OUT_ENGINE))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Tamamlandi -> {OUT_ENGINE}")


if __name__ == "__main__":
    main()
