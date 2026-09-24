"""Hafta 2: yolov8n.onnx dosyasindan dogrudan TensorRT Python API ile FP16 engine olusturur.

Ultralytics'in export(format='engine', half=True) yolu, yeni surumlerde agir bir
bagimlilik olan 'nvidia-modelopt' paketini kullanmaya calisiyor ve bu paketin
'lief' bagimliligi Windows Akilli Uygulama Denetimi tarafindan engelleniyor.
Bu script, TensorRT builder'in kendi yerlesik FP16 flag'ini kullanarak ayni
isi cok daha basit ve guvenilir sekilde yapar.

Kullanim:
    python src/build_fp16_engine.py
"""
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

# Bazı Windows sistemlerinde Smart App Control, ml_dtypes'in yerel DLL'sini
# engelleyebiliyor. ONNX FP32->FP16 dönüşümü düşük bitli özel türlere ihtiyaç
# duymadığı için yalnızca bu import başarısız olduğunda temel NumPy türleriyle
# sınırlı bir uyumluluk modülü sağlıyoruz.
try:
    import ml_dtypes  # noqa: F401
except (ImportError, OSError):
    ml_dtypes_compat = types.ModuleType("ml_dtypes")
    for name in (
        "bfloat16", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2",
        "float8_e5m2fnuz", "float4_e2m1fn", "float8_e8m0fnu",
    ):
        setattr(ml_dtypes_compat, name, np.float16)
    for name in ("int4", "int2"):
        setattr(ml_dtypes_compat, name, np.int8)
    for name in ("uint4", "uint2"):
        setattr(ml_dtypes_compat, name, np.uint8)
    ml_dtypes_compat.finfo = np.finfo
    ml_dtypes_compat.iinfo = np.iinfo
    sys.modules["ml_dtypes"] = ml_dtypes_compat

import onnx
import tensorrt as trt
from onnxconverter_common import float16

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
ONNX_FILE = MODELS_DIR / "yolov8n.onnx"
OUT_ENGINE = MODELS_DIR / "yolov8n_fp16.engine"


def main():
    if not ONNX_FILE.exists():
        raise FileNotFoundError(f"{ONNX_FILE} bulunamadi. Once export_models.py calistir.")

    # TensorRT'nin C++ parser'i Turkce karakterli ("ö") yollari okuyamiyor,
    # bu yuzden ONNX dosyasini ASCII-uyumlu gecici bir klasore kopyalayip
    # oradan okuyoruz.
    tmp_dir = Path(tempfile.gettempdir()) / "yolo_export_tmp"
    tmp_dir.mkdir(exist_ok=True)
    tmp_onnx = tmp_dir / "yolov8n.onnx"
    shutil.copy(ONNX_FILE, tmp_onnx)
    tmp_onnx_fp16 = tmp_dir / "yolov8n_fp16.onnx"
    tmp_engine = tmp_dir / "yolov8n_fp16.engine"

    print("ONNX modeli FP16'ya donusturuluyor...")
    fp32_model = onnx.load(str(tmp_onnx))
    fp16_model = float16.convert_float_to_float16(fp32_model, keep_io_types=True)
    onnx.save(fp16_model, str(tmp_onnx_fp16))

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    print(f"ONNX (FP16) dosyasi okunuyor: {tmp_onnx_fp16}")
    with open(tmp_onnx_fp16, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse hatasi")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    print("FP16 TensorRT engine olusturuluyor (birkac dakika surebilir)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine olusturulamadi")

    with open(tmp_engine, "wb") as f:
        f.write(serialized_engine)

    shutil.move(str(tmp_engine), str(OUT_ENGINE))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Tamamlandi -> {OUT_ENGINE}")


if __name__ == "__main__":
    main()
