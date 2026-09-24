"""Hafta 4: YOLOv8 + NVIDIA performans demo arayuzu (Streamlit).

Calistirma:
    venv\\Scripts\\streamlit.exe run src\\app.py

Mimari notu: video dosyalari (ornek video / yuklenen dosya) baştan sona islenip
gercek bir H.264 mp4 olarak kaydediliyor, sonra tarayicinin kendi native video
oynaticisiyla (st.video) gosteriliyor - bu, kare kare goruntu gondermekten
(st.image dongusu) cok daha akici. Webcam gercekten canli oldugu icin bu yontem
uygun degil, onun icin streamlit-webrtc ile gercek WebRTC akisi kullaniliyor
(kamera tarayici uzerinden aciliyor, bu yuzden webcam zaten platform bagimsiz).

Video isleme, iptal edilebilmesi icin ayri bir thread'de calisiyor (VideoJob +
_video_worker) - ana Streamlit thread'i st.fragment ile bu thread'in durumunu
periyodik olarak okuyup ekrani gunceller.
"""
import hmac
import os
import shutil
import tempfile
import threading
import time
import importlib.util
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import torch
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"


def ensure_base_model() -> None:
    """Download the small PyTorch model on first start of a clean clone."""
    target = MODELS_DIR / "yolov8n.pt"
    if target.exists():
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        YOLO(str(target))
    except Exception:
        # Arayuz asagida anlasilir bir "model bulunamadi" mesaji gosterecek.
        # Ag erisimi olmayan ortamlarda uygulamanin tamamen cokmesini engelle.
        return


ensure_base_model()

MODELS = [
    {"label": "PyTorch FP32", "file": "yolov8n.pt", "fps": 73.60, "map": 0.3681,
     "requires_cuda": False},
    {"label": "TensorRT FP32", "file": "yolov8n_fp32.engine", "fps": 137.66, "map": 0.3678,
     "requires_cuda": True},
    {"label": "TensorRT FP16", "file": "yolov8n_fp16.engine", "fps": 147.79, "map": 0.3678,
     "requires_cuda": True},
    {"label": "TensorRT INT8", "file": "yolov8n_int8.engine", "fps": 142.92, "map": 0.3272,
     "requires_cuda": True},
]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TENSORRT_AVAILABLE = importlib.util.find_spec("tensorrt") is not None
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_VIDEO_SECONDS = 5 * 60
TEMP_DIR_PREFIX = "yolo_staj_"
TEMP_DIR_MAX_AGE_SECONDS = 6 * 60 * 60

# Streamlit ayni model nesnesini farkli kullanici oturumlari arasinda cache'ler.
# Ultralytics/TensorRT nesneleri ayni anda birden fazla thread'den predict almaya
# uygun olmadigi icin tum inference cagrilarini tek kilitle siraya koyuyoruz.
INFERENCE_LOCK = threading.Lock()

# Trafik/yaya sahnelerinde model bazen (ozellikle tepeden cekilmis acilarda) insan
# siluetlerini hayvanla karistirabiliyor ("person" -> "horse"/"bird"/"cow" gibi).
# Bu proje trafik/yaya odakli oldugu icin, ilgisiz COCO siniflarini (hayvanlar,
# ev esyalari vb.) filtreleyip sadece insan+arac siniflarini gostermek, guven
# esigini yukseltmekten daha guvenilir bir cozum - dusuk VEYA yuksek guven
# skoruyla gelen yanlis siniflandirmalari da eliyor.
RELEVANT_CLASS_IDS = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck
# Not: "train" (6) sinifi kasten cikarildi - tunel/koridor gibi uzun, tekduze
# isikli sahnelerde tum kareyi "train" olarak yanlis etiketleme egiliminde
# (dusuk guvenle de olsa), ve bu proje icin zaten gerekli bir sinif degildi.

# FP16/INT8 engine'leri kendi TensorRT scriptimizle olusturduk (ultralytics'in
# export() yolunu degil), bu yuzden sinif isimleri metadata'si eksik kaliyor ve
# "class0" gibi gorunuyor. Standart COCO isimlerini elle geri koyuyoruz.
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag",
    27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
    32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
    36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
    57: "couch", 58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet",
    62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
    72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors",
    77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}

st.set_page_config(page_title="YOLO Performans", page_icon="◆", layout="wide")


def cleanup_stale_work_dirs():
    """Onceki oturumlardan kalan eski gecici video klasorlerini guvenle temizler."""
    temp_root = Path(tempfile.gettempdir()).resolve()
    cutoff = time.time() - TEMP_DIR_MAX_AGE_SECONDS
    for candidate in temp_root.glob(f"{TEMP_DIR_PREFIX}*"):
        try:
            resolved = candidate.resolve()
            # Silme hedefini sistem gecici klasorunun dogrudan altinda ve yalnizca
            # uygulamanin kendi on ekiyle sinirla.
            if resolved.parent != temp_root or not resolved.name.startswith(TEMP_DIR_PREFIX):
                continue
            if resolved.is_dir() and resolved.stat().st_mtime < cutoff:
                shutil.rmtree(resolved, ignore_errors=True)
        except OSError:
            # Baska bir oturum klasoru o anda kullaniyorsa sonraki turda denenir.
            continue


cleanup_stale_work_dirs()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

:root {
  --bg: #0C0E12;
  --panel: #12151B;
  --line: rgba(255,255,255,0.08);
  --text: #E6E4DF;
  --muted: #868B94;
  --accent: #C99A44;
  --accent-soft: rgba(201,154,68,0.12);
}

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

#MainMenu, footer {visibility: hidden;}
div[data-testid="stStatusWidget"] {display: none;}
div[data-testid="stDecoration"] {display: none;}
div[data-testid="stAppDeployButton"] {display: none;}
div[data-testid="stToolbarActions"] {display: none;}
/* header'i tamamen gizlemiyoruz - sidebar'i kapattiktan sonra geri acma
   butonu (stExpandSidebarButton) da header'in icinde yasiyor. */
header[data-testid="stHeader"] {background: transparent;}

.block-container {padding-top: 2.75rem; padding-bottom: 3rem; max-width: 1180px;}

/* Not: buradaki etiketler CSS text-transform:uppercase yerine kaynakta zaten
   BUYUK harfle yaziliyor - tarayicilar bu donusumde Turkce "i" harfini "I"
   (noktasiz) yapiyor, "I" (noktali) degil. */

/* header */
.app-title { font-size: 1.6rem; font-weight: 600; color: var(--text); margin: 0 0 0.3rem 0; line-height: 1.3; }
.app-sub { font-size: 0.92rem; color: var(--muted); margin-bottom: 0; }

/* sidebar */
section[data-testid="stSidebar"] {
  background: var(--panel); border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
.panel-label {
  font-size: 0.68rem; letter-spacing: 0.1em;
  color: var(--muted); font-weight: 600; margin: 1.4rem 0 0.5rem 0;
}
.panel-label:first-child { margin-top: 0; }

/* inputs */
div[data-baseweb="select"] > div, .stTextInput input {
  background: var(--bg) !important; border: 1px solid var(--line) !important;
  border-radius: 6px !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] { background-color: var(--accent) !important; }
.stSlider [data-baseweb="slider"] > div > div { background: var(--accent) !important; }

div[role="radiogroup"] label {
  padding: 0.15rem 0; font-size: 0.9rem;
}
div[role="radiogroup"] label div:first-child > div {
  border-color: var(--muted) !important;
}
div[role="radiogroup"] label input:checked + div > div {
  border-color: var(--accent) !important; background-color: var(--accent) !important;
}

/* buttons */
.stButton > button {
  border-radius: 6px; font-weight: 500; font-size: 0.9rem;
  padding: 0.5rem 1rem; transition: opacity 0.15s ease, border-color 0.15s ease;
  box-shadow: none;
}
.stButton > button:disabled { opacity: 0.4; }
button[kind="primary"] {
  background: var(--accent) !important; color: #16130A !important; border: none !important;
}
button[kind="primary"]:hover { opacity: 0.88; }
button[kind="secondary"] {
  background: transparent !important; color: var(--text) !important;
  border: 1px solid var(--line) !important;
}
button[kind="secondary"]:hover { border-color: var(--muted) !important; }

/* reference table */
.ref-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.ref-table th {
  text-align: left; font-weight: 500; color: var(--muted); font-size: 0.68rem;
  letter-spacing: 0.06em; text-transform: uppercase; padding: 0 0 0.5rem 0;
  border-bottom: 1px solid var(--line);
}
.ref-table td {
  padding: 0.45rem 0; color: var(--text); border-bottom: 1px solid var(--line);
  font-variant-numeric: tabular-nums;
}
.ref-table tr:last-child td { border-bottom: none; }
.ref-table td:not(:first-child), .ref-table th:not(:first-child) { text-align: right; }

/* viewer frame */
.viewer-label {
  font-size: 0.72rem; letter-spacing: 0.08em;
  color: var(--muted); margin: 1.6rem 0 0.6rem 0; font-weight: 500;
}
div[data-testid="stImage"] img, video {
  border: 1px solid var(--line); border-radius: 6px;
}
.status-line { font-size: 0.85rem; color: var(--muted); margin-top: 0.4rem; }

/* stat strip (replaces boxed metrics) */
.stat-strip { display: flex; gap: 0; border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line); margin-top: 1rem; }
.stat-item { flex: 1; padding: 0.7rem 1rem; border-right: 1px solid var(--line); }
.stat-item:last-child { border-right: none; }
.stat-label { font-size: 0.68rem; letter-spacing: 0.08em;
  color: var(--muted); margin-bottom: 0.25rem; }
.stat-value { font-size: 1.15rem; font-weight: 500; color: var(--text);
  font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)


def require_access_password():
    """Protect temporary public demos when APP_PASSWORD is configured."""
    configured_password = os.getenv("APP_PASSWORD")
    if not configured_password or st.session_state.get("authenticated"):
        return

    st.markdown('<div class="app-title">YOLOv8 Demo Girişi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-sub">Bu geçici demo yalnızca yetkili erişim içindir.</div>',
        unsafe_allow_html=True,
    )
    with st.form("access_form"):
        entered_password = st.text_input("Erişim parolası", type="password")
        submitted = st.form_submit_button("Giriş", type="primary")

    if submitted:
        if hmac.compare_digest(entered_password, configured_password):
            st.session_state.authenticated = True
            st.rerun()
        time.sleep(1)
        st.error("Erişim parolası hatalı.")

    st.stop()


require_access_password()


@st.cache_resource
def load_model(path_str: str):
    return YOLO(path_str)


def fix_class_names(result):
    if not result.names or any(str(v).startswith("class") for v in result.names.values()):
        result.names = COCO_NAMES
    return result


def render_stats(placeholder, fps, det, model_name, device):
    placeholder.markdown(f"""
    <div class="stat-strip">
      <div class="stat-item"><div class="stat-label">FPS</div><div class="stat-value">{fps}</div></div>
      <div class="stat-item"><div class="stat-label">TESPİT</div><div class="stat-value">{det}</div></div>
      <div class="stat-item"><div class="stat-label">MODEL</div><div class="stat-value">{model_name}</div></div>
      <div class="stat-item"><div class="stat-label">CİHAZ</div><div class="stat-value">{device}</div></div>
    </div>
    """, unsafe_allow_html=True)


def validate_video_file(path: Path) -> tuple[bool, str]:
    """Dosyanin gercekten acilabilir/okunabilir bir video olup olmadigini kontrol eder."""
    if not path.exists():
        return False, "Dosya bulunamadı."
    if path.stat().st_size == 0:
        return False, "Dosya boş görünüyor."

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False, "Bu dosya geçerli bir video olarak okunamadı. Lütfen desteklenen bir video dosyası seçin."
        ok, _ = cap.read()
        if not ok:
            return False, "Video dosyasından kare okunamadı. Dosya bozuk olabilir."
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps > 0 and frame_count > 0 and frame_count / fps > MAX_VIDEO_SECONDS:
            return False, f"Video en fazla {MAX_VIDEO_SECONDS // 60} dakika olabilir."
    finally:
        cap.release()
    return True, ""


class VideoJob:
    """Video isleme thread'i ile ana Streamlit script'i arasindaki paylasilan durum.

    streamlit-webrtc'nin kendi video_processor'unu ayri thread'de calistirip
    ana thread'den periyodik okuma yapan deseninin aynisi - bu uygulamada zaten
    kanitlanmis bir yontem.
    """

    def __init__(self):
        self.status = "idle"  # idle | running | done | cancelled | error
        self.progress = 0.0
        self.frame_count = 0
        self.total_frames = 0
        self.start_time = None
        self.error = ""
        self.work_dir: Path | None = None
        self.graduated = False  # terminal duruma gectikten sonra tek seferlik rerun yapildi mi
        self.cancel_event = threading.Event()
        self.thread: threading.Thread | None = None

        # Tekli mod
        self.output_path: Path | None = None
        self.stats = None
        self.download_name = "processed_video.mp4"

        # Karsilastirma modu (iki model yan yana)
        self.compare = False
        self.model_label_a = ""
        self.model_label_b = ""
        self.output_path_a: Path | None = None
        self.output_path_b: Path | None = None
        self.stats_a = None
        self.stats_b = None

    def cleanup_work_dir(self):
        if self.work_dir and self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)


class _JobCancelled(Exception):
    """Bir gecis sirasinda iptal edildigini yukari bildirmek icin."""


def _encode_pass(job: VideoJob, source_path: Path, model, conf: float, device: str,
                  out_path: Path, max_w: int, progress_lo: float, progress_hi: float,
                  classes: list[int] | None = None) -> dict:
    """Videoyu tek bir modelle bastan sona isler, out_path'e H.264 olarak yazar.

    progress_lo/hi, bu gecisin job.progress uzerinde hangi araliga eslenecegini
    belirtir - karsilastirma modunda iki gecis oldugu icin (0.0-0.5) ve (0.5-1.0)
    olarak kullaniliyor, tek model modunda (0.0-1.0).
    """
    cap = cv2.VideoCapture(str(source_path))
    container = None
    stream = None
    frame_count = 0
    total_det = 0

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        job.total_frames = total

        # Ilk cagrida TensorRT/CUDA context kurulumu birkac yuz ms surebiliyor - bunu
        # zamanlamadan once bitirelim, yoksa ortalama FPS gercekci olmayan sekilde
        # dusuk cikiyor (ozellikle kisa videolarda).
        warmup_frame = np.zeros((640, 640, 3), dtype="uint8")
        with INFERENCE_LOCK:
            model.predict(warmup_frame, imgsz=640, device=device, verbose=False)
        pass_start = time.perf_counter()

        while True:
            if job.cancel_event.is_set():
                raise _JobCancelled

            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            if w > max_w:
                scale = max_w / w
                w, h = int(w * scale), int(h * scale)
            w, h = (w // 2) * 2, (h // 2) * 2  # libx264 cift sayi genislik/yukseklik istiyor
            frame = cv2.resize(frame, (w, h))

            with INFERENCE_LOCK:
                results = model.predict(
                    frame, imgsz=640, conf=conf, device=device,
                    classes=classes, verbose=False,
                )
            result = fix_class_names(results[0])
            annotated = result.plot()
            total_det += len(result.boxes)

            if container is None:
                container = av.open(str(out_path), mode="w")
                stream = container.add_stream("libx264", rate=int(round(fps)))
                stream.width = w
                stream.height = h
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "23", "preset": "ultrafast"}

            # annotated zaten BGR - PyAV'a dogrudan bgr24 olarak verip tek adimda
            # yuv420p'ye ceviriyoruz.
            vframe = av.VideoFrame.from_ndarray(annotated, format="bgr24")
            vframe = vframe.reformat(format="yuv420p")
            for packet in stream.encode(vframe):
                container.mux(packet)

            frame_count += 1
            job.frame_count = frame_count
            local_progress = min(frame_count / total, 1.0) if total > 0 else 0.0
            job.progress = progress_lo + local_progress * (progress_hi - progress_lo)

        if stream is not None:
            for packet in stream.encode():
                container.mux(packet)

        elapsed = time.perf_counter() - pass_start
        return {
            "avg_fps": frame_count / elapsed if elapsed > 0 else 0.0,
            "avg_det": (total_det / frame_count) if frame_count else 0.0,
        }
    finally:
        cap.release()
        if container is not None:
            container.close()


def _video_worker(job: VideoJob, source_path: Path, model_a, conf: float, device: str,
                   model_b=None, max_w: int = 960, classes: list[int] | None = None):
    """Video isleme - ayri thread'de calisir, Streamlit API'sine dokunmaz.

    model_b verilirse karsilastirma modu: ayni video iki modelle sirayla islenir
    (ayni anda degil - GPU/TensorRT belleginde iki modeli birden calistirmamak
    icin sirali gidiyoruz, daha guvenli).
    """
    job.status = "running"
    job.start_time = time.perf_counter()
    try:
        if model_b is None:
            out_path = job.work_dir / "processed_output.mp4"
            job.stats = _encode_pass(job, source_path, model_a, conf, device, out_path, max_w, 0.0, 1.0,
                                      classes=classes)
            job.output_path = out_path
        else:
            out_a = job.work_dir / "compare_a.mp4"
            out_b = job.work_dir / "compare_b.mp4"
            job.stats_a = _encode_pass(job, source_path, model_a, conf, device, out_a, max_w, 0.0, 0.5,
                                        classes=classes)
            job.stats_b = _encode_pass(job, source_path, model_b, conf, device, out_b, max_w, 0.5, 1.0,
                                        classes=classes)
            job.output_path_a = out_a
            job.output_path_b = out_b
        job.status = "done"
    except _JobCancelled:
        job.status = "cancelled"
    except Exception as exc:  # noqa: BLE001 - kullaniciya anlasilir mesaj gostermek icin genis yakaliyoruz
        job.error = str(exc)
        job.status = "error"
    finally:
        if job.status in ("cancelled", "error") and job.work_dir:
            job.cleanup_work_dir()


def create_job_work_dir() -> Path:
    """Onceki isin (varsa) klasorunu temizler, yeni benzersiz bir calisma klasoru acar.

    Boylece art arda islenen videolar birbirinin dosyasini ezmez ve eski gecici
    dosyalar birikmez.
    """
    old_job: VideoJob | None = st.session_state.get("video_job")
    if old_job is not None:
        old_job.cancel_event.set()
        old_job.cleanup_work_dir()
    return Path(tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX))


def start_video_job(work_dir: Path, source_path: Path, download_name: str, model, conf: float, device: str,
                     model_label: str = "", model_b=None, model_label_b: str = "",
                     classes: list[int] | None = None):
    """Verilen calisma klasorunde yeni bir VideoJob baslatir (arka plan thread'i).

    model_b verilirse karsilastirma modu acilir (bkz. _video_worker).
    """
    job = VideoJob()
    job.work_dir = work_dir
    job.download_name = download_name
    job.compare = model_b is not None
    job.model_label_a = model_label
    job.model_label_b = model_label_b
    st.session_state.video_job = job

    thread = threading.Thread(
        target=_video_worker, args=(job, source_path, model, conf, device, model_b, 960, classes), daemon=True,
    )
    job.thread = thread
    thread.start()


def reset_video_job_on_source_change():
    """Remove stale video output when switching between file, sample and webcam."""
    old_job: VideoJob | None = st.session_state.pop("video_job", None)
    if old_job is None:
        return
    old_job.cancel_event.set()
    # Calisan thread iptal durumunda kendi klasorunu finally blogunda temizler.
    # Tamamlanmis islerde ise artik indirilecek bir cikti kalmayacagi icin burada
    # guvenle temizleyebiliriz.
    if old_job.status != "running":
        old_job.cleanup_work_dir()


class YOLOVideoProcessor(VideoProcessorBase):
    """Webcam icin: her kareyi WebRTC uzerinden alir, tespit cizer, geri yollar.

    Kamera tarayici uzerinden (getUserMedia) aciliyor, bu yuzden bu kisim zaten
    platform bagimsiz - Windows'a ozel bir API kullanilmiyor.
    """

    def __init__(self):
        self.model = None
        self.conf = 0.25
        self.classes: list[int] | None = None
        self.last_fps = 0.0
        self.last_det = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # tarayici goruntuyu ayna yerine oldugu gibi veriyor
        if self.model is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        t0 = time.perf_counter()
        with INFERENCE_LOCK:
            results = self.model.predict(
                img, imgsz=640, conf=self.conf, device=DEVICE,
                classes=self.classes, verbose=False,
            )
        result = fix_class_names(results[0])
        annotated = result.plot()
        self.last_fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
        self.last_det = len(result.boxes)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


st.markdown('<div class="app-title">YOLOv8 Nesne Tespiti</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">NVIDIA TensorRT optimizasyon seviyeleri arasında canlı karşılaştırma</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="panel-label">MODEL</div>', unsafe_allow_html=True)

    # TensorRT engine dosyalari NVIDIA GPU ve uyumlu TensorRT calisma zamani
    # gerektirir. CPU tabanli bir bulut ortamina dosyalar kopyalanmis olsa bile
    # secenek olarak gostermek model yuklenirken hata verilmesine yol acar.
    available = [
        m for m in MODELS
        if (MODELS_DIR / m["file"]).exists()
        and (not m["requires_cuda"] or (DEVICE.startswith("cuda") and TENSORRT_AVAILABLE))
    ]
    missing = [m["label"] for m in MODELS if not (MODELS_DIR / m["file"]).exists()]
    unavailable_on_device = [
        m["label"] for m in MODELS
        if (MODELS_DIR / m["file"]).exists()
        and m["requires_cuda"]
        and (not DEVICE.startswith("cuda") or not TENSORRT_AVAILABLE)
    ]
    if missing:
        st.caption(f"Bulunamayan modeller: {', '.join(missing)}")
    if unavailable_on_device:
        st.info(
            "Bu sunucuda uyumlu NVIDIA GPU/TensorRT ortamı bulunmadığı için "
            "TensorRT modelleri devre dışı. PyTorch FP32 ile tanıtım modu kullanılabilir."
        )

    labels = [m["label"] for m in available]
    default_idx = labels.index("TensorRT FP16") if "TensorRT FP16" in labels else 0
    model_label = st.selectbox("Model", labels, index=default_idx if labels else 0,
                                label_visibility="collapsed")

    st.markdown('<div class="panel-label">GÜVEN EŞİĞİ</div>', unsafe_allow_html=True)
    conf = st.slider("Güven eşiği", 0.1, 0.9, 0.25, 0.05, label_visibility="collapsed")
    st.caption("Eşiği değiştirdikten sonra videoyu tekrar işlemen gerekir "
               "(eşik, modelin kendi hesaplama aşamasında kullanılıyor).")

    only_relevant = st.checkbox("Sadece insan/araç sınıflarını göster", value=True)
    st.caption("Trafik/yaya sahnelerinde model bazen insan siluetlerini hayvanla "
               "karıştırabiliyor (özellikle tepeden açılarda) - bu seçenek böyle "
               "alakasız sınıfları eler.")
    class_filter = RELEVANT_CLASS_IDS if only_relevant else None

    st.markdown('<div class="panel-label">KAYNAK</div>', unsafe_allow_html=True)
    source_type = st.radio(
        "Kaynak",
        ["Örnek video", "Webcam", "Dosya yükle"],
        key="source_type",
        on_change=reset_video_job_on_source_change,
        label_visibility="collapsed",
    )
    uploaded = None
    if source_type == "Dosya yükle":
        uploaded = st.file_uploader("Video seç", type=["mp4", "avi", "mov"], label_visibility="collapsed")
        st.caption(f"En fazla {MAX_UPLOAD_BYTES // 1024 // 1024} MB ve "
                   f"{MAX_VIDEO_SECONDS // 60} dakika.")
        st.warning(
            "Kişisel veri veya izinsiz güvenlik kamerası kaydı yüklemeyin. "
            "Dosya yalnızca işlem süresince geçici olarak saklanır."
        )

    compare_mode = False
    compare_label = None
    if source_type != "Webcam" and len(labels) > 1:
        compare_mode = st.checkbox("İki modeli yan yana karşılaştır")
        if compare_mode:
            other_labels = [lbl for lbl in labels if lbl != model_label]
            compare_label = st.selectbox("Karşılaştırılacak ikinci model", other_labels,
                                          label_visibility="collapsed")

    st.markdown('<div class="panel-label">NİHAİ REFERANS SONUÇLARI</div>', unsafe_allow_html=True)
    rows = "".join(
        f"<tr><td>{m['label']}</td><td>{m['fps']}</td><td>{m['map']:.3f}</td></tr>"
        for m in MODELS
    )
    st.markdown(f"""
    <table class="ref-table">
      <thead><tr><th>Model</th><th>FPS</th><th>mAP</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)
    st.caption(
        "AGPL-3.0 · "
        "[Kaynak kod](https://github.com/HudayiAdatepe/yolo-nvidia-staj) · "
        "[ORCID](https://orcid.org/0009-0005-2389-3061)"
    )

if source_type == "Webcam":
    st.markdown('<div class="viewer-label">CANLI GÖRÜNTÜ</div>', unsafe_allow_html=True)

    stat_placeholder = st.empty()

    if available:
        model_path = next(m["file"] for m in available if m["label"] == model_label)
        model = load_model(str(MODELS_DIR / model_path))

        ctx = webrtc_streamer(
            key="webcam-yolo",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=YOLOVideoProcessor,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )
        if ctx.video_processor:
            ctx.video_processor.model = model
            ctx.video_processor.conf = conf
            ctx.video_processor.classes = class_filter

        @st.fragment(run_every=0.5)
        def show_webcam_stats():
            if ctx.state.playing and ctx.video_processor:
                render_stats(stat_placeholder, f"{ctx.video_processor.last_fps:.1f}",
                             ctx.video_processor.last_det, model_label, DEVICE)
            else:
                render_stats(stat_placeholder, "&ndash;", "&ndash;", model_label, DEVICE)

        show_webcam_stats()
    else:
        render_stats(stat_placeholder, "&ndash;", "&ndash;", "&ndash;", DEVICE)

else:
    job: VideoJob | None = st.session_state.get("video_job")
    is_running = job is not None and job.status == "running"

    btn_col1, btn_col2 = st.columns([1, 1])
    process = btn_col1.button("Videoyu İşle", type="primary", disabled=is_running or not available)
    cancel = btn_col2.button("İşlemi İptal Et", type="secondary", disabled=not is_running)

    st.markdown('<div class="viewer-label">GÖRÜNTÜ</div>', unsafe_allow_html=True)

    if cancel and job is not None:
        job.cancel_event.set()

    if process and available:
        model_path = next(m["file"] for m in available if m["label"] == model_label)
        model = load_model(str(MODELS_DIR / model_path))

        source_path = None
        download_name = "processed_video.mp4"
        work_dir = None

        if source_type == "Örnek video":
            sample = next((DATA_DIR / "test_videos").glob("*.mp4"), None)
            if sample is None:
                st.error("data/test_videos/ altında örnek video bulunamadı.")
            else:
                ok, msg = validate_video_file(sample)
                if not ok:
                    st.error(msg)
                else:
                    source_path = sample
        else:
            if uploaded is None:
                st.warning("Önce bir video dosyası yükle.")
            elif uploaded.size > MAX_UPLOAD_BYTES:
                st.error(f"Dosya en fazla {MAX_UPLOAD_BYTES // 1024 // 1024} MB olabilir.")
            else:
                # Girdi dosyasini da isin kendi calisma klasorune koyuyoruz, boylece
                # tek bir cleanup noktasi yeterli oluyor (input + output ayni yerde).
                work_dir = create_job_work_dir()
                tmp_path = work_dir / f"input_video{Path(uploaded.name).suffix or '.mp4'}"
                tmp_path.write_bytes(uploaded.getvalue())
                ok, msg = validate_video_file(tmp_path)
                if not ok:
                    st.error(msg)
                    shutil.rmtree(work_dir, ignore_errors=True)
                    work_dir = None
                else:
                    source_path = tmp_path
                    download_name = f"{Path(uploaded.name).stem}_processed.mp4"

        if source_path is not None:
            if work_dir is None:
                work_dir = create_job_work_dir()
            model_b = None
            if compare_mode and compare_label:
                compare_path = next(m["file"] for m in available if m["label"] == compare_label)
                model_b = load_model(str(MODELS_DIR / compare_path))
            start_video_job(work_dir, source_path, download_name, model, conf, DEVICE,
                             model_label=model_label, model_b=model_b, model_label_b=compare_label or "",
                             classes=class_filter)
            st.rerun()

    # Bu placeholder'lar bilerek fragment'in KENDI icinde olusturuluyor (disaridan
    # verilmiyor). Streamlit, bir container disaridaki tam calistirmada hic
    # doldurulmadan sadece fragment icinde ilk kez doldurulursa
    # StreamlitInvalidLayoutContextError firlatiyor - hepsini fragment'e tasimak
    # bunu onluyor.
    #
    # run_every sadece is calisirken aktif: is bitince (done/cancelled/error) hala
    # 0.3 saniyede bir tikleyip video elementini sifirdan yeniden olusturmaya devam
    # ederse, kullanici oynatmaya baslayamadan video surekli resetleniyor ("sayfa
    # kendini yeniliyor" hissi). Is bitince fragment'in kendisi TEK SEFERLIK bir
    # tam rerun tetikleyip (asagida "graduated" bayragi) dis kod bu fonksiyonu
    # run_every=None ile yeniden tanimliyor, boylece tikleme duruyor.
    _job_now = st.session_state.get("video_job")
    _poll_interval = 0.3 if (_job_now and _job_now.status not in ("done", "cancelled", "error")) else None

    @st.fragment(run_every=_poll_interval)
    def show_job_status():
        job = st.session_state.get("video_job")
        progress_slot = st.empty()
        status_slot = st.empty()

        if job is None:
            stat_placeholder = st.empty()
            render_stats(stat_placeholder, "&ndash;", "&ndash;",
                         model_label if available else "&ndash;", DEVICE)
            return

        if job.status == "running":
            elapsed = time.perf_counter() - job.start_time if job.start_time else 0.0
            eta_text = ""
            if job.progress > 0.05 and elapsed > 1.0:
                remaining = elapsed / job.progress * (1 - job.progress)
                eta_text = f" &middot; yaklaşık {remaining:.0f} saniye kaldı"
            elif elapsed > 0.5:
                eta_text = " &middot; işlem süresi hesaplanıyor..."
            progress_slot.progress(
                job.progress,
                text=f"Video işleniyor - kare {job.frame_count}/{job.total_frames or '?'}",
            )
            status_slot.markdown(f'<div class="status-line">İşleniyor{eta_text}</div>',
                                  unsafe_allow_html=True)
            stat_placeholder = st.empty()
            render_stats(stat_placeholder, "&ndash;", "&ndash;", model_label, DEVICE)

        elif job.status == "done" and job.compare:
            col_a, col_b = st.columns(2)
            for col, label, out_path, stats, tag in (
                (col_a, job.model_label_a, job.output_path_a, job.stats_a, "a"),
                (col_b, job.model_label_b, job.output_path_b, job.stats_b, "b"),
            ):
                with col:
                    st.markdown(f'<div class="status-line">{label}</div>', unsafe_allow_html=True)
                    if out_path and out_path.exists():
                        st.video(str(out_path))
                        st.download_button(
                            "İndir", data=out_path.read_bytes(),
                            file_name=f"{job.download_name.rsplit('.', 1)[0]}_{tag}.mp4",
                            mime="video/mp4", key=f"download_{tag}",
                        )
                    s = stats or {"avg_fps": 0.0, "avg_det": 0.0}
                    st.caption(f"FPS: {s['avg_fps']:.1f} &middot; Tespit: {s['avg_det']:.1f} ort.")

        elif job.status == "done":
            video_slot = st.empty()
            download_slot = st.empty()
            stat_placeholder = st.empty()
            video_slot.video(str(job.output_path))
            if job.output_path and job.output_path.exists():
                download_slot.download_button(
                    "Videoyu İndir", data=job.output_path.read_bytes(),
                    file_name=job.download_name, mime="video/mp4",
                )
            s = job.stats or {"avg_fps": 0.0, "avg_det": 0.0}
            render_stats(stat_placeholder, f"{s['avg_fps']:.1f}", f"{s['avg_det']:.1f} ort.",
                         model_label, DEVICE)

        elif job.status == "cancelled":
            status_slot.info("Video işleme iptal edildi.")
            stat_placeholder = st.empty()
            render_stats(stat_placeholder, "&ndash;", "&ndash;", model_label, DEVICE)

        elif job.status == "error":
            status_slot.error(
                "Video işlenirken bir sorun oluştu. Dosyanın bozuk olmadığından "
                f"emin olup tekrar dener misin? (Detay: {job.error})"
            )
            stat_placeholder = st.empty()
            render_stats(stat_placeholder, "&ndash;", "&ndash;", model_label, DEVICE)

        # Is az once bir terminal duruma gectiyse (done/cancelled/error), disarisini
        # bir kez daha calistirip run_every'i kapatmasi icin tek seferlik rerun.
        # ("idle" durumu haric tutuluyor - is henuz thread'de baslamadan status
        # gecici olarak "idle" gorunebilir, bunu yanlislikla "bitti" saymamak lazim.)
        if job.status in ("done", "cancelled", "error") and not job.graduated:
            job.graduated = True
            st.rerun()

    show_job_status()
