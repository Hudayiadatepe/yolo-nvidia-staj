"""Hafta 1: Ortam dogrulama - CUDA / cuDNN / PyTorch / Ultralytics kurulumunu kontrol eder."""
import sys


def main():
    print("=" * 60)
    print("ORTAM KONTROLU")
    print("=" * 60)

    print(f"\nPython: {sys.version}")

    try:
        import torch
        print(f"\nPyTorch: {torch.__version__}")
        print(f"CUDA mevcut mu: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA surumu (torch build): {torch.version.cuda}")
            print(f"cuDNN surumu: {torch.backends.cudnn.version()}")
            print(f"GPU sayisi: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  [{i}] {props.name} - {props.total_memory / 1024**3:.1f} GB"
                      f" - compute capability {props.major}.{props.minor}")
        else:
            print("UYARI: CUDA kullanilamiyor, CPU'ya dusulecek.")
    except ImportError:
        print("\nHATA: torch kurulu degil.")
        return

    try:
        import ultralytics
        print(f"\nUltralytics: {ultralytics.__version__}")
    except ImportError:
        print("\nHATA: ultralytics kurulu degil.")
        return

    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
    except ImportError:
        print("UYARI: opencv-python kurulu degil.")

    print("\n" + "=" * 60)
    print("Kontrol tamamlandi.")
    print("=" * 60)


if __name__ == "__main__":
    main()
