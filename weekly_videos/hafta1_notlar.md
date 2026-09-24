# Hafta 1 — Video Teslimi Notları

## Bu hafta yapılanlar
- Python sanal ortamı (`venv`) kuruldu.
- RTX 5060 (Blackwell mimarisi) için CUDA 12.8 destekli PyTorch kuruldu.
- Ultralytics YOLOv8 kütüphanesi kuruldu.
- Ortam doğrulama scripti (`src/check_env.py`) yazıldı: CUDA/cuDNN/PyTorch/Ultralytics
  sürümlerini ve GPU bilgisini kontrol ediyor.
- Baseline inference scripti (`src/baseline_inference.py`) yazıldı:
  - Pretrained YOLOv8n ağırlıkları indirilip yüklendi.
  - Örnek görsel üzerinde tespit doğrulandı (`results/week1_sample_detection.jpg`).
  - Saf model inference hızı (FPS, ortalama gecikme) ölçüldü (FP32, PyTorch, optimizasyonsuz).
  - Sonuçlar `results/week1_baseline_results.json` içine kaydedildi.

## Sonuçlar (özet)
- Cihaz: NVIDIA GeForce RTX 5060 Laptop GPU (CUDA 12.8, cuDNN 9.19, compute capability 12.0 / Blackwell)
- Model: YOLOv8n (pretrained, FP32, PyTorch, hiçbir optimizasyon yok — bu haftanın referans/baseline'ı)
- Örnek görsel (bus.jpg) üzerinde 6 nesne doğru tespit edildi → `results/week1_sample_detection.jpg`
- Saf model inference hızı (640x640, 100 ölçüm, I/O hariç): **182.79 FPS**, ortalama gecikme **5.47 ms**
- Gerçek video üzerinde uçtan uca hız (`solutions_ci_demo.mp4`, Ultralytics resmi demo videosu,
  okuma + inference + son işleme dahil): **76.19 FPS** (62 kare)
- Not: Kurulumda bir sorunla uğraştık — PyTorch'un en yeni sürümü (2.11.0) kendi iç modülünde
  bozuk bir import zinciri içeriyordu, ayrıca Windows'un "Akıllı Uygulama Denetimi" güvenlik
  özelliği numpy'nin çok yeni bir derlenmiş dosyasını güvenilmez bularak engelliyordu. Daha
  stabil/oturmuş sürümlere (PyTorch 2.7.1 + numpy 2.1.3) sabitleyince sorun çözüldü ve performans
  da belirgin şekilde arttı — bu, "en yeni sürüm her zaman en iyisi değildir, kararlılık da
  önemlidir" şeklinde staj raporunda bahsedilebilecek gerçek bir öğrenme deneyimi.

## Video için konuşma notları (taslak)
1. Projenin amacını kısaca özetle (YOLO + NVIDIA performans analizi).
2. Bu hafta ortamı nasıl kurduğunu göster (CUDA/PyTorch/Ultralytics kontrolü).
3. Baseline YOLOv8 inference'ı canlı çalıştır, örnek tespiti göster.
4. Ölçülen baseline FPS/gecikme sayılarını paylaş — bu, sonraki haftalarda
   (TensorRT, FP16, INT8) karşılaştırma yapılacak referans noktası olacak.
5. Gelecek hafta planı: ONNX → TensorRT dönüşümü, FP32 vs FP16 kıyaslaması.

## Hocaya sorulacak açık sorular
- YOLO versiyon tercihi (YOLOv8 ile devam / farklı bir sürüm mü?)
- Masaüstü mü web arayüzü mü öncelikli?
- Özel bir veri seti gerekiyor mu, yoksa COCO val2017 yeterli mi?
- Hedef donanım sadece RTX 5060 mı, yoksa Jetson gibi bir edge cihaz da eklenecek mi?
- Teslimat formatı (rapor/kod/sunum) ve toplam staj süresi netleşti mi?
