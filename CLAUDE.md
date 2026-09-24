# Proje: YOLO + NVIDIA Performans Analizi ve Gerçek Zamanlı Uygulama

## Bağlam
3. sınıf Bilgisayar Mühendisliği öğrencisi olarak yaz stajı projesi. Hoca tarafından
verilen konu, 5 haftalık staj süresince geliştirilecek. Her Cuma o haftaya kadar
yapılanları özetleyen bir video hoca ile paylaşılacak.

## Amaç
Güncel YOLO (YOLOv8) nesne tespiti mimarisini NVIDIA yazılım bileşenleri
(CUDA, cuDNN, TensorRT) ve farklı hassasiyet formatları (FP32, FP16, INT8)
kullanarak optimize etmek, performans kazançlarını (FPS, gecikme, doğruluk/mAP)
nicel metriklerle analiz etmek ve sonuçları canlı video akışında test edilebilen
kullanıcı dostu bir masaüstü/web arayüzünde sunmak.

## Donanım / Ortam
- Yerel GPU: RTX 5060, Ryzen 7 AI 350, 16GB RAM (Windows)
- Alternatif: Google Colab T4
- Model: YOLOv8 (Ultralytics), pretrained ağırlıklar (sıfırdan eğitim yok)
- Veri seti: COCO val2017 (kıyaslama/test için, ~1GB)

## 5 Haftalık Plan
- **Hafta 1:** Ortam kurulumu (CUDA/cuDNN/PyTorch/Ultralytics), baseline YOLOv8
  inference, FPS ölçümü
- **Hafta 2:** ONNX → TensorRT dönüşümü, FP32 vs FP16 kıyaslaması (FPS, latency, mAP)
- **Hafta 3:** INT8 kantizasyon (kalibrasyon veri seti ile), FP32/FP16/INT8 tam
  karşılaştırma, grafikler
- **Hafta 4:** Masaüstü/web arayüzü (model ve optimizasyon seviyesi seçimi,
  webcam/video girişi, anlık FPS + tespit sonucu gösterimi)
- **Hafta 5:** İyileştirmeler, edge case'ler, final rapor/sunum hazırlığı

## Klasör Yapısı
- `src/` — kaynak kod (inference, dönüşüm scriptleri, arayüz)
- `models/` — pretrained ve dönüştürülmüş (ONNX/TensorRT) model dosyaları
- `data/` — test videoları, COCO val2017 alt kümesi
- `results/` — FPS/mAP/latency tabloları, grafikler
- `weekly_videos/` — haftalık teslim videolarının notları/scriptleri
- `notebooks/` — deneysel/keşif amaçlı Jupyter defterleri

## Notlar
- Hocadan netleştirme bekleyen sorular: YOLO versiyon tercihi, masaüstü/web
  önceliği, özel veri seti gerekip gerekmediği, hedef donanım (sadece RTX 5060 mı
  yoksa Jetson gibi bir edge cihaz mı), teslimat formatı (rapor/kod/sunum), tam
  staj süresi. Cevaplar gelince bu dosya güncellenmeli.
- Kod yazarken: PyTorch + Ultralytics YOLOv8 kullan, TensorRT dönüşümü için
  `trtexec` veya `torch2trt`/`ultralytics export` akışını tercih et.
- Kullanıcı Türkçe konuşuyor, günlük/samimi bir dille yardımcı ol; açıklamalar
  net ve eksiksiz olsun, gereksiz parçalara bölünmesin.
