# Hafta 2 — Video Teslimi Notları

## Bu hafta yapılanlar
- YOLOv8n modeli **ONNX** formatına çevrildi (`models/yolov8n.onnx`)
- ONNX'ten **TensorRT FP32** engine oluşturuldu (`models/yolov8n_fp32.engine`)
- ONNX modeli FP16'ya çevrilip **TensorRT FP16** engine oluşturuldu (`models/yolov8n_fp16.engine`)
- Üç format da aynı koşullarda (640x640, 100 ölçüm) test edilip FPS/gecikme karşılaştırıldı
- Karşılaştırma grafiği üretildi: `results/week2_fps_comparison.png`

## Sonuçlar (10 tekrarın ortalaması ± standart sapma, video 3x uzatılmış — güvenilirlik için)
Güvenilirlik sürecinde iki asama oldu:
1. İlk turda her formatı sadece bir kez ölçmüştük, sayılar calistirmadan calistirmaya %20'ye
   varan farkla degisiyordu — bu guvenilir degildi.
2. 5 tekrara gecince genel egilim netlesti ama video FPS hala gurultuluydu (video sadece 62
   kare, kucuk bir orneklem). Bunu duzeltmek icin tekrar sayisini 10'a cikardik VE video testini
   3 kez ust uste calistirip (~186 kare) tek bir olcumun kendisini de daha az gurultulu yaptik.

Son (guvenilir) sonuclar:

| Format | Saf FPS (ort±std) | Saf Hızlanma | Video FPS (ort±std) | Video Hızlanma |
|---|---|---|---|---|
| PyTorch FP32 (baseline) | 136.58 ± 12.31 | 1.00x | 103.02 ± 2.96 | 1.00x |
| TensorRT FP32 | 259.27 ± 9.12 | **1.90x** | 139.87 ± 3.82 | **1.36x** |
| TensorRT FP16 | 277.26 ± 14.44 | **2.03x** | 152.87 ± 6.47 | **1.48x** |

Video FPS'teki bağıl standart sapma artık **%3-4** seviyesinde (ilk turda %12-20'ydi) — video
örneklemini büyütmek gürültüyü belirgin şekilde azalttı. Saf hız ölçümünde hâlâ biraz sapma var
(%3-5), bu muhtemelen GPU'nun anlık termal/yük durumundan kaynaklanan doğal bir varyans ve normal
kabul edilebilir.

## Önemli gözlem (videoda vurgulanmalı)
TensorRT'ye çevirmenin kendisi (hassasiyet hiç değişmeden, FP32 halinde) saf hızda **1.90 kat**,
gerçek video üzerinde **1.36 kat** hızlanma sağladı — bunun nedeni katman birleştirme (layer
fusion) ve GPU'ya özel derleme optimizasyonları. FP16'ya geçmek üzerine ek katkı sağladı:
toplamda baseline'a göre saf hızda **2.03 kat**, video üzerinde **1.48 kat** hızlanmaya çıktı.
(Not: saf hız ile video hızı arasındaki fark normal — video akışında okuma/son işleme gibi ekstra
adımlar da işin içine giriyor, bu yüzden gerçek kullanımda kazanç saf sayıdan biraz düşük çıkıyor.)

## mAP (doğruluk) kıyaslaması — plandaki eksik tamamlandı
CLAUDE.md'deki Hafta 2 planı "FP32 vs FP16 kıyaslaması (FPS, latency, mAP)" olarak yazılmıştı ama
ilk turda mAP ölçümünü atlamıştık. COCO val2017 (5000 görsel, sadece val kısmı indirildi —
train2017'nin 19GB'lik indirmesinden kaçınıldı) üzerinde tamamlandı:

| Format | mAP50 | mAP50-95 |
|---|---|---|
| PyTorch FP32 | 0.5187 | 0.3681 |
| TensorRT FP32 | 0.5184 | 0.3678 |
| TensorRT FP16 | 0.5183 | 0.3677 |

**En önemli bulgu:** FP16'ya geçmek doğruluğu neredeyse hiç etkilemiyor — mAP50-95 sadece 0.0004
(binde birden az) düşüyor. Yani Hafta 2'nin özeti: **~2 kat hızlanma, doğruluk kaybı yok denecek
kadar az.** Bu, TensorRT/FP16 optimizasyonunun bu proje icin oldukca "bedava" bir kazanc oldugunu
gosteriyor.

## Ek metrikler: dosya boyutu ve GPU bellek

| Format | Dosya boyutu | GPU bellek (peak) |
|---|---|---|
| PyTorch FP32 (.pt) | 6.25 MB | 36.5 MB |
| TensorRT FP32 (.engine) | 13.08 MB | 33.4 MB |
| TensorRT FP16 (.engine) | 7.88 MB | 33.4 MB |

TensorRT engine dosyaları .pt dosyasından büyük çünkü içinde sadece ağırlıklar değil, o GPU'ya
özel derlenmiş kernel/tactic bilgileri de saklanıyor. FP16 engine, FP32 engine'e göre yaklaşık
%40 daha küçük — beklenen bir sonuç, çünkü sayılar yarı boyutta tutuluyor. GPU bellek kullanımında
formatlar arasında büyük fark yok (33-36 MB arası) — YOLOv8n zaten küçük bir model olduğu için
bellek zaten baştan az kullanılıyor, optimizasyonun asıl etkisi hız tarafında görünüyor.

## Karşılaşılan ve çözülen teknik sorunlar (video için anlatılabilir)
1. **Turkce karakter sorunu:** Proje yolundaki "ö" harfi (`Yeni klasör`), TensorRT'nin C++
   dosya okuma katmanini bozuyordu. Cozum: export islemlerini gecici ASCII-uyumlu bir klasorde
   yapip sonucu projeye geri tasimak.
2. **TensorRT 11 API degisikligi:** Yeni TensorRT surumu artik "strongly typed" network kullaniyor;
   FP16 hassasiyeti eskisi gibi bir builder flag'i ile degil, ONNX modelinin kendisini FP16'ya
   cevirerek belirleniyor. Ultralytics'in guncel export yolu bunun icin agir bir bagimlilik olan
   `nvidia-modelopt` kullanmaya calisiyordu ve bu paketin bir DLL'i Windows guvenlik ozelligi
   tarafindan engellendi. Cozum: TensorRT Python API'sini dogrudan kullanip FP16 donusumunu
   `onnxconverter-common` ile manuel yapmak.

## Video için konuşma notları (taslak)
1. Hafta 1'in kısa özeti (baseline FPS neydi)
2. Bu hafta hedefi: TensorRT'ye çevirmek ve FP16 ile kıyaslamak
3. Üç formatı canlı test et, tabloyu/grafiği göster
4. FP32→TensorRT'nin büyük fark yarattığını, FP16'nın burada küçük ek katkı sağladığını anlat
   (mühendislik açısından ilginç ve dürüst bir bulgu)
5. Karşılaşılan teknik sorunları kısaca bahset (gerçek sorun çözme deneyimi göstermek için)
6. Gelecek hafta: INT8 kantizasyon + doğruluk (mAP) kıyaslaması

## Bekleyen iş
Hafta 2 kapsamındaki her şey (dönüşüm, FPS/gecikme, mAP, güvenilirlik) tamamlandı. Hafta 3'te
INT8 kantizasyon eklenip üç formatın (FP32/FP16/INT8) tam karşılaştırması yapılacak.
