# Hafta 2 Video Kaydı — Konuşma Metni

Toplam süre tahmini: ~3-4 dakika.

---

## 1) Giriş (20 sn)

> "Merhaba hocam, ben [adın]. Bu, Hafta 2 video teslimim. Geçen hafta baseline YOLOv8 hızını
> ölçmüştüm, bu hafta hedefim modeli NVIDIA TensorRT'ye çevirip FP32 ve FP16 hassasiyetlerini
> hem hız hem doğruluk açısından karşılaştırmaktı."

---

## 2) Dönüşüm süreci (30 sn)

> "Önce YOLOv8n modelini ONNX formatına çevirdim, sonra ONNX'ten iki ayrı TensorRT engine
> oluşturdum: biri FP32, biri FP16 hassasiyette. TensorRT, modeli GPU'ya özel optimize ediyor —
> katmanları birleştiriyor ve donanıma özel en hızlı hesaplama yolunu seçiyor."

`[EKRANDA: models/ klasörünü göster — yolov8n.pt, .onnx, yolov8n_fp32.engine, yolov8n_fp16.engine]`

---

## 3) Hız karşılaştırması (1 dk)

> "Üç formatı da hem sahte bir görüntüyle (saf model hızı) hem gerçek bir video üzerinde uçtan
> uca test ettim. Güvenilir sonuç almak için her formatı tek seferlik değil, 10 kez ayrı ayrı
> ölçüp ortalamasını aldım — grafikteki hata çubukları bu ölçümlerin ne kadar tutarlı olduğunu
> gösteriyor."

`[EKRANDA: results/week2_fps_comparison.png grafiğini göster]`

> "Sonuçlar: PyTorch'un ham hali videoda ortalama 103 FPS veriyordu. TensorRT'ye çevirince bu
> 140 FPS'e, FP16'ya geçince 153 FPS'e çıktı — yaklaşık 1.5 kat hızlanma, saf model hızında ise
> 2 kata yakın bir kazanç var. İlginç bir yan bulgu: ilk denemede ölçümler çalıştırmadan
> çalıştırmaya oldukça değişkendi; örneklem sayısını ve video uzunluğunu artırınca sapma çok
> daraldı — yani güvenilir bir ölçüm için tek seferlik test yeterli değilmiş, bunu da öğrendim."

---

## 4) Doğruluk (mAP) karşılaştırması — bu haftanın en kritik kısmı (1 dk)

> "Hızlanma tek başına yeterli değil — optimizasyonun doğruluğu ne kadar bozduğunu da ölçmem
> gerekiyordu. Bunun için COCO val2017 veri setinin 5000 görsellik doğrulama kümesini indirip
> her üç formatta da mAP ölçtüm."

`[EKRANDA: results/week2_map_comparison.json veya tabloyu göster]`

> "Sonuç oldukça çarpıcı: PyTorch'ta mAP50-95 0.3681 iken, TensorRT FP16'da 0.3677 — yani
> doğruluk kaybı binde birden az. Yaklaşık 2 kat hızlanmayı, neredeyse hiç doğruluk kaybetmeden
> elde ettim."

---

## 5) Ek metrikler ve karşılaşılan sorunlar (40 sn)

> "Ayrıca dosya boyutu ve GPU bellek kullanımını da karşılaştırdım — TensorRT engine dosyaları
> PyTorch dosyasından biraz büyük çünkü içinde GPU'ya özel derlenmiş kernel bilgisi de saklanıyor,
> ama FP16 engine FP32'den yaklaşık %40 daha küçük."

> "Bu hafta iki teknik sorunla uğraştım: proje klasör adımdaki Türkçe karakter TensorRT'nin
> dosya okumasını bozuyordu, ve TensorRT'nin yeni sürümü FP16'yı artık farklı bir yöntemle
> istiyordu. İkisini de çözdüm."

---

## 6) Kapanış (20 sn)

> "Özetle Hafta 2: TensorRT'ye geçişle ~1.5-2 kat hızlanma, doğruluk kaybı ihmal edilebilir
> düzeyde, üstüne performans daha kararlı hale geldi. Gelecek hafta INT8 kantizasyonla üçüncü
> bir hassasiyet seviyesini de ekleyip tüm formatları birlikte karşılaştıracağım. Teşekkürler."

---

## Kayıt öncesi kontrol listesi
- [ ] `results/week2_fps_comparison.png` önceden açık sekme olarak hazır
- [ ] `results/week2_size_memory_comparison.png` hazır
- [ ] `results/annotated_video/solutions_ci_demo.avi` istersen kısa bir kesit gösterebilirsin
- [ ] Terminal büyük/okunaklı
