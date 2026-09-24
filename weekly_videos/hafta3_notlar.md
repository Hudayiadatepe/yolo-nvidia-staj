# Hafta 3 — Video Teslimi Notları

## Bu hafta yapılanlar
- YOLOv8n modeli **TensorRT INT8** engine'ine çevrildi (`models/yolov8n_int8.engine`),
  kalibrasyon için COCO val2017'den 512 görsel kullanıldı (Ultralytics'in ModelOpt tabanlı
  kantizasyon akışı üzerinden).
- Dört format da (PyTorch FP32, TensorRT FP32, TensorRT FP16, TensorRT INT8) aynı koşullarda
  FPS, gecikme, dosya boyutu, GPU bellek ve mAP açısından karşılaştırıldı.
- Grafikler üretildi: `results/week3_fps_comparison.png`, `results/week3_size_memory_comparison.png`

## Sonuçlar (10 tekrarın ortalaması ± std sapma)

| Format | Saf FPS | Video FPS | Dosya | mAP50-95 |
|---|---|---|---|---|
| PyTorch FP32 | 91.61 ± 5.55 | 73.7 ± 12.49 | 6.25 MB | 0.3681 |
| TensorRT FP32 | 242.25 ± 16.43 (2.64x) | 130.93 ± 7.08 (1.78x) | 13.08 MB | 0.3678 |
| TensorRT FP16 | 259.03 ± 18.44 (2.83x) | 156.83 ± 5.21 (2.13x) | 7.88 MB | 0.3677 |
| TensorRT INT8 | 258.9 ± 12.09 (2.83x) | 146.83 ± 2.1 (1.99x) | 6.59 MB | **0.3272** |

## En önemli bulgu — INT8 beklendiği gibi çıkmadı, ve bu değerli bir sonuç
İki şey dikkat çekici:

1. **INT8, FP16'dan daha hızlı değil.** Saf hızda ikisi neredeyse eşit (258.9 vs 259.03 FPS),
   video üzerinde INT8 hatta FP16'dan **daha yavaş** (146.83 vs 156.83 FPS).
2. **INT8 doğrulukta belirgin kayıp veriyor.** mAP50-95, FP16'da 0.3677 iken INT8'de 0.3272'ye
   düşüyor — yaklaşık **%11 bağıl kayıp**. FP16'nın getirdiği (binde birden az) kayıpla
   kıyaslanamayacak kadar büyük.

**Yorum:** Bu proje için (YOLOv8n gibi zaten küçük bir model, RTX 5060 gibi FP16 tensor
çekirdekleri güçlü bir GPU üzerinde) INT8 kantizasyonun ekstra bir hız kazancı getirmediği,
üstüne doğruluktan ciddi ödün verdirdiği görülüyor. Yani bu donanım/model kombinasyonunda
**FP16, INT8'e göre daha iyi bir "tatlı nokta" (sweet spot)** — daha hızlı olmasa bile eşit
hızda çok daha az doğruluk kaybıyla çalışıyor. Bu, "her zaman en agresif optimizasyon en iyisi
değildir" şeklinde raporda vurgulanabilecek, gerçek ölçümle desteklenen dürüst bir mühendislik
sonucu.

(Not: Daha büyük bir modelde — YOLOv8m/l/x gibi — veya farklı bir GPU'da INT8'in avantajı daha
belirgin çıkabilir; bu sonuç bu spesifik model+donanım kombinasyonuna özgü.)

## İyileştirme denemesi: farklı kalibrasyon yöntemi
İlk INT8 sonucunun beklenenden kötü çıkması üzerine, kalibrasyonu iyileştirip iyileştiremeyeceğimizi
test ettik. Ultralytics'in varsayılanı `calibration_method="max"` + 512 görsel idi; bunu
`modelopt`'un kendi varsayılanı olan `"entropy"` (TensorRT'nin klasik KL-divergence yöntemi) ve
1000 görsele çıkararak tekrar denedik (`src/export_int8_v2.py`).

| Format | Saf FPS | Video FPS | Dosya | mAP50-95 |
|---|---|---|---|---|
| TensorRT FP16 | 259.03 | **156.83** | 7.88 MB | **0.3677** |
| TensorRT INT8 v1 (max, 512 görsel) | **258.9** | 146.83 | 6.59 MB | 0.3272 |
| TensorRT INT8 v2 (entropy, 1000 görsel) | 219.17 | 137.53 | 6.57 MB | 0.3511 |

**Sonuç:** Entropy yöntemi doğruluğu gerçekten iyileştirdi (mAP50-95: 0.3272 → 0.3511, FP16'ya
olan fark %11'den %4.5'e indi) ama bu sefer **hızdan** ödün verdi — v2, hem v1'den hem FP16'dan
daha yavaş çıktı. Modelin toplam 731 düğümünden sadece 202'sinin (%28) kantize edilmesi her iki
denemede de değişmedi — bu, modelin/grafiğin yapısal bir özelliği, kalibrasyon yöntemiyle
değişmiyor.

**Nihai karar:** İki farklı INT8 stratejisini de denedik, ikisi de **FP16'yı hiçbir eksende
(ne hızda ne doğrulukta) geçemedi**. Bu artık tesadüf değil, tutarlı bir bulgu: **bu proje için
doğru seçim FP16.** INT8'i daha fazla zorlamak (farklı kalibrasyon yöntemleri, daha fazla görsel)
muhtemelen benzer sonuçlar verecektir — zaman ve emek/kazanç dengesi açısından mantıklı değil,
bu yüzden Hafta 4'e FP16 varsayılan olarak taşınıyor.

## Karşılaşılan teknik durum
Geçen hafta FP16 için karşılaştığımız `modelopt`/`lief` DLL engelleme sorunu bu hafta kendiliğinden
çözülmüştü (Windows'un Akıllı Uygulama Denetimi zamanla paketi güvenilir olarak tanımış olmalı),
bu yüzden INT8 için Ultralytics'in kendi yerleşik kantizasyon akışını (`quantize=8, data=...`)
doğrudan kullanabildik — ekstra bir workaround gerekmedi.

## Video için konuşma notları (taslak)
1. Hafta 2 özeti (FP32/FP16 sonuçları)
2. Bu hafta: INT8 kantizasyon eklendi, kalibrasyon COCO val2017 ile yapıldı
3. 4 formatı karşılaştıran grafiği göster
4. **Ana bulguyu vurgula:** INT8 hızda ek kazanç sağlamadı ama doğrulukta ciddi kayıp verdi —
   bu proje için FP16'nın daha iyi seçim olduğu sonucuna varıldı
5. Gelecek hafta: masaüstü/web arayüzü — model ve optimizasyon seviyesi seçilebilir, canlı FPS
   gösterimi

## Bekleyen iş
Hafta 3 tamamlandı — INT8 iki farklı yöntemle denendi, ikisinde de FP16 daha iyi çıktı. Hafta 4'te
arayüz geliştirmeye geçilecek, varsayılan/önerilen optimizasyon seviyesi **FP16** olacak (bu
haftanın bulgusuna dayanarak), ama kullanıcı isterse INT8'i de seçip kendi gözüyle kıyaslayabilecek
(arayüzde tüm formatları seçilebilir tutmak, bu bulguyu canlı olarak gösterme fırsatı da verir).
