# Hafta 3 Video Kaydı — Konuşma Metni

Toplam süre tahmini: ~3-4 dakika.

---

## 1) Giriş (20 sn)

> "Merhaba hocam, ben [adın]. Bu, Hafta 3 video teslimim. Geçen hafta FP32 ve FP16'yı
> karşılaştırmıştım, bu hafta hedefim üçüncü bir hassasiyet seviyesi olan INT8 kantizasyonu
> eklemek ve dört formatı (FP32, TensorRT FP32, FP16, INT8) birlikte karşılaştırmaktı."

---

## 2) INT8 dönüşümü (30 sn)

> "INT8, sayıları 8-bit tam sayıya yuvarlayan en agresif sıkıştırma yöntemi. Bunun için modelin
> hangi değer aralıklarında çalıştığını öğrenmesi gerekiyor — bu yüzden COCO val2017'den 512
> görsellik bir kalibrasyon kümesiyle modeli 'kalibre' ettim, sonra TensorRT INT8 engine'ini
> oluşturdum."

`[EKRANDA: models/yolov8n_int8.engine dosyasını göster]`

---

## 3) Dört formatın karşılaştırması (1 dk)

> "Dört formatı da 10 tekrarla, hem saf hız hem gerçek video üzerinde test ettim."

`[EKRANDA: results/week3_fps_comparison.png grafiğini göster]`

> "Beklentim INT8'in en hızlısı olmasıydı — teoride öyle olması gerekiyor. Ama sonuç şaşırtıcı
> çıktı: INT8, FP16 ile neredeyse aynı hızda, hatta gerçek video testinde FP16'dan biraz daha
> yavaş."

---

## 4) Doğruluk karşılaştırması — asıl kritik nokta (1 dk)

`[EKRANDA: results/week3_map_comparison.json veya tabloyu göster]`

> "Doğruluğa bakınca durum netleşti: FP16'da mAP50-95 0.3677 iken, INT8'de 0.3272'ye düşüyor —
> yaklaşık %11'lik bir kayıp. Yani INT8 hem hız avantajı sağlamıyor hem de belirgin doğruluk
> kaybı getiriyor."

> "Bunun neden olduğunu araştırdım: modelin toplam 731 hesaplama düğümünden sadece 202'si,
> yani yaklaşık dörtte biri gerçekten INT8'e çevrilebiliyor — geri kalanı FP16/FP32'de kalıyor.
> Model tam INT8 değil, karma bir hal alıyor, bu da hem hız kazancını sınırlıyor hem doğruluğu
> düşürüyor."

---

## 5) İyileştirme denemesi (40 sn)

> "Bunu düzeltebilir miyim diye, farklı bir kalibrasyon yöntemi de denedim — varsayılan 'max'
> yerine 'entropy' yöntemini ve 512 yerine 1000 kalibrasyon görseli kullandım."

`[EKRANDA: sonuç tablosunu göster - v1 vs v2]`

> "Doğruluk gerçekten iyileşti, ama bu sefer hız düştü — yeni versiyon hem eskisinden hem
> FP16'dan yavaş çıktı. Yani iki farklı stratejiyi denedim, ikisinde de FP16 hem hızda hem
> doğrulukta önde kaldı."

---

## 6) Kapanış ve karar (30 sn)

> "Sonuç olarak: bu model — YOLOv8n, zaten küçük bir mimari — ve bu GPU — RTX 5060, FP16
> konusunda zaten çok güçlü — kombinasyonunda INT8'in vaat ettiği ekstra hız kazancı
> gerçekleşmiyor, üstüne doğruluktan ödün veriyor. Bu yüzden bu proje için en iyi seçimin FP16
> olduğuna karar verdim, bunu iki ayrı denemeyle doğruladım.

> Gelecek hafta bu bulguyu kullanarak masaüstü/web arayüzünü geliştireceğim — varsayılan olarak
> FP16 kullanılacak, ama kullanıcı isterse diğer formatları da seçip bu farkı canlı görebilecek.
> Teşekkürler."

---

## Kayıt öncesi kontrol listesi
- [ ] `results/week3_fps_comparison.png` sekme olarak açık
- [ ] `results/week3_size_memory_comparison.png` sekme olarak açık
- [ ] `results/week3_map_comparison.json` açık (4 format) veya elle hazırlanmış tablo
- [ ] Terminal büyük/okunaklı
