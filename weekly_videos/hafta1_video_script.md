# Hafta 1 Video Kaydı — Konuşma Metni

Toplam süre tahmini: ~4-5 dakika. Ekran paylaşımıyla kaydet (terminal + VS Code/editör
açık olsun). Parantez içindeki `[EKRANDA: ...]` notları o an ekranda ne göstereceğini
söylüyor.

---

## 1) Giriş (30 sn)

> "Merhaba hocam, ben [adın]. Bu, YOLO ve NVIDIA performans analizi stajımın birinci
> hafta videosu. Bu hafta hedefim, geliştirme ortamını kurmak ve YOLOv8 modelinin
> optimizasyonsuz, yani 'baseline' halinin ne kadar hızlı çalıştığını ölçmekti —
> çünkü önümüzdeki haftalarda TensorRT ve farklı hassasiyet formatlarıyla
> (FP16, INT8) yapacağım optimizasyonları bu referans sayıyla karşılaştıracağım."

`[EKRANDA: proje klasörü — README.md ve CLAUDE.md açık, 5 haftalık planı kısaca göster]`

---

## 2) Ortam kurulumu (1 dk)

> "İlk iş geliştirme ortamını kurmaktı. Python için izole bir sanal ortam oluşturdum,
> ve içine PyTorch, Ultralytics YOLOv8 kütüphanesi ve OpenCV gibi gerekli paketleri
> kurdum."

> "Burada bir detay var: benim GPU'm RTX 5060, bu NVIDIA'nın en yeni Blackwell
> mimarisini kullanıyor. PyTorch'un standart sürümü bu mimariyi henüz tanımıyor,
> bu yüzden CUDA 12.8 destekli özel bir PyTorch derlemesi kurmam gerekti. Bu,
> donanım-yazılım uyumluluğunun projenin en başından itibaren dikkat edilmesi
> gereken bir konu olduğunu gösterdi."

`[EKRANDA: requirements.txt dosyasını göster, sonra terminalde]`
```bash
venv\Scripts\activate
pip list | findstr torch
```

---

## 3) Ortam doğrulama (1 dk)

> "Kurulumun doğru çalıştığını kontrol etmek için bir doğrulama scripti yazdım:
> check_env.py. Bunu çalıştırdığımda PyTorch'un GPU'yu gördüğünü, CUDA ve cuDNN
> sürümlerini ve GPU bilgilerini gösteriyor."

`[EKRANDA: terminalde canlı çalıştır]`
```bash
python src\check_env.py
```

> "Görüldüğü gibi: PyTorch 2.11, CUDA 12.8, cuDNN sürümü aktif, ve GPU olarak
> RTX 5060 Laptop GPU, 8GB bellek, compute capability 12.0 olarak tanınıyor.
> Yani donanım hızlandırma kullanıma hazır."

---

## 4) Baseline YOLOv8 inference ve FPS ölçümü (1.5 dk)

> "Sonra baseline_inference.py scriptini yazdım. Bu script üç şey yapıyor:
> Birincisi, pretrained YOLOv8n ağırlıklarını indirip yüklüyor. İkincisi, örnek
> bir görsel üzerinde nesne tespiti çalıştırarak modelin doğru çalıştığını
> doğruluyor. Üçüncüsü, sabit boyutlu görüntülerle 100 kere art arda inference
> çalıştırıp ortalama hızı — yani FPS ve gecikme süresini — ölçüyor."

`[EKRANDA: terminalde canlı çalıştır]`
```bash
python src\baseline_inference.py
```

> "Örnek görselde 6 nesne doğru tespit edildi — [EKRANDA: results/week1_sample_detection.jpg
> dosyasını aç ve göster, otobüs + yayalar üzerindeki kutucukları işaret et]."

> "Ve asıl önemli sayı: saf model hızı 182.79 FPS, yani model saniyede ortalama
> 183 kare işleyebiliyor, her kare 5.47 milisaniyede işleniyor. Bu, hiçbir
> optimizasyon yapılmamış, saf PyTorch FP32 modelinin hızı."

> "Ayrıca gerçek bir video üzerinde de test ettim — okuma, tespit ve çizim
> dahil, yani uçtan uca hız. Bu sefer sonuç 76.19 FPS çıktı. Görüldüğü gibi
> gerçek kullanımda hız biraz düşüyor, çünkü kamera/video okuma ve görsel
> işleme adımları da devreye giriyor. Bu iki sayıyı önümüzdeki haftalarda
> TensorRT dönüşümü ve FP16/INT8 kantizasyonuyla ne kadar artırabildiğimi
> göstermek için referans olarak kullanacağım."

`[EKRANDA: results/week1_baseline_results.json dosyasını aç, sayıları göster]`

---

## 5) Kapanış ve gelecek hafta planı (30 sn)

> "Özetle bu hafta: geliştirme ortamını kurdum, GPU'nun doğru tanındığını
> doğruladım, ve YOLOv8'in optimizasyonsuz baseline hızını ölçtüm — saf model
> hızı 182.79 FPS, gerçek video üzerinde uçtan uca 76.19 FPS, RTX 5060 üzerinde
> FP32 hassasiyetle."

> "Gelecek hafta planım: modeli ONNX formatına, oradan da TensorRT motoruna
> dönüştürmek, ve FP32 ile FP16 hassasiyet formatlarını FPS, gecikme ve
> doğruluk (mAP) açısından karşılaştırmak."

> "Sorularım varsa: [CLAUDE.md'deki açık soruları buraya ekle — YOLO versiyonu,
> masaüstü/web önceliği, özel veri seti gerekip gerekmediği vb.] Teşekkürler,
> görüşmek üzere."

---

## Kayıt öncesi kontrol listesi
- [ ] Terminal ve editör ekranı temiz/okunaklı (yazı boyutunu büyüt)
- [ ] `results/week1_sample_detection.jpg` önceden açılmış sekme olarak hazır
- [ ] Mikrofon test edildi
- [ ] Ekran kaydı: Windows'ta `Win + G` (Xbox Game Bar) ile başlatılabilir,
      ya da OBS Studio kullanılabilir
