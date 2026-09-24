# Canlıya Alma Rehberi

## Mevcut mimarinin dağıtıma etkisi

Uygulama Streamlit ile geliştirilmiştir. PyTorch modeli CPU veya CUDA ile çalışabilir;
TensorRT engine dosyaları ise NVIDIA GPU, CUDA ve uyumlu TensorRT sürümü gerektirir.
Mevcut `.engine` dosyaları Windows ve RTX 5060 ortamında oluşturulduğu için başka
bir işletim sistemi/GPU/TensorRT ortamına doğrudan taşınmamalıdır.

## Seçenek A — CPU tanıtım sürümü

Amaç, hocanın ve hakemlerin kurulum yapmadan arayüze erişebilmesidir. Bu sürümde
yalnızca `yolov8n.pt` kullanılır. TensorRT karşılaştırma sonuçları arayüzde
referans tablo olarak gösterilebilir ancak canlı TensorRT çalıştırılmaz.

Adımlar:

1. Projeyi bir Git deposuna dönüştür.
2. `venv/`, COCO veri seti, `runs/`, yüklenen videolar ve geçici çıktıları depoya alma.
3. Kaynak kodu GitHub'a gönder.
4. Streamlit Community Cloud'da yeni uygulama oluştur.
5. Uygulama girişini `src/app.py` olarak ayarla. Bu konum nedeniyle servis,
   CPU ortamı için hazırlanan `src/requirements.txt` dosyasını kullanır.
   Uygulama ilk açılışta eksik `yolov8n.pt` dosyasını Ultralytics kaynağından
   indirir; bulut servisinin ilk başlangıcı bu nedenle daha uzun sürebilir.
6. Örnek video dosyasının lisansını doğrula veya yeniden kullanıma uygun bir
   örnekle değiştir.
7. Dosya yükleme, video işleme, indirme ve webcam akışını HTTPS adresinde test et.

Community Cloud GitHub deposuna erişim ister, Debian Linux üzerinde çalışır ve
GPU sağlamaz. Bu nedenle bu seçenek yalnızca PyTorch CPU tanıtımı içindir.

Avantajı: Hızlı ve düşük maliyetlidir. Dezavantajı: Makalenin ana konusu olan
TensorRT modelleri canlı olarak çalışmaz ve CPU'da video işleme yavaş olabilir.

## Seçenek B — Tam NVIDIA GPU sürümü

Tam işlevli yayın için NVIDIA GPU'lu Linux sunucu gerekir.

Adımlar:

1. CUDA/TensorRT destekli bir NVIDIA GPU sunucusu oluştur.
2. Sürücü, CUDA, cuDNN, TensorRT ve Python sürümlerini kaydet.
3. `yolov8n.pt` veya ONNX modelini sunucuya aktar.
4. `python src/prepare_models.py --engines` komutuyla FP32 ve FP16 engine'lerini
   **hedef GPU sunucusunda yeniden oluştur**.
5. `python src/prepare_models.py --download-coco --int8` komutuyla kalibrasyon
   verisini indir ve INT8 engine'ini oluştur.
6. Engine'lerin mAP ve FPS değerlerini hedef sunucuda tekrar doğrula.
7. Streamlit uygulamasını bir servis veya container olarak başlat.
8. Ters proxy üzerinden HTTPS, alan adı, istek boyutu ve zaman aşımı ayarlarını yap.
9. Geçici video dosyaları için otomatik silme ve disk kotası uygula.
10. Aynı anda birden fazla kullanıcının GPU belleğini tüketmesini önlemek için
    kuyruk/eşzamanlılık sınırı ekle.
11. Uygulamayı mobil/tarayiıcı, yükleme, webcam, iptal ve hata senaryolarıyla test et.

## En hızlı geçici tam demo

Yerel RTX 5060 bilgisayarındaki uygulama HTTPS tüneli üzerinden geçici olarak
paylaşılabilir. Bu yöntem kalıcı yayın değildir: bilgisayar açık kalmalı, internet
bağlantısına bağımlıdır ve herkese açık dosya yükleme güvenlik/kaynak riski yaratır.
Bu nedenle yalnızca kontrollü hoca demosunda, tahmin edilmesi zor bir adres ve kısa
süreli oturumla kullanılmalıdır.

## Yayın öncesi kontrol listesi

- [x] Tekrarlanabilir yerel kurulum ve model hazırlama komutları mevcut
- [x] Uygulama yeniden başlatıldığında açılıyor
- [ ] HTTPS etkin
- [ ] Webcam için tarayıcı izni çalışıyor
- [ ] Bozuk ve büyük dosya hataları kontrollü
- [ ] Geçici dosyalar temizleniyor
- [ ] Aynı anda iki kullanıcı senaryosu test edildi
- [x] Model/engine sürüm bilgileri kayıtlı
- [ ] Örnek video ve model lisansları belgelenmiş
- [ ] Kişisel veri içeren yüklemeler için uyarı metni eklenmiş

Uygulamada ilk korumalar uygulanmıştır: video yüklemeleri 200 MB ve 5 dakika ile
sınırlıdır, model inference çağrıları oturumlar arasında kilitlenir ve altı saatten
eski geçici iş klasörleri otomatik temizlenir.

## Kaynaklar

- Streamlit Community Cloud dosya yapısı:
  https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization
- Streamlit Community Cloud bağımlılıkları:
  https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- NVIDIA TensorRT engine uyumluluğu:
  https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/support-matrix.html
