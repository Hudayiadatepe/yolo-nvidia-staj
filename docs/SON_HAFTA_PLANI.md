# Son Hafta Çalışma Planı

Bu plan, danışmanın dördüncü hafta geri bildirimindeki istekleri uygulanabilir
iş paketlerine dönüştürür.

## 1. Uygulama iyileştirmeleri

### Öncelik 1 — Teslim edilebilirlik

- Proje kurulumunu temiz bir ortamda tekrarlanabilir hâle getir.
- Yerel geliştirme ve canlı ortam bağımlılıklarını ayır.
- Sabit bilgisayar yollarını kaldır.
- Uygulamayı CPU ortamında PyTorch tanıtım moduna düşürebilir hâle getir.
- NVIDIA GPU olmayan ortamda TensorRT modellerini seçim listesinden kaldır.
- Kurulum, çalıştırma, deneyleri tekrarlama ve sınırlılıkları README'de belgeleyerek teslimi kolaylaştır.

### Öncelik 2 — Kullanıcı deneyimi

- Model yükleme ve video işleme hatalarını kullanıcıya anlaşılır biçimde göster.
- Maksimum dosya boyutu ve desteklenen biçimleri açıkça belirt.
- CPU tanıtım modu ile tam NVIDIA GPU modu arasındaki farkı arayüzde göster.
- Sonuç ekranına gecikme ve toplam işleme süresi eklemeyi değerlendir.

### Öncelik 3 — Bilimsel tekrarlanabilirlik

- Tüm benchmark sonuçlarına tarih, donanım, yazılım sürümü, video kimliği,
  kare sayısı, görüntü boyutu ve tekrar sayısını ekle.
- Deneyleri aynı test videosu ve aynı koşullarla yeniden çalıştır.
- FPS yanında ortalama gecikme, standart sapma ve hızlanma oranını raporla.
- INT8 v1 ve v2 sonuçlarını tek bir nihai tabloda birleştir.

## 2. Canlıya alma

Canlıya alma iki ayrı hedef olarak ele alınmalıdır:

1. **CPU tanıtım sürümü:** Herkese açık bağlantı; yalnızca PyTorch modeli.
2. **Tam GPU sürümü:** PyTorch ve TensorRT FP32/FP16/INT8 modelleri; NVIDIA GPU'lu
   sunucu ve hedef sunucuda yeniden oluşturulan TensorRT engine dosyaları.

Ayrıntılı uygulama adımları `docs/CANLIYA_ALMA.md` dosyasındadır.

## 3. Makale

Makalenin önerilen ana katkısı:

> YOLOv8n nesne tespit modelinin tüketici sınıfı yeni nesil bir NVIDIA GPU'da
> FP32, FP16 ve iki farklı INT8 kalibrasyon stratejisiyle gerçek zamanlı performans
> ve doğruluk açısından deneysel olarak karşılaştırılması.

CDEJ kapsamıyla uyum için çalışma, genel bir YOLO demosu olarak değil,
**akıllı gözetim ve siber-fiziksel sistemlerde gerçek zamanlı uç yapay zekâsı**
bağlamında sunulmalıdır. Bu bağlam makalede yapay biçimde eklenmemeli; amaç,
kullanım senaryosu, sınırlılıklar ve sonuçlarla gerçekten desteklenmelidir.

## 4. Teslim sırası

1. Deney koşullarını sabitle ve nihai benchmark'ı al.
2. Uygulama iyileştirmelerini ve dağıtım dosyalarını tamamla.
3. CPU tanıtım sürümünü yayınla; bütçe/altyapı uygunsa GPU sürümünü yayınla.
4. Staj raporunu tamamla.
5. Aynı deneylerden, CDEJ şablonunda ayrı bir bilimsel makale hazırla.
6. Yazar bilgili ve kör hakemlik için anonim iki Word dosyası oluştur.
7. Danışman kontrolünden sonra benzerlik raporunu al ve telif formunu tamamla.
8. Danışmanın açık onayından sonra DergiPark'a gönder.
