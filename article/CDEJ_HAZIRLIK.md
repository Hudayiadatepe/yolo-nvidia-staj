# CDEJ Makale Hazırlığı

## Önerilen başlık

**Akıllı Gözetim Sistemlerinde YOLOv8n Çıkarımının NVIDIA TensorRT ile
Optimizasyonu: FP32, FP16 ve INT8 Hassasiyetlerinin Deneysel Karşılaştırması**

English title:

**Optimization of YOLOv8n Inference with NVIDIA TensorRT for Smart Surveillance
Systems: An Experimental Comparison of FP32, FP16, and INT8 Precisions**

## Araştırma sorusu

Tüketici sınıfı yeni nesil bir NVIDIA GPU'da YOLOv8n için TensorRT FP32, FP16 ve
INT8 kullanımı; uçtan uca FPS, saf çıkarım hızı, model boyutu ve COCO mAP
bakımından nasıl bir hız-doğruluk dengesi oluşturmaktadır?

## Özgün katkı iddiası

- RTX 5060 Laptop GPU üzerinde YOLOv8n için tekrarlı ve hata paylı benchmark.
- PyTorch FP32, TensorRT FP32, TensorRT FP16 ve iki farklı INT8 kalibrasyonunun
  aynı deney düzeninde karşılaştırılması.
- INT8'in her donanım/model bileşiminde en iyi seçenek olmadığını gösteren negatif
  fakat uygulama açısından değerli bulgu.
- Bulguların video ve webcam destekli bir prototipte gösterilmesi.

## Önerilen bölümler

1. Öz
2. Abstract
3. Giriş
4. İlgili Çalışmalar
5. Materyal ve Yöntem
   - Donanım ve yazılım ortamı
   - YOLOv8n ve COCO val2017
   - ONNX ve TensorRT dönüşümü
   - FP32, FP16 ve INT8 yapılandırmaları
   - Kalibrasyon stratejileri
   - Benchmark protokolü ve metrikler
6. Bulgular
   - Saf çıkarım performansı
   - Uçtan uca video performansı
   - Doğruluk sonuçları
   - Model boyutu ve GPU bellek kullanımı
7. Tartışma
8. Sınırlılıklar ve Geçerlilik Tehditleri
9. Sonuç
10. Etik Beyanlar ve Üretken Yapay Zekâ Kullanım Beyanı
11. Kaynakça

## Doğrulanmış ana bulgu

24 Eylül 2026 tarihinde tek ve sabit protokolle alınan sonuçlarda TensorRT FP16,
PyTorch FP32'ye göre saf çıkarımda 3,26 kat ve video hattında 2,01 kat hızlanmıştır.
mAP50-95 değeri 0,3681'den 0,3678'e değişmiştir. TensorRT INT8 ise 0,3272
mAP50-95 ile belirgin doğruluk kaybı göstermiş ve FP16'dan daha yavaş kalmıştır.
Bu donanım ve model bileşiminde FP16 en dengeli seçenektir.

## Dergi uyum kontrolü

- Makale Türkçe veya İngilizce olabilir.
- Word biçiminde yazar bilgili bir dosya hazırlanmalıdır.
- Çift kör hakemlik için yazar bilgileri kaldırılmış ikinci Word dosyası gerekir.
- Telif hakkı bildirim formu PDF olarak yüklenmelidir.
- Kaynakça hariç benzerlik oranı en fazla %25 olmalıdır.
- Gönderim e-posta ile değil DergiPark üzerinden yapılır.
- Makale danışman kontrolü tamamlanmadan gönderilmemelidir.
- Kullanılan üretken yapay zekâ aracı, sürümü ve kullanım amacı Yöntem veya
  Teşekkür bölümünde açıkça beyan edilmelidir.
- Yapay zekâ tarafından oluşturulan metin yalnızca taslak kabul edilmeli; bilimsel
  yorum, doğrulama, kaynak kontrolü ve nihai anlatım yazar tarafından yapılmalıdır.
- Dergi kapsamıyla bağ açık kurulmalıdır: çalışma, akıllı gözetim ve fiziksel
  güvenlik sistemlerinde uç yapay zekâ çıkarımının gecikme/doğruluk dengesi olarak
  çerçevelenmiştir.
- Kullanılacak resmî dosya `template/CDEJ_template_TR_official_2026.docx` dosyasıdır.
  `template/CDEJ_resmi_sablon.docx` farklı bir dergiye ait olduğundan kullanılmamalıdır.

## Makaleye başlamadan önce eksik bilgiler

- Staj yapılan kurum ve tarih aralığı
- Nihai deneylerde kullanılan tam test videosu ve kaynağı/lisansı
- Dizüstü bilgisayarın deney sırasındaki güç/performans modu
- Uygulamanın canlı adresi ve kaynak kod deposu

## Yazar bilgileri

- Hüdayi Hamza Adatepe
- Düzce Üniversitesi, Bilgisayar Mühendisliği Bölümü
- hudayi224160@ogr.duzce.edu.tr
- ORCID: 0009-0005-2389-3061
- GitHub: HudayiAdatepe
