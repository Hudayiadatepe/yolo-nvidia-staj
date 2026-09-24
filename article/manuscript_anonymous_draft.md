# Akıllı Gözetim Sistemlerinde YOLOv8n Çıkarımının NVIDIA TensorRT ile Optimizasyonu: FP32, FP16 ve INT8 Hassasiyetlerinin Deneysel Karşılaştırması

## Optimization of YOLOv8n Inference with NVIDIA TensorRT for Smart Surveillance Systems: An Experimental Comparison of FP32, FP16, and INT8 Precisions

## Öz

Akıllı gözetim sistemlerinde nesne tespitinin düşük gecikmeyle gerçekleştirilmesi, olayların zamanında belirlenmesi bakımından önem taşımaktadır. Bu çalışmada YOLOv8n modelinin PyTorch FP32, TensorRT FP32, TensorRT FP16 ve TensorRT INT8 çalışma biçimleri tüketici sınıfı bir NVIDIA GeForce RTX 5060 Laptop GPU üzerinde karşılaştırılmıştır. Hız deneyleri 640×640 piksel girdide, her çalışma biçimi için 10 ısınma yinelemesi, 100 zamanlanmış yineleme ve 10 bağımsız tekrar kullanılarak yürütülmüştür. Doğruluk, COCO val2017 veri kümesindeki 5.000 görüntü üzerinde mAP50 ve mAP50–95 ölçütleriyle değerlendirilmiştir. TensorRT FP16; 342,52±20,03 saf çıkarım FPS ve 147,79±3,29 uçtan uca video FPS değerine ulaşarak PyTorch FP32'ye göre sırasıyla 3,26 ve 2,01 kat hızlanmıştır. mAP50–95 değeri PyTorch FP32 için 0,3681, TensorRT FP16 için 0,3678 olarak ölçülmüştür. TensorRT INT8, 0,3272 mAP50–95 ile belirgin doğruluk kaybı göstermiş ve bu donanımda FP16'dan daha düşük hız üretmiştir. Sonuçlar, incelenen donanım-yazılım bileşiminde FP16'nın gerçek zamanlı akıllı gözetim uygulamaları için en dengeli çalışma noktası olduğunu göstermektedir.

**Anahtar kelimeler:** nesne tespiti, YOLOv8n, TensorRT, karma hassasiyet, akıllı gözetim, uç yapay zekâ

## Abstract

Low-latency object detection is important for timely event identification in smart surveillance systems. This study compares PyTorch FP32, TensorRT FP32, TensorRT FP16, and TensorRT INT8 variants of YOLOv8n on a consumer-grade NVIDIA GeForce RTX 5060 Laptop GPU. Speed experiments used 640×640 input, 10 warm-up iterations, 100 timed iterations, and 10 independent repetitions per variant. Accuracy was evaluated on 5,000 COCO val2017 images using mAP50 and mAP50–95. TensorRT FP16 achieved 342.52±20.03 FPS for model-only inference and 147.79±3.29 FPS for the end-to-end video pipeline, corresponding to 3.26× and 2.01× speedups over PyTorch FP32. The mAP50–95 scores were 0.3681 for PyTorch FP32 and 0.3678 for TensorRT FP16. TensorRT INT8 yielded a marked accuracy reduction to 0.3272 mAP50–95 and was slower than FP16 on the tested hardware. The findings indicate that FP16 provides the most balanced operating point for real-time smart-surveillance workloads in the examined hardware-software configuration.

**Keywords:** object detection, YOLOv8n, TensorRT, mixed precision, smart surveillance, edge AI

## 1. Giriş

Kamera tabanlı fiziksel güvenlik sistemleri, geniş görüntü akışlarının eş zamanlı işlenmesini gerektirir. Derin öğrenme tabanlı nesne tespit modelleri yüksek doğruluk sunabilse de gecikme, işlem gücü ve bellek gereksinimleri uç cihazlarda kullanımı sınırlayabilir. Tek aşamalı YOLO ailesi, tespit işlemini tek bir ağ geçişiyle gerçekleştirerek gerçek zamanlı uygulamalar için elverişli bir yaklaşım sunmaktadır [1].

NVIDIA TensorRT; katman birleştirme, çekirdek seçimi ve düşük hassasiyetli hesaplama gibi optimizasyonlarla eğitilmiş ağların çıkarımını hızlandırmayı amaçlar. Bununla birlikte daha düşük sayısal hassasiyet her sistemde aynı hız kazancını sağlamaz ve özellikle INT8 dönüşümü kalibrasyon verisine bağlı doğruluk kaybı oluşturabilir [3,4]. Bu nedenle yalnızca teorik işlem kapasitesine dayanmak yerine hedef donanım üzerinde hız ve doğruluğun birlikte ölçülmesi gerekir.

Bu çalışmanın araştırma sorusu şöyledir: Tüketici sınıfı yeni nesil bir NVIDIA GPU'da YOLOv8n için TensorRT FP32, FP16 ve INT8 kullanımı; saf çıkarım hızı, uçtan uca video hızı, model boyutu ve COCO doğruluğu bakımından nasıl bir denge oluşturmaktadır? Çalışmanın katkısı, dört çalışma biçimini aynı protokolde tekrarlı ölçümlerle karşılaştırması ve sonuçları webcam/video destekli bir araştırma prototipiyle göstermesidir.

## 2. Materyal ve Yöntem

### 2.1. Donanım ve yazılım ortamı

Deneyler NVIDIA GeForce RTX 5060 Laptop GPU ve 616.92 sürücüsü bulunan Windows 11 sisteminde yapılmıştır. Yazılım ortamı Python 3.12.10, PyTorch 2.7.1+cu128, CUDA 12.8, cuDNN 9.7.1, TensorRT 11.2.1.2, Ultralytics 8.4.127 ve OpenCV 5.0.0 bileşenlerinden oluşmaktadır. GPU'nun CUDA hesaplama yeteneği 12.0'dır.

### 2.2. Model ve veri

Temel model olarak COCO üzerinde önceden eğitilmiş YOLOv8n kullanılmıştır. Doğruluk değerlendirmesi, COCO val2017 bölümündeki 5.000 görüntü üzerinde yapılmıştır [2]. Hız deneylerindeki video hattı 768×576 çözünürlükte, 10 FPS ve 150 kareden oluşan 15 saniyelik bir video kullanmıştır. Video kaynağı/lisansı nihai makalede açıkça belirtilmelidir.

### 2.3. Dönüşüm ve hassasiyet biçimleri

PyTorch FP32 ağırlıkları referans kabul edilmiştir. TensorRT FP32 ve FP16 engine dosyaları ONNX ara gösterimi üzerinden hedef GPU'da oluşturulmuştur. INT8 engine kalibrasyon verisi kullanılarak üretilmiştir. TensorRT engine dosyaları GPU mimarisi ve çalışma zamanı sürümüne bağlı olduğundan başka bir cihaza taşınabilir ikili dosyalar olarak değerlendirilmemiştir.

### 2.4. Ölçüm protokolü

Saf çıkarım deneyinde modelin 1×3×640×640 boyutlu CUDA tensörü üzerindeki tahmin süresi ölçülmüştür. Her bağımsız tekrarda 10 ısınma ve 100 zamanlanmış yineleme uygulanmış; deney 10 kez tekrarlanmıştır. Video deneyinde kod çözme ve model çıkarımı birlikte ölçülmüş, sonuç çizimi ölçümün dışında bırakılmıştır. Video her tekrarda üç kez işlenmiştir. FPS sonuçları ortalama±standart sapma olarak raporlanmıştır. Doğruluk için mAP50 ve COCO tarzı mAP50–95 ölçülmüştür.

## 3. Bulgular

| Çalışma biçimi | Saf FPS | Video FPS | mAP50 | mAP50–95 | Dosya boyutu |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32 | 105,19±5,89 | 73,60±5,06 | 0,5187 | 0,3681 | 6,25 MB |
| TensorRT FP32 | 273,90±13,11 | 137,66±2,02 | 0,5185 | 0,3678 | 13,14 MB |
| TensorRT FP16 | **342,52±20,03** | **147,79±3,29** | 0,5185 | 0,3678 | 7,86 MB |
| TensorRT INT8 | 306,91±13,11 | 142,92±2,68 | 0,4720 | 0,3272 | 6,59 MB |

TensorRT FP32, PyTorch FP32'ye göre saf çıkarımda 2,60 kat; video hattında 1,87 kat hızlanmıştır. En yüksek hız TensorRT FP16 ile elde edilmiştir. FP16, PyTorch FP32'ye göre saf çıkarımda 3,26 kat, video hattında 2,01 kat hızlanırken mAP50–95 farkı 0,0003 olmuştur.

INT8 dosyası FP16 dosyasından %16,2 daha küçük olmasına rağmen saf çıkarım ve video hızında FP16'nın gerisinde kalmıştır. Ayrıca mAP50–95 değeri FP16'ya kıyasla yaklaşık %11,0 göreli azalmıştır. Bu bulgu, daha düşük bit genişliğinin hedef donanımda otomatik olarak daha yüksek uygulama performansı anlamına gelmediğini göstermektedir.

## 4. Tartışma

FP32 ve FP16 TensorRT sonuçlarının mAP değerleri birbirine çok yakındır. Bu durum FP16'nın incelenen ağ ve veri kümesinde sayısal hassasiyeti büyük ölçüde koruduğunu gösterir. FP16'nın Tensor Çekirdeklerinden yararlanması ve daha düşük veri hareketi gereksinimi, ölçülen hız artışının muhtemel nedenleridir. Buna karşın INT8 sonucu kalibrasyonun doğruluk üzerindeki etkisini açıkça göstermektedir. Nicemleme literatürü, INT8 başarısının uygun ölçekleme ve temsil edici kalibrasyon verisine bağlı olduğunu vurgulamaktadır [3,4].

Video FPS değerlerinin saf çıkarım FPS değerlerinden düşük olması beklenen bir sonuçtur; çünkü video hattı kod çözme ve veri hazırlama maliyetlerini de içerir. Gerçek uygulama kapasitesi değerlendirilirken yalnızca model çekirdeğinin FPS değeri yerine uçtan uca hattın göz önünde bulundurulması gerekir.

Fiziksel güvenlik bağlamında düşük gecikme, olayların daha erken işaretlenmesine yardımcı olabilir. Bununla birlikte prototip yalnızca nesne tespiti gerçekleştirir; tehdit veya anomali kararı vermez. Yanlış pozitif/negatif sonuçların operasyonel etkisi, kullanım senaryosuna özgü veri ve insan denetimiyle ayrıca değerlendirilmelidir.

## 5. Sınırlılıklar ve Geçerlilik Tehditleri

Çalışma tek GPU, tek model boyutu ve tek genel amaçlı veri kümesiyle sınırlıdır. Güç tüketimi, termal kararlılık ve uzun süreli yük ölçülmemiştir. Video deneyi tek bir kısa örnekle yürütülmüştür. INT8 kalibrasyon veri seçimi ve boyutu sonuçları etkileyebilir. Dizüstü bilgisayarın güç profili de tekrarlanabilirlik bakımından nihai sürümde raporlanmalıdır. Bu nedenle sonuçlar farklı GPU'lara veya güvenlik kameralarının alan verisine doğrudan genellenmemelidir.

## 6. Sonuç

YOLOv8n'nin NVIDIA TensorRT ile optimize edilmesi, temel PyTorch FP32 uygulamasına kıyasla belirgin hız kazancı sağlamıştır. Test edilen sistemde FP16; en yüksek saf çıkarım ve video hızını, referansa çok yakın COCO doğruluğuyla birlikte sunmuştur. INT8 daha küçük dosya üretmesine karşın hem FP16'dan yavaş kalmış hem de belirgin doğruluk kaybı oluşturmuştur. Bu nedenle mevcut donanım ve yazılım bileşiminde gerçek zamanlı akıllı gözetim prototipi için önerilen çalışma biçimi TensorRT FP16'dır.

Gelecek çalışmalarda alan-özgü güvenlik kamerası verileri, farklı YOLO model boyutları, temsil edici INT8 kalibrasyon kümeleri, enerji tüketimi ve uzun süreli termal performans incelenmelidir.

## Beyanlar

**Etik kurul:** Çalışmada insan katılımcı deneyi veya kişisel veri analizi yapılmamıştır. Kullanılan test videosunun kaynak ve kullanım izni, nihai sürümden önce yazar tarafından doğrulanmalıdır.

**Çıkar çatışması:** Yazar çıkar çatışması bulunmadığını beyan eder.

**Finansman:** Çalışma için dış finansman alınmamıştır. *(Yazar tarafından doğrulanmalıdır.)*

**Veri ve kod erişilebilirliği:** Kaynak kod ve deney çıktıları, kabul edilen sürümden önce herkese açık GitHub deposu ve Zenodo DOI'si üzerinden paylaşılacaktır.

**Üretken yapay zekâ kullanımı:** OpenAI Codex; kod gözden geçirme, deney betiklerinin düzenlenmesi, dil iyileştirme ve makale taslağının yapılandırılmasında yardımcı araç olarak kullanılmıştır. Bulguların doğrulanması, kaynakların kontrolü, bilimsel yorum ve nihai metin sorumluluğu yazara aittir.

## Kaynakça

[1] Redmon, J., Divvala, S., Girshick, R. ve Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 779–788. https://doi.org/10.1109/CVPR.2016.91

[2] Lin, T.-Y. vd. (2014). Microsoft COCO: Common Objects in Context. *Computer Vision – ECCV 2014*, 740–755. https://doi.org/10.1007/978-3-319-10602-1_48

[3] Jacob, B. vd. (2018). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2704–2713. https://doi.org/10.1109/CVPR.2018.00286

[4] NVIDIA. (2026). Accuracy Considerations. *NVIDIA TensorRT Documentation*. Erişim tarihi: 24 Eylül 2026, https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html

> Bu dosya anonim hakem kopyasının çalışma taslağıdır. Resmî CDEJ Word şablonuna aktarılmadan ve danışman kontrolünden geçmeden gönderilmemelidir.
