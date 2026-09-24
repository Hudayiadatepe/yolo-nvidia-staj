# Yayın Öncesi Test Raporu

**Tarih:** 24 Eylül 2026<br>
**Sistem:** Windows 11, NVIDIA GeForce RTX 5060 Laptop GPU, CUDA 12.8,
TensorRT 11.2.1.2, PyTorch 2.7.1+cu128

## Geçen testler

- [x] `pip check`: bozuk veya çakışan bağımlılık bulunmadı.
- [x] `pip install --dry-run -r requirements.txt`: sabitlenen GPU bağımlılıkları çözümlendi.
- [x] Python sözdizimi/derleme kontrolü: tüm `src/` betikleri derlendi.
- [x] Streamlit başlangıç testi: uygulama istisnasız açıldı.
- [x] Sunucu sağlık kontrolü: `/_stcore/health` yanıtı `ok`.
- [x] PyTorch FP32: gerçek görüntü üzerinde çıkarım başarılı.
- [x] TensorRT FP32: gerçek görüntü üzerinde çıkarım başarılı.
- [x] TensorRT FP16: gerçek görüntü üzerinde çıkarım başarılı.
- [x] TensorRT INT8: gerçek görüntü üzerinde çıkarım başarılı.
- [x] Örnek video: FP16 ile işlendi, video oynatıcı ve indirme düğmesi oluştu.
- [x] Dosya yükleme: 1,5 MB MP4 tarayıcıdan yüklendi ve FP16 ile işlendi.
- [x] Kaynak değişimi: tamamlanmış video çıktısı webcam moduna geçerken temizlendi.
- [x] WebRTC bileşeni: `START` ve `SELECT DEVICE` kontrolleri görüntülendi.
- [x] WebRTC başlatma: FP16 engine yüklendi ve bileşen `START` durumundan `STOP`
  durumuna geçti; uygulama tarafında istisna oluşmadı.
- [x] Yayın JSON dosyaları geçerli JSON olarak ayrıştırıldı.
- [x] GitHub Pages sayfasındaki göreli dosya bağlantılarının hedefleri mevcut.
- [x] Git sahneleme denetiminde boşluk hatası bulunmadı.
- [x] Herkese açık GitHub deposu oluşturuldu ve `main` dalı başarıyla gönderildi.
- [x] GitHub Pages sitesi masaüstü ve 390×844 mobil görünümde doğrulandı;
  yatay taşma görülmedi ve iki sonuç grafiği eksiksiz yüklendi.

## Kullanıcı onayı/harici hesap gerektiren testler

- [ ] Fiziksel kamera paylaşabilen Chrome/Edge üzerinde gerçek webcam karesi ve
  tespit çıktısı doğrulanacak. Codex uygulama içi tarayıcısı medya aygıtı
  sunmadığından bu oturumda görüntü akışı alınamadı.
- [ ] GitHub release sonrasında Zenodo DOI kaydı doğrulanacak.

## Bilinen dağıtım davranışı

TensorRT engine dosyaları Git'e eklenmez ve hedef NVIDIA GPU'da yeniden
oluşturulur. Temiz kopyada PyTorch ağırlığı `python src/prepare_models.py --base`
komutuyla veya uygulamanın ilk açılışında indirilir. GitHub Pages sayfası statiktir;
GPU çıkarımı çalıştırmaz, deney sonuçlarını ve demo bağlantısını sunar.
