# Hafta 4 Video Kaydı — Konuşma Metni

Toplam süre tahmini: ~2-2.5 dakika. Ekran kaydı tarayıcıda (`localhost:8501`) çekilecek,
uygulama önceden çalışır durumda olsun. Parantez içindeki `[EKRANDA: ...]` notları o an
ekranda ne göstereceğini söylüyor.

---

## 1) Giriş (15 sn)

> "Merhaba hocam, ben [adın]. Hafta 4'te, üç haftalık optimizasyon çalışmamı kullanıcının
> canlı test edebileceği bir web arayüzüne dönüştürdüm — model seçimi, video/webcam girişi,
> anlık FPS ve tespit gösterimiyle."

`[EKRANDA: localhost:8501 - uygulamanın ana sayfası]`

---

## 2) Arayüz ve canlı video işleme (55 sn)

> "Arayüz sade: solda kontrol paneli, dört optimizasyon seviyesinden birini seçebiliyorum —
> varsayılan FP16, çünkü Hafta 3'te en iyi sonucu o verdi. Güven eşiğini ayarlayabiliyorum,
> kaynak olarak örnek video, webcam ya da kendi dosyanı seçebiliyorsun."

`[EKRANDA: model dropdown'u göster, sonra Videoyu İşle butonuna bas]`

> "İşle'ye basınca video arka planda işleniyor, ilerleme çubuğu ve tahmini kalan süre
> gösteriliyor, istersen iptal edebiliyorsun. Bitince video akıcı şekilde oynatılıyor,
> altında FPS, tespit sayısı ve indirme seçeneği var."

`[EKRANDA: işlenmiş videoyu oynat, altındaki istatistikleri işaret et]`

---

## 3) Kısa bir bulgu (20 sn)

> "Test sırasında tepeden açılarda modelin insanları hayvanla karıştırdığını fark ettim —
> COCO çoğunlukla göz hizasından çekilmiş fotoğraflarla eğitilmiş. Bu proje trafik/yaya odaklı
> olduğu için sadece insan/araç sınıflarını gösteren bir filtre ekleyerek çözdüm."

`[EKRANDA: "Sadece insan/araç sınıflarını göster" checkbox'ını kapat/aç]`

---

## 4) FP16 vs INT8 canlı karşılaştırma (35 sn)

> "Geçen haftanın bulgusunu burada canlı da gösterebiliyorum: 'İki modeli yan yana
> karşılaştır' seçeneğiyle aynı videoyu FP16 ve INT8 ile aynı anda işletiyorum."

`[EKRANDA: karşılaştırma modunu aç, FP16 ve INT8 seç, işle]`

> "Sonuç Hafta 3'le tutarlı: INT8 daha az tespit üretiyor, vaat ettiği hız kazancı da
> doğruluktaki bu kaybı karşılamıyor. Bu yüzden varsayılan seçim FP16 kaldı."

---

## 5) Kapanış (15 sn)

> "Webcam modu da var, kamera tarayıcı üzerinden doğrudan açılıyor — onu kendimi göstererek
> gelecek hafta final videosunda demo edeceğim. Bu hafta tam işlevli bir web arayüzü
> tamamladım; gelecek hafta final rapor ve sunum hazırlığına başlıyorum. Teşekkürler."

---

## Kayıt öncesi kontrol listesi
- [ ] `venv\Scripts\streamlit.exe run src\app.py` ile uygulama çalışır durumda
- [ ] Tarayıcıda `localhost:8501` açık, pencere yeterince büyük/okunaklı
- [ ] Örnek video ile en az bir kere önceden test edilmiş (ilk çalıştırmada model/engine
      yükleme birkaç saniye sürebiliyor, kayıtta bu bekleme görünmesin diye)
- [ ] Mikrofon test edildi
