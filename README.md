# YOLO + NVIDIA Performans Analizi Staj Projesi

Derin Öğrenme Tabanlı Nesne Tespiti Modellerinin (YOLO) NVIDIA Yazılım Bileşenleri
ile Performans Analizi ve Gerçek Zamanlı Masaüstü Uygulaması Geliştirilmesi.

Detaylı proje bağlamı ve 5 haftalık plan için `CLAUDE.md` dosyasına bakın —
Claude Code bu dosyayı otomatik okuyarak projeyi anlar.

## Hızlı Başlangıç
```bash
python -m venv venv
venv\Scripts\activate      # Windows
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/prepare_models.py --base
streamlit run src/app.py
```

Bu yol PyTorch FP32 tanıtım modunu hazırlar. İlk çalıştırmada `yolov8n.pt`
Ultralytics kaynağından otomatik indirilebilir.

### Tam NVIDIA GPU modu

CUDA 12.8 uyumlu NVIDIA sürücüsü kurulduktan sonra:

```bash
python src/check_env.py
python src/prepare_models.py --engines
python src/prepare_models.py --download-coco --int8
python src/prepare_models.py --status
streamlit run src/app.py
```

`--download-coco` yaklaşık 1 GB COCO val2017 verisi indirir. TensorRT engine
dosyaları hedef GPU'da üretilir ve Git deposuna eklenmez. Başka GPU'dan alınmış
engine dosyalarını kullanmayın.

## Klasörler
- `src/` — kod
- `models/` — model ağırlıkları (pretrained, ONNX, TensorRT engine)
- `data/` — veri seti / test videoları
- `results/` — kıyaslama sonuçları, grafikler
- `weekly_videos/` — haftalık video teslimlerine dair notlar
- `notebooks/` — deneysel çalışmalar
- `docs/` — son hafta planı ve canlıya alma dokümantasyonu
- `article/` — CDEJ makale hazırlığı ve resmî dergi şablonu
- `publication/` — GitHub, Zenodo ve CDEJ gönderim kontrol listesi

## Arayüzü Çalıştırma

```bash
venv\Scripts\streamlit.exe run src\app.py
```

Uygulama NVIDIA GPU ve TensorRT bulunan yerel ortamda FP32, FP16 ve INT8
engine'lerini kullanır. Uyumlu NVIDIA GPU/TensorRT bulunmayan bir canlı ortamda
yalnızca PyTorch FP32 modeliyle tanıtım modunda çalışır.

Canlıya alma seçenekleri ve hedef sunucuda engine oluşturma gereksinimi için
`docs/CANLIYA_ALMA.md` dosyasına bakın.

## Önemli Dağıtım Notu

TensorRT `.engine` dosyaları varsayılan olarak oluşturuldukları işletim sistemi,
TensorRT sürümü ve GPU mimarisine bağlıdır. Bu depodaki engine dosyaları farklı
bir sunucuya kopyalanarak kullanılmamalı; hedef NVIDIA GPU sunucusunda yeniden
oluşturulmalıdır.

## Yayın Sonuçları

RTX 5060 Laptop GPU üzerindeki nihai ölçümlerde TensorRT FP16, 342,52 saf çıkarım
FPS ve 147,79 uçtan uca video FPS değerine ulaşmıştır. PyTorch FP32'ye göre
hızlanma sırasıyla 3,26 ve 2,01 kattır. COCO val2017 mAP50-95 değeri PyTorch FP32
için 0,3681, TensorRT FP16 için 0,3678 olarak ölçülmüştür. Ayrıntılı ham sonuçlar
`results/publication_*.json` dosyalarındadır.

- Akademik tanıtım sayfası: `docs/index.html`
- Anonim makale taslağı: `article/manuscript_anonymous_draft.md`
- Yayın adımları: `publication/YAYIN_KONTROL_LISTESI.md`

## Lisans ve Atıf

Bu proje GNU Affero General Public License v3.0 (`AGPL-3.0-only`) altında
yayımlanmaktadır. Ağ üzerinden erişime sunulan değiştirilmiş sürümlerin kaynak
kodu da kullanıcılara sunulmalıdır. Ayrıntılar için `LICENSE` dosyasına bakın.

Akademik kullanımda atıf bilgisi `CITATION.cff` dosyasındadır. Yazar:
Hüdayi Hamza Adatepe, Düzce Üniversitesi, ORCID 0009-0005-2389-3061.
