# FPI-MPI Hand Detection

SAM 2 + MediaPipe entegrasyonu ile gerçek zamanlı el tespiti ve nesne segmentasyonu.

## ⚡ Hızlı Başlangıç

### Otomatik Kurulum (Önerilen)

**Windows:**
```bash
git clone https://github.com/omerkurueks/fpi-mpi-hand-detection.git
cd fpi-mpi-hand-detection
quick_setup.bat
```

**Linux/Mac:**
```bash
git clone https://github.com/omerkurueks/fpi-mpi-hand-detection.git
cd fpi-mpi-hand-detection
chmod +x quick_setup.sh
./quick_setup.sh
```

**Python Script (Çapraz Platform):**
```bash
git clone https://github.com/omerkurueks/fpi-mpi-hand-detection.git
cd fpi-mpi-hand-detection
python setup.py
```

### İlk Test
```bash
# Virtual environment aktifleştir
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# SAM 2 test
python tests/test_sam2_final.py

# Hızlı MediaPipe test
python tests/test_mediapipe_fast.py
```

## 🎯 Özellikler

- **SAM 2 (Segment Anything Model 2)** ile zero-shot object segmentation
- **MediaPipe** ile real-time hand detection ve tracking
- **RTSP stream** desteği ile IP kamera entegrasyonu
- **CUDA acceleration** desteği
- **Real-time processing** optimizasyonları

## 🚀 Kurulum (Yeni Bilgisayarda)

### Adım 1: Sistem Gereksinimleri
```bash
# Python 3.8+ gerekli
python --version

# CUDA kontrolü (GPU kullanımı için)
nvidia-smi
```

### Adım 2: Repository Clone
```bash
git clone https://github.com/omerkurueks/fpi-mpi-hand-detection.git
cd fpi-mpi-hand-detection
```

### Adım 3: Python Virtual Environment
```bash
# Virtual environment oluştur
python -m venv .venv

# Aktifleştir
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### Adım 4: Temel Dependencies
```bash
# PyTorch CUDA version (önemli!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Diğer dependencies
pip install -r requirements.txt
```

### Adım 5: SAM 2 Kurulumu
```bash
# SAM 2 zaten projede var, sadece install et
cd sam2_official
pip install -e .
cd ..
```

### Adım 6: Model Dosyalarını İndir
```bash
# Model klasörü oluştur
mkdir models
mkdir models\sam2

# SAM 2.1 Hiera Large model indir (898MB)
# Windows PowerShell:
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" -OutFile "models\sam2\sam2.1_hiera_large.pt"

# Linux/Mac:
# wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O models/sam2/sam2.1_hiera_large.pt
```

### Adım 7: Kurulum Testi
```bash
# CUDA ve dependencies kontrolü
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# SAM 2 testi
python tests\test_sam2_final.py

# MediaPipe testi
python tests\test_mediapipe_fast.py
```
# Download from: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### 5. Config Setup
```bash
mkdir -p configs/sam2.1
# Download config: https://github.com/facebookresearch/sam2/blob/main/sam2_configs/sam2.1_hiera_l.yaml
```

## 🎮 Kullanım

### 🎯 Ana Uygulama (main.py)

```bash
# Webcam ile çalıştır (varsayılan)
python main.py

# Hızlı mod (sadece MediaPipe - 30+ FPS)
python main.py --mode fast

# Tam mod (SAM 2 + MediaPipe - 5-10 FPS)
python main.py --mode full

# RTSP kamera ile
python main.py --rtsp "rtsp://user:pass@ip:port/stream"

# Video dosyası ile
python main.py --video "video.mp4"
```

### ⌨️ Kontroller
- **ESC**: Çıkış
- **SPACE**: Pause/Resume
- **S**: Screenshot kaydet

### 🧪 Test Dosyaları
```bash
# SAM 2 kurulum testi
python tests/test_sam2_final.py

# Hızlı MediaPipe testi
python tests/test_mediapipe_fast.py

# RTSP stream testi
python tests/test_sam2_rtsp.py
```

## 📋 Sistem Gereksinimleri

### Minimum
- Python 3.8+
- CUDA 11.8+ (GPU acceleration için)
- 8GB RAM
- 4GB GPU memory

### Önerilen
- Python 3.9+
- CUDA 12.1+
- 16GB RAM
- 8GB+ GPU memory (RTX 3060 veya üzeri)

## 🔧 Konfigürasyon

### GPU Ayarları
```python
# CUDA kullanımı
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths
SAM2_CHECKPOINT = "models/sam2/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
```

### RTSP Stream
```python
# RTSP URL örneği
RTSP_URL = "rtsp://username:password@ip_address:port/stream"
```

## 📊 Performans

### Test Sonuçları
- **MediaPipe Only**: 30+ FPS (640x480)
- **SAM 2 + MediaPipe**: 5-10 FPS (416x240, optimized)
- **Full Resolution SAM 2**: 1-3 FPS (640x480)

### Optimizasyon Stratejileri
- Frame skipping (her N frame'de SAM 2)
- Düşük çözünürlük processing
- ROI-based segmentation
- Model caching

## 🛠️ Troubleshooting

### Import Errors (SAM2MediaPipeDetector bulunamadı)
```bash
# Debug script çalıştır
python debug_imports.py

# Yaygın çözümler:
# 1. Virtual environment aktif mi?
.venv\Scripts\activate  # Windows

# 2. SAM 2 kurulumu
cd sam2_official
pip install -e .
cd ..

# 3. Dependencies tekrar yükle
pip install -r requirements.txt --force-reinstall
```

### CUDA Errors
```bash
# PyTorch CUDA version check
python -c "import torch; print(torch.cuda.is_available())"

# CUDA-enabled PyTorch install
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### SAM 2 Import Errors
```bash
# SAM 2 reinstall
cd sam2_official
pip install -e . --force-reinstall
```

### OpenCV RTSP Issues
```bash
# OpenCV with FFMPEG support
pip uninstall opencv-python
pip install opencv-python-headless
```

**Kullanım Alanları:**
- Endüstriyel kalite kontrol süreçlerinde işçi davranış analizi
- Laboratuvar ortamlarında araştırmacı aktivite takibi
- Eğitim videolarında öğrenci etkileşim analizi
- Retail ortamlarında müşteri ürün inceleme davranışları

## 🏗️ Sistem Mimarisi

### Temel Bileşenler

```
┌─────────────────┐    ┌───────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│   Hand Detection  │───▶│ Object Detection│
│  (USB/RTSP/MP4) │    │   (MediaPipe)     │    │     (YOLO)      │
└─────────────────┘    └───────────────────┘    └─────────────────┘
                                │                          │
                                ▼                          ▼
┌─────────────────┐    ┌───────────────────┐    ┌─────────────────┐
│ Event Logging   │◀───│  State Machine    │◀───│ Motion Analysis │
│  (JSONL/CSV)    │    │   (FSM Logic)     │    │ (Optical Flow)  │
└─────────────────┘    └───────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌───────────────────┐
                       │   API Endpoints   │
                       │    (FastAPI)      │
                       └───────────────────┘
```

### Veri Akışı

1. **Görüntü Giriş** → Video akışından frame-by-frame okuma
2. **El Tespiti** → MediaPipe ile gerçek zamanlı el landmark'ları
3. **Nesne Tespiti** → YOLO ile çevredeki nesnelerin tespiti
4. **Eşleştirme** → IoU/mesafe tabanlı el-nesne eşleştirmesi
5. **Hareket Analizi** → Optik akış ile hareket metriklerinin hesaplanması
6. **Durum Yönetimi** → FSM ile inceleme davranışının tespiti
7. **Kayıt & Raporlama** → Event'lerin kaydedilmesi ve API üzerinden sunumu

## 🚀 Hızlı Başlangıç

### Önkoşullar

```bash
# Python 3.10+ gerekli
python --version

# Git ile proje klonlama
git clone <repository-url>
cd fpi-mpi-hand-detection
```

### Kurulum

#### 1. Virtual Environment Oluşturma

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS  
python -m venv venv
source venv/bin/activate
```

#### 2. Bağımlılıkları Yükleme

```bash
# Temel bağımlılıklar
pip install -r requirements.txt

# CUDA destekli PyTorch (GPU kullanımı için)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Konfigürasyon

```bash
# Varsayılan konfigürasyonları kopyalama
cp configs/logic.yaml.example configs/logic.yaml
cp configs/model.yaml.example configs/model.yaml
cp configs/data.yaml.example configs/data.yaml
```

### 🐳 Docker ile Kurulum

```bash
# Tek komutla başlatma
docker-compose up api

# Arkaplanda çalıştırma
docker-compose up -d api

# Diğer servisler
docker-compose --profile train up training    # Model eğitimi
docker-compose --profile eval up evaluation   # Değerlendirme
docker-compose --profile cache up redis       # Cache sistemi
```

## 📋 Kullanım Kılavuzu

### 1. Gerçek Zamanlı İnference

```bash
# Webcam ile canlı tespit
python scripts/infer.py --source 0 --config configs/logic.yaml

# Video dosyası ile tespit
python scripts/infer.py --source video.mp4 --config configs/logic.yaml --output results/

# RTSP stream ile tespit
python scripts/infer.py --source rtsp://admin:HeysemAI246@192.168.150.59 --config configs/logic.yaml
```

### 2. API Server

```bash
# FastAPI server başlatma
uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# API dökümantasyonu
# http://localhost:8000/docs
```

#### API Endpoints

- **POST** `/infer` - Frame-by-frame inference
- **GET** `/events` - Event geçmişini alma
- **GET** `/events/current` - Aktif event'ler
- **GET** `/health` - Sistem sağlık durumu
- **GET** `/stats` - İstatistikler

### 3. Model Eğitimi

```bash
# YOLO modeli eğitimi
python scripts/train.py --config configs/model.yaml --data data/yolo_dataset/

# Eğitim parametreleri
python scripts/train.py --epochs 100 --batch-size 16 --imgsz 640 --device 0
```

### 4. Değerlendirme

```bash
# Model performans değerlendirmesi
python scripts/eval.py --predictions data/predictions/ --ground-truth data/ground_truth/

# Temporal IoU analizi
python scripts/eval.py --temporal-analysis --window-size 30
```

## ⚙️ Konfigürasyon

### Ana Konfigürasyon (configs/logic.yaml)

```yaml
# Video giriş ayarları
video:
  source: 0                    # 0=webcam, 'video.mp4', 'rtsp://...'
  fps: 30
  resolution: [640, 480]
  buffer_size: 1

# El tespit ayarları
hands:
  max_num_hands: 2
  min_detection_confidence: 0.7
  min_tracking_confidence: 0.5
  model_complexity: 1

# Nesne tespit ayarları  
objects:
  model_path: "models/yolo_objects.pt"
  confidence_threshold: 0.5
  iou_threshold: 0.45
  device: "auto"               # "cpu", "cuda", "auto"

# Hareket analizi
motion:
  method: "farneback"          # "farneback", "dis", "lucas_kanade"
  scale: 0.5
  flow_threshold: 2.0

# İnceleme davranış tespiti
inhand:
  matching_strategy: "center_iou"
  iou_threshold: 0.3
  match_threshold: 50.0
  min_confidence: 0.5

# Durum makinesi
fsm:
  start_threshold: 2.0         # Hareket eşiği (inceleme başlangıcı)
  stop_threshold: 1.0          # Hareket eşiği (inceleme bitişi) 
  min_frames_start: 3          # Minimum frame sayısı (başlangıç)
  min_frames_stop: 3           # Minimum frame sayısı (bitiş)
  grace_frames: 5              # Tespit kaybı toleransı
  confidence_threshold: 0.7    # Minimum güven skoru

# Görselleştirme
visualization:
  show_hands: true
  show_objects: true
  show_flow: false
  show_tracks: true
  font_size: 0.6
  line_thickness: 2
```

### Model Konfigürasyonu (configs/model.yaml)

```yaml
# YOLO eğitim ayarları
model:
  architecture: "yolov8n"      # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  pretrained: true
  classes: 80                  # COCO sınıf sayısı

training:
  epochs: 100
  batch_size: 16
  imgsz: 640
  optimizer: "SGD"
  lr0: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  
augmentation:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 0.0
  translate: 0.1
  scale: 0.5
  shear: 0.0
  perspective: 0.0
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0
  mixup: 0.0
```

## 📊 Veri Formatları

### Event Logging (JSONL)

```json
{
  "event_id": "evt_20240101_123456_001",
  "track_id": 1,
  "start_time": 1704110096.123,
  "end_time": 1704110098.456,
  "duration": 2.333,
  "hand_bbox": [120, 150, 200, 250],
  "object_bbox": [130, 160, 190, 230],
  "confidence": 0.85,
  "motion_metrics": {
    "flow_magnitude": 2.5,
    "centroid_movement": 1.2,
    "area_ratio": 1.1,
    "bbox_change": 0.8
  },
  "metadata": {
    "hand_type": "Right",
    "grip_strength": 0.7,
    "object_class": "bottle"
  }
}
```

### YOLO Dataset Format

```
data/yolo_dataset/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   └── img_002.jpg
│   └── val/
│       ├── img_003.jpg
│       └── img_004.jpg
├── labels/
│   ├── train/
│   │   ├── img_001.txt    # class x_center y_center width height
│   │   └── img_002.txt
│   └── val/
│       ├── img_003.txt
│       └── img_004.txt
└── data.yaml
```

## 🧪 Test Etme

```bash
# Tüm testleri çalıştırma
python -m pytest tests/ -v

# Belirli modül testleri
python -m pytest tests/test_fsm.py -v
python -m pytest tests/test_inhand.py -v
python -m pytest tests/test_motion.py -v

# Coverage raporu
python -m pytest tests/ --cov=src --cov-report=html
```

## 📈 Performans Metrikleri

### Sistem Performansı

- **FPS**: Saniyede işlenen frame sayısı
- **Latency**: Frame işleme süresi (ms)
- **Memory**: RAM kullanımı (MB)
- **CPU/GPU**: İşlemci kullanım yüzdesi

### Tespit Performansı

- **Hand Detection**: MediaPipe accuracy/precision
- **Object Detection**: YOLO mAP@0.5, mAP@0.5:0.95
- **Tracking**: MOT metrics (MOTA, MOTP, IDF1)
- **Temporal IoU**: Zaman bazlı overlap analizi

### Değerlendirme Örneği

```bash
# Performans raporu oluşturma
python scripts/eval.py --predictions results/ --ground-truth annotations/ --report performance_report.html

# Gerçek zamanlı metrikler
curl http://localhost:8000/stats
```

## 🔧 Gelişmiş Kullanım

### Custom Object Detection

```python
# Kendi YOLO modelinizi kullanma
from src.detect.yolo_wrapper import YOLOWrapper

yolo = YOLOWrapper("path/to/your/model.pt")
detections = yolo.detect(frame, confidence=0.5)
```

### Custom Motion Analysis

```python
# Kendi hareket analizörünüzü oluşturma
from src.motion.optical_flow import OpticalFlowAnalyzer

analyzer = OpticalFlowAnalyzer(config)
flow = analyzer.compute_frame(frame)
metrics = analyzer.analyze_region(bbox, prev_bbox)
```

### API Integration

```python
import requests

# Inference API kullanımı
with open("frame.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/infer",
        files={"frame": f},
        data={"timestamp": time.time()}
    )
    
result = response.json()
print(f"Detected {len(result['tracks'])} hand-object pairs")
```

## 🛠️ Troubleshooting

### Yaygın Problemler

#### 1. MediaPipe Import Hatası
```bash
# Çözüm
pip uninstall mediapipe
pip install mediapipe
```

#### 2. CUDA Memory Error
```bash
# GPU memory temizleme
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### 3. OpenCV Video Capture
```bash
# Codec yükleme (Linux)
sudo apt-get install ffmpeg
```

#### 4. Performance Issues
```python
# Konfigürasyonda optimizasyon
motion:
  scale: 0.25        # Daha düşük çözünürlük
  
hands:
  model_complexity: 0  # Daha hızlı model
```

### Debug Mode

```bash
# Verbose logging
python scripts/infer.py --source 0 --debug --log-level DEBUG

# Visualization ile debug
python scripts/infer.py --source 0 --show-debug --save-debug-frames
```

## 📁 Proje Yapısı

```
fpi-mpi-hand-detection/
├── 📁 src/                    # Ana kaynak kodları
│   ├── 📁 api/               # FastAPI endpoints
│   ├── 📁 detect/            # Tespit modülleri (hands, objects)
│   ├── 📁 logic/             # İş mantığı (FSM, inhand)
│   ├── 📁 motion/            # Hareket analizi
│   ├── 📁 io/                # I/O işlemleri
│   ├── 📁 viz/               # Görselleştirme
│   ├── 📄 config.py          # Konfigürasyon yönetimi
│   └── 📄 pipeline.py        # Ana pipeline
├── 📁 configs/               # Konfigürasyon dosyaları
├── 📁 scripts/               # CLI araçları
├── 📁 tools/                 # Veri dönüşüm araçları
├── 📁 tests/                 # Unit testler
├── 📁 data/                  # Veri dizini
│   ├── 📁 raw/              # Ham veri
│   ├── 📁 processed/        # İşlenmiş veri
│   ├── 📁 events/           # Event logları
│   └── 📁 models/           # Eğitilmiş modeller
├── 📁 docs/                  # Dokümantasyon
├── 📄 requirements.txt       # Python bağımlılıkları
├── 📄 Dockerfile            # Docker image tarifi
├── 📄 docker-compose.yml    # Multi-service orchestration
└── 📄 README.md             # Bu dosya
```

## 🚀 Genişleme Yol Haritası

### Kısa Vadeli (1-3 ay)
- [x] **MVP Tamamlama**: Temel tespit ve FSM sistemi
- [x] **API Endpoints**: RESTful API ile entegrasyon
- [x] **Docker Support**: Containerized deployment
- [ ] **Model Optimization**: ONNX/TensorRT conversion
- [ ] **Multi-camera Support**: Çoklu kamera desteği
- [ ] **Real-time Dashboard**: Web-based monitoring

### Orta Vadeli (3-6 ay)
- [ ] **Advanced Tracking**: DeepSORT/ByteTrack entegrasyonu
- [ ] **3D Hand Pose**: MediaPipe World landmarks
- [ ] **Action Recognition**: Temporal CNN models
- [ ] **Data Augmentation**: Synthetic data generation
- [ ] **Edge Deployment**: Jetson/RaspberryPi support
- [ ] **Cloud Integration**: AWS/Azure deployment

### Uzun Vadeli (6+ ay)
- [ ] **Transformer Models**: ViT-based detection
- [ ] **Federated Learning**: Distributed training
- [ ] **AR Visualization**: Real-time AR overlay
- [ ] **Behavior Analytics**: Advanced pattern recognition
- [ ] **Multi-modal Fusion**: Audio-visual analysis
- [ ] **Domain Adaptation**: Industry-specific models

## 🤝 Katkı Sağlama

### Development Setup

```bash
# Development dependencies
pip install -r requirements-dev.txt

# Pre-commit hooks
pre-commit install

# Code formatting
black src/ tests/
isort src/ tests/

# Linting
flake8 src/ tests/
mypy src/
```

### Contribution Guidelines

1. **Fork** projeyi
2. **Feature branch** oluşturun (`git checkout -b feature/amazing-feature`)
3. **Commit** değişikliklerinizi (`git commit -m 'Add amazing feature'`)
4. **Push** branch'e (`git push origin feature/amazing-feature`)
5. **Pull Request** açın

### Code Standards

- **Python**: PEP 8 uyumlu
- **Type Hints**: Tüm fonksiyonlarda
- **Docstrings**: Google style
- **Tests**: %90+ coverage hedefi
- **Logging**: Structured logging with loguru

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

## 🙏 Teşekkürler

- **MediaPipe**: Google's hand detection framework
- **YOLOv8**: Ultralytics object detection
- **OpenCV**: Computer vision operations
- **FastAPI**: Modern web framework
- **PyTorch**: Deep learning framework

## 📞 İletişim

- **Proje**: [GitHub Repository](https://github.com/your-username/fpi-mpi-hand-detection)
- **Issues**: [GitHub Issues](https://github.com/your-username/fpi-mpi-hand-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/fpi-mpi-hand-detection/discussions)

---

**🔍 Elde Nesne İnceleme Tespiti** - Gelişmiş bilgisayar görmesi ile el-nesne etkileşim analizi

*Bu README, projenin kapsamlı kullanım kılavuzu ve teknik dokümantasyonudur. Güncel bilgiler için repository'yi takip edin.*
