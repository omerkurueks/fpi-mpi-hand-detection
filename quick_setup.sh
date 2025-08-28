#!/bin/bash

echo "========================================"
echo "FPI-MPI Hand Detection - Quick Setup"
echo "========================================"

echo ""
echo "🐍 Python versiyonu kontrol ediliyor..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python bulunamadı! Python 3.8+ yükleyin."
    exit 1
fi

echo ""
echo "🎮 CUDA kontrolü..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU tespit edildi"
else
    echo "⚠️ NVIDIA GPU bulunamadı - CPU modunda çalışacak"
fi

echo ""
echo "📦 Virtual environment oluşturuluyor..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Virtual environment oluşturuldu"
else
    echo "✅ Virtual environment zaten mevcut"
fi

echo ""
echo "🔄 Virtual environment aktifleştiriliyor..."
source .venv/bin/activate

echo ""
echo "📥 PyTorch CUDA yükleniyor..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "📥 Dependencies yükleniyor..."
pip install -r requirements.txt

echo ""
echo "🤖 SAM 2 kuruluyor..."
cd sam2_official
pip install -e .
cd ..

echo ""
echo "📁 Model klasörü oluşturuluyor..."
mkdir -p models/sam2

echo ""
echo "📥 SAM 2 model indiriliyor (898MB)..."
if [ ! -f "models/sam2/sam2.1_hiera_large.pt" ]; then
    echo "Model indiriliyor, lütfen bekleyin..."
    wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O models/sam2/sam2.1_hiera_large.pt
    echo "✅ Model indirildi"
else
    echo "✅ Model zaten mevcut"
fi

echo ""
echo "🧪 Kurulum testi..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

echo ""
echo "🎉 Kurulum tamamlandı!"
echo ""
echo "📋 Test komutları:"
echo "   python tests/test_sam2_final.py        # SAM 2 testi"
echo "   python tests/test_mediapipe_fast.py    # MediaPipe testi"
echo "   python tests/test_sam2_rtsp.py         # RTSP stream testi"
echo ""
echo "Virtual environment aktif. Çalışmaya başlayabilirsiniz!"
echo ""
