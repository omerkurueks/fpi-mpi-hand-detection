@echo off
echo ========================================
echo FPI-MPI Hand Detection - Quick Setup
echo ========================================

echo.
echo 🐍 Python versiyonu kontrol ediliyor...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python bulunamadı! Python 3.8+ yükleyin.
    pause
    exit /b 1
)

echo.
echo 🎮 CUDA kontrolü...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ NVIDIA GPU tespit edildi
) else (
    echo ⚠️ NVIDIA GPU bulunamadı - CPU modunda çalışacak
)

echo.
echo 📦 Virtual environment oluşturuluyor...
if not exist .venv (
    python -m venv .venv
    echo ✅ Virtual environment oluşturuldu
) else (
    echo ✅ Virtual environment zaten mevcut
)

echo.
echo 🔄 Virtual environment aktifleştiriliyor...
call .venv\Scripts\activate

echo.
echo 📥 PyTorch CUDA yükleniyor...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo 📥 Dependencies yükleniyor...
pip install -r requirements.txt

echo.
echo 🤖 SAM 2 kuruluyor...
cd sam2_official
pip install -e .
cd ..

echo.
echo 📁 Model klasörü oluşturuluyor...
if not exist models mkdir models
if not exist models\sam2 mkdir models\sam2

echo.
echo 📥 SAM 2 model indiriliyor (898MB)...
if not exist models\sam2\sam2.1_hiera_large.pt (
    echo Model indiriliyor, lütfen bekleyin...
    powershell -Command "Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt' -OutFile 'models\sam2\sam2.1_hiera_large.pt'"
    echo ✅ Model indirildi
) else (
    echo ✅ Model zaten mevcut
)

echo.
echo 🧪 Kurulum testi...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

echo.
echo 🔍 Import debug testi...
python debug_imports.py

echo.
echo 🎉 Kurulum tamamlandı!
echo.
echo 📋 Ana kullanım:
echo    python main.py                    # Webcam ile çalıştır
echo    python main.py --mode fast        # Hızlı mod (MediaPipe only)
echo    python main.py --rtsp URL         # RTSP stream ile
echo.
echo 🧪 Test komutları:
echo    python debug_imports.py              # Import sorunları debug
echo    python tests\test_sam2_final.py      # SAM 2 testi
echo    python tests\test_mediapipe_fast.py  # MediaPipe testi
echo.
echo Virtual environment aktif. main.py ile başlayabilirsiniz!
echo.

pause
