#!/usr/bin/env python3
"""
FPI-MPI Hand Detection - Setup Script
Yeni bilgisayarda otomatik kurulum scripti
"""

import os
import sys
import subprocess
import platform
import urllib.request
from pathlib import Path

def run_command(command, description):
    """Komut çalıştır ve sonucu göster"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} tamamlandı")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} başarısız: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Python versiyonunu kontrol et"""
    print("🐍 Python versiyonu kontrol ediliyor...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Python 3.8+ gerekli")
        return False

def check_cuda():
    """CUDA varlığını kontrol et"""
    print("🎮 CUDA kontrolü...")
    try:
        result = subprocess.run("nvidia-smi", capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU tespit edildi")
            return True
        else:
            print("⚠️ NVIDIA GPU bulunamadı - CPU modunda çalışacak")
            return False
    except FileNotFoundError:
        print("⚠️ nvidia-smi bulunamadı - CPU modunda çalışacak")
        return False

def setup_virtual_environment():
    """Virtual environment oluştur"""
    if not os.path.exists(".venv"):
        return run_command("python -m venv .venv", "Virtual environment oluşturma")
    else:
        print("✅ Virtual environment zaten mevcut")
        return True

def install_pytorch_cuda():
    """PyTorch CUDA versiyonunu yükle"""
    is_windows = platform.system() == "Windows"
    activate_cmd = ".venv\\Scripts\\activate" if is_windows else "source .venv/bin/activate"
    
    if is_windows:
        command = f"{activate_cmd} && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    else:
        command = f"{activate_cmd} && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    
    return run_command(command, "PyTorch CUDA kurulumu")

def install_requirements():
    """Requirements.txt'den paketleri yükle"""
    is_windows = platform.system() == "Windows"
    activate_cmd = ".venv\\Scripts\\activate" if is_windows else "source .venv/bin/activate"
    command = f"{activate_cmd} && pip install -r requirements.txt"
    return run_command(command, "Dependencies kurulumu")

def install_sam2():
    """SAM 2'yi yükle"""
    if os.path.exists("sam2_official"):
        is_windows = platform.system() == "Windows"
        activate_cmd = ".venv\\Scripts\\activate" if is_windows else "source .venv/bin/activate"
        command = f"{activate_cmd} && cd sam2_official && pip install -e ."
        return run_command(command, "SAM 2 kurulumu")
    else:
        print("❌ sam2_official klasörü bulunamadı")
        return False

def download_sam2_model():
    """SAM 2 model dosyasını indir"""
    model_dir = Path("models/sam2")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "sam2.1_hiera_large.pt"
    
    if model_path.exists():
        print("✅ SAM 2 model zaten mevcut")
        return True
    
    print("📥 SAM 2 model indiriliyor (898MB)...")
    url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    
    try:
        urllib.request.urlretrieve(url, model_path)
        print("✅ SAM 2 model indirildi")
        return True
    except Exception as e:
        print(f"❌ Model indirme başarısız: {e}")
        return False

def test_installation():
    """Kurulumu test et"""
    print("\n🧪 Kurulum testi başlıyor...")
    
    is_windows = platform.system() == "Windows"
    activate_cmd = ".venv\\Scripts\\activate" if is_windows else "source .venv/bin/activate"
    
    # CUDA testi
    cuda_test = f"{activate_cmd} && python -c \"import torch; print(f'CUDA Available: {{torch.cuda.is_available()}}')\""
    run_command(cuda_test, "CUDA testi")
    
    # SAM 2 testi
    sam_test = f"{activate_cmd} && python tests/test_sam2_final.py"
    if os.path.exists("tests/test_sam2_final.py"):
        run_command(sam_test, "SAM 2 testi")
    
    print("\n🎉 Kurulum tamamlandı!")
    print("\n📋 Kullanım:")
    print("   python tests/test_sam2_final.py        # SAM 2 testi")
    print("   python tests/test_mediapipe_fast.py    # MediaPipe testi")
    print("   python tests/test_sam2_rtsp.py         # RTSP stream testi")

def main():
    print("🚀 FPI-MPI Hand Detection - Setup Script")
    print("=" * 50)
    
    # Temel kontroller
    if not check_python_version():
        return
    
    cuda_available = check_cuda()
    
    # Kurulum adımları
    steps = [
        ("Virtual Environment", setup_virtual_environment),
        ("PyTorch CUDA", install_pytorch_cuda),
        ("Dependencies", install_requirements),
        ("SAM 2", install_sam2),
        ("SAM 2 Model", download_sam2_model),
    ]
    
    for step_name, step_func in steps:
        print(f"\n📦 {step_name} kurulumu...")
        if not step_func():
            print(f"❌ {step_name} kurulumu başarısız. İşlem durduruldu.")
            return
    
    # Test
    test_installation()

if __name__ == "__main__":
    main()
