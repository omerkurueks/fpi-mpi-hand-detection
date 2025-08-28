#!/usr/bin/env python3
"""
Import Debug Script - SAM2MediaPipeDetector import sorunlarını test eder
"""

import sys
import os
from pathlib import Path

def test_imports():
    print("🔍 Import Debug - SAM2MediaPipeDetector")
    print("=" * 50)
    
    # Mevcut dizin
    current_dir = Path.cwd()
    print(f"📁 Mevcut dizin: {current_dir}")
    
    # Python path kontrol
    print(f"🐍 Python path:")
    for i, path in enumerate(sys.path):
        print(f"   {i}: {path}")
    
    print("\n📂 Dosya yapısı kontrol:")
    
    # src klasörü var mı?
    src_dir = current_dir / "src"
    if src_dir.exists():
        print(f"✅ src klasörü bulundu: {src_dir}")
    else:
        print(f"❌ src klasörü bulunamadı: {src_dir}")
        return False
    
    # detect klasörü var mı?
    detect_dir = src_dir / "detect"
    if detect_dir.exists():
        print(f"✅ detect klasörü bulundu: {detect_dir}")
    else:
        print(f"❌ detect klasörü bulunamadı: {detect_dir}")
        return False
    
    # sam2_mediapipe_detector.py var mı?
    detector_file = detect_dir / "sam2_mediapipe_detector.py"
    if detector_file.exists():
        print(f"✅ sam2_mediapipe_detector.py bulundu: {detector_file}")
    else:
        print(f"❌ sam2_mediapipe_detector.py bulunamadı: {detector_file}")
        return False
    
    # __init__.py dosyaları var mı?
    src_init = src_dir / "__init__.py"
    detect_init = detect_dir / "__init__.py"
    
    if src_init.exists():
        print(f"✅ src/__init__.py bulundu")
    else:
        print(f"❌ src/__init__.py bulunamadı")
    
    if detect_init.exists():
        print(f"✅ detect/__init__.py bulundu")
    else:
        print(f"❌ detect/__init__.py bulunamadı")
    
    print("\n🧪 Import testleri:")
    
    # Python path'e src ekle
    src_path = str(src_dir)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"➕ src klasörü Python path'e eklendi: {src_path}")
    
    # Method 1: src.detect.sam2_mediapipe_detector
    try:
        from src.detect.sam2_mediapipe_detector import SAM2MediaPipeDetector
        print("✅ Method 1 (src.detect.sam2_mediapipe_detector) - BAŞARILI")
        return True
    except ImportError as e:
        print(f"❌ Method 1 başarısız: {e}")
    
    # Method 2: detect.sam2_mediapipe_detector (src path'e eklendikten sonra)
    try:
        from detect.sam2_mediapipe_detector import SAM2MediaPipeDetector
        print("✅ Method 2 (detect.sam2_mediapipe_detector) - BAŞARILI")
        return True
    except ImportError as e:
        print(f"❌ Method 2 başarısız: {e}")
    
    # Method 3: Direct file import
    try:
        spec = __import__("importlib.util").util.spec_from_file_location(
            "sam2_mediapipe_detector", detector_file
        )
        module = __import__("importlib.util").util.module_from_spec(spec)
        spec.loader.exec_module(module)
        SAM2MediaPipeDetector = module.SAM2MediaPipeDetector
        print("✅ Method 3 (Direct file import) - BAŞARILI")
        return True
    except Exception as e:
        print(f"❌ Method 3 başarısız: {e}")
    
    print("\n❌ Tüm import methodları başarısız!")
    
    print("\n🔧 Önerilen çözümler:")
    print("1. Virtual environment aktif mi kontrol edin")
    print("2. SAM 2 kurulumu: cd sam2_official && pip install -e .")
    print("3. Dependencies: pip install -r requirements.txt")
    print("4. Python version: Python 3.8+ gerekli")
    
    return False

def test_dependencies():
    print("\n📦 Dependency kontrolleri:")
    
    dependencies = [
        "torch",
        "torchvision", 
        "cv2",
        "mediapipe",
        "numpy"
    ]
    
    for dep in dependencies:
        try:
            if dep == "cv2":
                import cv2
                print(f"✅ OpenCV: {cv2.__version__}")
            elif dep == "torch":
                import torch
                print(f"✅ PyTorch: {torch.__version__}")
                print(f"   CUDA available: {torch.cuda.is_available()}")
            elif dep == "mediapipe":
                import mediapipe as mp
                print(f"✅ MediaPipe: {mp.__version__}")
            elif dep == "numpy":
                import numpy as np
                print(f"✅ NumPy: {np.__version__}")
            elif dep == "torchvision":
                import torchvision
                print(f"✅ TorchVision: {torchvision.__version__}")
        except ImportError as e:
            print(f"❌ {dep}: {e}")

def test_sam2():
    print("\n🤖 SAM 2 kontrolleri:")
    
    # sam2_official klasörü var mı?
    sam2_dir = Path.cwd() / "sam2_official"
    if sam2_dir.exists():
        print(f"✅ sam2_official klasörü bulundu: {sam2_dir}")
    else:
        print(f"❌ sam2_official klasörü bulunamadı: {sam2_dir}")
        return False
    
    # SAM 2 import testi
    try:
        sys.path.insert(0, str(sam2_dir))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("✅ SAM 2 imports başarılı")
        return True
    except ImportError as e:
        print(f"❌ SAM 2 import başarısız: {e}")
        print("   Çözüm: cd sam2_official && pip install -e .")
        return False

def main():
    print("🚀 FPI-MPI Hand Detection - Import Debug Tool")
    print("=" * 60)
    
    # Testleri çalıştır
    import_ok = test_imports()
    test_dependencies()
    sam2_ok = test_sam2()
    
    print("\n" + "=" * 60)
    if import_ok and sam2_ok:
        print("🎉 Tüm import testleri BAŞARILI! main.py çalıştırabilirsiniz.")
    else:
        print("❌ Import sorunları tespit edildi. Yukarıdaki çözümleri uygulayın.")

if __name__ == "__main__":
    main()
