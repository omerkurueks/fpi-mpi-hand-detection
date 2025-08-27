"""
SAM 2 + MediaPipe Test - Güncellenmiş Kod
"""

import cv2
import numpy as np
import sys
import os
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("🚀 SAM 2 + MediaPipe Test Başlatılıyor...")

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
print("🟢 Path setup tamam")

try:
    print("📦 SAM2MediaPipeDetector import ediliyor...")
    from detect.sam2_mediapipe_detector import SAM2MediaPipeDetector
    print("✅ SAM2MediaPipeDetector başarıyla import edildi")
    
    # Detector'ı başlat (CUDA ile)
    print("🚀 Detector başlatılıyor (CUDA)...")
    detector = SAM2MediaPipeDetector(device="cuda")
    print("✅ Detector başarıyla başlatıldı")
    
    # SAM 2 durumunu kontrol et
    print(f"🔍 SAM 2 Predictor durumu: {detector.sam2_predictor is not None}")
    print(f"🔍 Video Predictor durumu: {detector.video_predictor is not None}")
    
    if detector.sam2_predictor is not None:
        print("🎉 SAM 2 başarıyla çalışıyor!")
        
        # Test image oluştur
        print("🖼️ Test image oluşturuluyor...")
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[100:200, 100:200] = (255, 255, 255)  # Beyaz kare
        
        # Basit detection test
        print("🔍 Detection testi başlatılıyor...")
        results = detector.detect_hands_and_segments(test_image)
        
        print(f"✅ Detection tamamlandı:")
        print(f"   📏 Image shape: {test_image.shape}")
        print(f"   👋 El sayısı: {len(results['hands'])}")
        print(f"   🎭 Segment sayısı: {len(results['segments'])}")
        print(f"   🤝 Etkileşim sayısı: {len(results['hand_object_interactions'])}")
        
        print("🎉 Test başarılı! SAM 2 + MediaPipe çalışıyor!")
    else:
        print("❌ SAM 2 predictor oluşturulamadı")
    
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    print("📦 Eksik bağımlılıklar olabilir")
    
except Exception as e:
    print(f"❌ Genel hata: {e}")
    import traceback
    traceback.print_exc()
    
print("🏁 Test tamamlandı")
