"""
SAM 2 + MediaPipe RTSP Stream Test
Gerçek kamera ile el tespiti ve SAM 2 segmentasyon
"""

import cv2
import numpy as np
import sys
import os
import logging
import time

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("🚀 SAM 2 + MediaPipe RTSP Test Başlatılıyor...")

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from detect.sam2_mediapipe_detector import SAM2MediaPipeDetector
    print("✅ SAM2MediaPipeDetector import edildi")
    
    # RTSP Stream URL
    rtsp_url = "rtsp://admin:HeysemAI246@192.168.150.59"
    
    # Detector'ı başlat
    print("🚀 SAM 2 + MediaPipe Detector başlatılıyor...")
    detector = SAM2MediaPipeDetector(device="cuda")
    
    if detector.sam2_predictor is None:
        print("❌ SAM 2 başlatılamadı, çıkılıyor...")
        exit(1)
    
    print("✅ SAM 2 + MediaPipe hazır!")
    print(f"🎥 RTSP stream bağlanılıyor: {rtsp_url}")
    
    # RTSP stream başlat
    cap = cv2.VideoCapture(rtsp_url)
    
    # Buffer size ayarla (OpenCV version uyumluluğu için try-except)
    try:
        cap.set(cv2.CAP_PROP_BUFFER_SIZE, 1)
    except AttributeError:
        # Eski OpenCV versiyonları için alternatif
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except AttributeError:
            # Hiçbiri yoksa varsayılan ayarları kullan
            pass
    
    # RTSP optimizasyonları
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ RTSP stream açılamadı!")
        exit(1)
    
    print("✅ RTSP stream bağlandı")
    print("📹 Video işleme başlıyor... (ESC ile çıkış)")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame okunamadı")
            break
        
        frame_count += 1
        
        # Her 5 frame'de bir process et (performance için)
        if frame_count % 5 == 0:
            # SAM 2 + MediaPipe detection
            results = detector.detect_hands_and_segments(frame)
            
            # Results'ı kullan
            annotated_frame = results['annotated_image']
            hands = results['hands']
            segments = results['segments']
            interactions = results['hand_object_interactions']
            
            # FPS hesapla
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Info text ekle
            info_text = f"FPS: {fps:.1f} | Hands: {len(hands)} | Segments: {len(segments)} | Interactions: {len(interactions)}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # SAM 2 durumu
            sam2_status = "SAM 2: ON" if detector.sam2_predictor else "SAM 2: OFF"
            cv2.putText(annotated_frame, sam2_status, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Her 30 frame'de console log
            if frame_count % 30 == 0:
                print(f"📊 Frame {frame_count}: {len(hands)} el, {len(segments)} segment, {len(interactions)} etkileşim")
            
        else:
            annotated_frame = frame
        
        # Görüntüyü göster
        cv2.imshow('SAM 2 + MediaPipe - Hand Detection & Segmentation', annotated_frame)
        
        # ESC ile çıkış
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
    
    print("🏁 Video stream kapatılıyor...")
    cap.release()
    cv2.destroyAllWindows()
    
    # Final istatistikler
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"📊 Final İstatistikler:")
    print(f"   Total Frame: {frame_count}")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Average FPS: {avg_fps:.2f}")
    
except KeyboardInterrupt:
    print("⚠️ Kullanıcı tarafından durduruldu")
    
except Exception as e:
    print(f"❌ Hata: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    cv2.destroyAllWindows()
    print("🏁 Test tamamlandı")
