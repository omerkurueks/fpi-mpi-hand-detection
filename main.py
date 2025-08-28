#!/usr/bin/env python3
"""
FPI-MPI Hand Detection - Ana Çalıştırma Dosyası
SAM 2 + MediaPipe ile gerçek zamanlı el tespiti ve nesne segmentasyonu

Kullanım:
    python main.py                    # Webcam ile çalıştır
    python main.py --rtsp URL         # RTSP stream ile çalıştır
    python main.py --video video.mp4 # Video dosyası ile çalıştır
    python main.py --mode fast        # Sadece MediaPipe (hızlı)
    python main.py --mode full        # SAM 2 + MediaPipe (detaylı)
"""

import cv2
import argparse
import sys
import time
from pathlib import Path

# Python path'e src klasörünü ekle
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Ana detector import
try:
    from detect.sam2_mediapipe_detector import SAM2MediaPipeDetector
except ImportError as e:
    print("❌ SAM2MediaPipeDetector import edilemedi!")
    print(f"   Hata: {e}")
    print("   Çözümler:")
    print("   1. cd sam2_official && pip install -e .")
    print("   2. SAM 2 model dosyası var mı kontrol edin")
    print("   3. Virtual environment aktif mi kontrol edin")
    sys.exit(1)

# MediaPipe import
try:
    import mediapipe as mp
except ImportError:
    print("❌ MediaPipe import edilemedi!")
    print("   Kurulum: pip install mediapipe")
    sys.exit(1)

class FPIMPIApp:
    """Ana uygulama sınıfı"""
    
    def __init__(self, mode="full"):
        self.mode = mode
        self.detector = None
        self.mp_hands = None
        self.mp_drawing = None
        
        print(f"🚀 FPI-MPI Hand Detection başlatılıyor ({mode} mode)...")
        
        if mode == "full":
            print("🤖 SAM 2 + MediaPipe yükleniyor...")
            self.detector = SAM2MediaPipeDetector()
            print("✅ SAM 2 + MediaPipe hazır!")
        else:
            print("⚡ Sadece MediaPipe yükleniyor...")
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            print("✅ MediaPipe hazır!")
    
    def process_frame_fast(self, frame):
        """Hızlı MediaPipe-only processing"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)
        
        hand_count = 0
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Hand landmarks çiz
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                )
                
                # Hand bilgileri
                h, w = frame.shape[:2]
                wrist = hand_landmarks.landmark[0]
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                
                # El pozisyon bilgisi
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, f"Hand {idx+1}", 
                          (wrist_x, wrist_y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Basit gesture detection
                thumb_y = thumb_tip.y * h
                index_y = index_tip.y * h
                
                if abs(thumb_y - index_y) < 30:  # Yakın parmaklar
                    cv2.putText(frame, "Pinch", 
                              (wrist_x, wrist_y + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return frame, hand_count
    
    def process_frame_full(self, frame):
        """Tam SAM 2 + MediaPipe processing"""
        try:
            results = self.detector.detect_frame(frame)
            hand_count = 0
            
            if results and 'hands' in results:
                hand_count = len(results['hands'])
                
                for hand_info in results['hands']:
                    # Hand landmarks çiz
                    if 'landmarks' in hand_info:
                        landmarks = hand_info['landmarks']
                        for landmark in landmarks:
                            x = int(landmark['x'] * frame.shape[1])
                            y = int(landmark['y'] * frame.shape[0])
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    
                    # Hand detection bilgisi
                    cv2.putText(frame, f"Hand: {hand_info.get('label', 'Unknown')}", 
                              (10, 30 + hand_count * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # SAM 2 segmentation
                    if 'bbox' in hand_info:
                        bbox = hand_info['bbox']
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, "SAM 2 Region", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            return frame, hand_count
            
        except Exception as e:
            print(f"⚠️ Full processing error: {e}")
            # Fallback to fast mode
            return self.process_frame_fast(frame)
    
    def run_webcam(self):
        """Webcam ile çalıştır"""
        print("📷 Webcam açılıyor...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Webcam açılamadı!")
            return
        
        print("✅ Webcam hazır")
        self._run_video_loop(cap, "Webcam")
    
    def run_rtsp(self, rtsp_url):
        """RTSP stream ile çalıştır"""
        print(f"📡 RTSP stream bağlanılıyor: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url)
        
        # RTSP optimizasyonları
        try:
            cap.set(cv2.CAP_PROP_BUFFER_SIZE, 1)
        except AttributeError:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except AttributeError:
                pass
        
        if not cap.isOpened():
            print("❌ RTSP stream açılamadı!")
            return
        
        print("✅ RTSP stream bağlandı")
        self._run_video_loop(cap, "RTSP Stream")
    
    def run_video(self, video_path):
        """Video dosyası ile çalıştır"""
        print(f"🎥 Video açılıyor: {video_path}")
        
        if not Path(video_path).exists():
            print(f"❌ Video dosyası bulunamadı: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("❌ Video açılamadı!")
            return
        
        print("✅ Video hazır")
        self._run_video_loop(cap, f"Video: {Path(video_path).name}")
    
    def _run_video_loop(self, cap, source_name):
        """Ana video işleme döngüsü"""
        print(f"🎬 {source_name} processing başlıyor...")
        print(f"   Mode: {self.mode}")
        print("   ESC: Çıkış | SPACE: Pause | S: Screenshot")
        
        frame_count = 0
        fps_start = time.time()
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("📽️ Video sonu veya bağlantı kesildi")
                        break
                    
                    frame_count += 1
                    
                    # Frame processing
                    start_time = time.time()
                    if self.mode == "full":
                        processed_frame, hand_count = self.process_frame_full(frame)
                    else:
                        processed_frame, hand_count = self.process_frame_fast(frame)
                    
                    processing_time = time.time() - start_time
                    
                    # FPS hesapla
                    if frame_count % 30 == 0:
                        fps = 30 / (time.time() - fps_start)
                        fps_start = time.time()
                        print(f"📊 FPS: {fps:.1f}, Frame: {frame_count}, Hands: {hand_count}")
                    
                    # Bilgi overlay
                    cv2.putText(processed_frame, f"FPS: {1/processing_time:.1f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(processed_frame, f"Hands: {hand_count}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"Mode: {self.mode.upper()}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(processed_frame, f"Frame: {frame_count}", 
                               (10, processed_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    frame = processed_frame
                
                # Görüntüyü göster
                cv2.imshow(f'FPI-MPI Hand Detection - {source_name}', frame)
                
                # Klavye kontrolleri
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("👋 Kullanıcı tarafından durduruldu")
                    break
                elif key == ord(' '):  # SPACE
                    paused = not paused
                    status = "PAUSED" if paused else "RUNNING"
                    print(f"⏸️ {status}")
                elif key == ord('s'):  # S
                    screenshot_name = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_name, frame)
                    print(f"📸 Screenshot kaydedildi: {screenshot_name}")
                
        except KeyboardInterrupt:
            print("\n👋 Ctrl+C ile durduruldu")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("🏁 İşlem tamamlandı")

def main():
    parser = argparse.ArgumentParser(description="FPI-MPI Hand Detection - Ana Uygulama")
    parser.add_argument("--mode", choices=["fast", "full"], default="full",
                       help="İşleme modu: fast (sadece MediaPipe) veya full (SAM2+MediaPipe)")
    parser.add_argument("--rtsp", type=str, help="RTSP stream URL")
    parser.add_argument("--video", type=str, help="Video dosyası yolu")
    parser.add_argument("--webcam", action="store_true", help="Webcam kullan (varsayılan)")
    
    args = parser.parse_args()
    
    print("🔍 FPI-MPI Hand Detection")
    print("=" * 40)
    
    # Uygulama başlat
    app = FPIMPIApp(mode=args.mode)
    
    # Video kaynağına göre çalıştır
    if args.rtsp:
        app.run_rtsp(args.rtsp)
    elif args.video:
        app.run_video(args.video)
    else:
        app.run_webcam()

if __name__ == "__main__":
    main()
