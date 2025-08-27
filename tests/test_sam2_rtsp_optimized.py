#!/usr/bin/env python3
"""
SAM 2 + MediaPipe RTSP Test - Optimized Version
Performans iyileştirmeleri ile hızlandırılmış versiyon
"""

import cv2
import numpy as np
import time
from detect.sam2_mediapipe_detector import SAM2MediaPipeDetector

def main():
    print("🚀 SAM 2 + MediaPipe RTSP Test Başlatılıyor (Optimized)...")
    
    try:
        # SAM 2 + MediaPipe detector başlat
        print("✅ SAM2MediaPipeDetector import edildi")
        print("🚀 SAM 2 + MediaPipe Detector başlatılıyor...")
        detector = SAM2MediaPipeDetector()
        print("✅ SAM 2 + MediaPipe hazır!")
        
        # RTSP stream ayarları
        rtsp_url = "rtsp://admin:HeysemAI246@192.168.150.59"
        print(f"🎥 RTSP stream bağlanılıyor: {rtsp_url}")
        
        # RTSP stream başlat - optimized
        cap = cv2.VideoCapture(rtsp_url)
        
        # Buffer ve optimizasyon ayarları
        try:
            cap.set(cv2.CAP_PROP_BUFFER_SIZE, 1)
        except AttributeError:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except AttributeError:
                pass
        
        # Düşük çözünürlük ve FPS - PERFORMANS İÇİN
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)  # Daha küçük
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240) # Daha küçük
        cap.set(cv2.CAP_PROP_FPS, 15)           # Daha düşük FPS
        
        if not cap.isOpened():
            print("❌ RTSP stream açılamadı!")
            return
            
        print("✅ RTSP stream bağlandı")
        print("📹 Optimized processing başlıyor...")
        print("   - Düşük çözünürlük: 416x240")
        print("   - Frame skip: Her 3 frame'de 1")
        print("   - SAM 2: Sadece hand detection varsa")
        print("   - ESC tuşu ile çıkış")
        
        frame_count = 0
        fps_start = time.time()
        process_every_n_frames = 3  # Her 3 frame'de 1 işle
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Frame alınamadı")
                break
                
            frame_count += 1
            
            # PERFORMANS: Her N frame'de bir işle
            if frame_count % process_every_n_frames != 0:
                # Sadece MediaPipe hand detection (hızlı)
                results = detector.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        detector.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, detector.mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Basit hand info
                        h, w = frame.shape[:2]
                        wrist = hand_landmarks.landmark[0]
                        cv2.putText(frame, "Hand Detected", 
                                  (int(wrist.x * w), int(wrist.y * h) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # FULL PROCESSING: SAM 2 + MediaPipe (yavaş ama detaylı)
                print(f"🔍 Full processing frame {frame_count}")
                
                # Küçük frame ile SAM 2 test
                small_frame = cv2.resize(frame, (320, 180))  # Çok küçük SAM 2 için
                
                try:
                    results = detector.detect_frame(small_frame)
                    
                    if results and 'hands' in results:
                        for hand_info in results['hands']:
                            # Hand landmarks çiz
                            if 'landmarks' in hand_info:
                                landmarks = hand_info['landmarks']
                                for landmark in landmarks:
                                    x = int(landmark['x'] * frame.shape[1] / 320)  # Scale up
                                    y = int(landmark['y'] * frame.shape[0] / 180)  # Scale up
                                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                            
                            # Hand detection bilgisi
                            cv2.putText(frame, f"Hand: {hand_info.get('label', 'Unknown')}", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # SAM 2 ile hand region segmentation
                            if 'bbox' in hand_info:
                                bbox = hand_info['bbox']
                                # Bbox'ı scale up
                                x1 = int(bbox[0] * frame.shape[1] / 320)
                                y1 = int(bbox[1] * frame.shape[0] / 180)
                                x2 = int(bbox[2] * frame.shape[1] / 320)
                                y2 = int(bbox[3] * frame.shape[0] / 180)
                                
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.putText(frame, "SAM 2 Region", (x1, y1-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    
                except Exception as e:
                    print(f"⚠️ Processing error: {e}")
                    # Fallback to simple hand detection
                    results = detector.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            detector.mp_drawing.draw_landmarks(
                                frame, hand_landmarks, detector.mp_hands.HAND_CONNECTIONS
                            )
            
            # FPS hesapla
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                fps_start = time.time()
                print(f"📊 FPS: {fps:.1f}, Frame: {frame_count}")
            
            # FPS bilgisi ekranda
            cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Mode: {'FULL' if frame_count % process_every_n_frames == 0 else 'FAST'}", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Görüntüyü göster
            cv2.imshow('SAM 2 + MediaPipe RTSP (Optimized)', frame)
            
            # ESC ile çıkış
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                print("👋 Kullanıcı tarafından durduruldu")
                break
                
    except KeyboardInterrupt:
        print("\n👋 Ctrl+C ile durduruldu")
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass
        print("🏁 Test tamamlandı")

if __name__ == "__main__":
    main()
