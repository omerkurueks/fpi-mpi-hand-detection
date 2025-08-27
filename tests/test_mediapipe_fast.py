#!/usr/bin/env python3
"""
MediaPipe RTSP Test - Ultra Fast Version
Sadece MediaPipe hand detection - SAM 2 olmadan hızlı test
"""

import cv2
import mediapipe as mp
import time

def main():
    print("🚀 MediaPipe RTSP Test Başlatılıyor (Ultra Fast)...")
    
    # MediaPipe setup
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print("✅ MediaPipe hazır!")
    
    # RTSP stream ayarları
    rtsp_url = "rtsp://admin:HeysemAI246@192.168.150.59"
    print(f"🎥 RTSP stream bağlanılıyor: {rtsp_url}")
    
    # RTSP stream başlat
    cap = cv2.VideoCapture(rtsp_url)
    
    # Buffer ayarları
    try:
        cap.set(cv2.CAP_PROP_BUFFER_SIZE, 1)
    except AttributeError:
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except AttributeError:
            pass
    
    # Orta çözünürlük - hız/kalite dengesi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ RTSP stream açılamadı!")
        return
        
    print("✅ RTSP stream bağlandı")
    print("📹 Ultra fast MediaPipe processing başlıyor...")
    print("   - Sadece MediaPipe hand detection")
    print("   - Full FPS processing")
    print("   - ESC tuşu ile çıkış")
    
    frame_count = 0
    fps_start = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Frame alınamadı")
                break
                
            frame_count += 1
            
            # MediaPipe hand detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            hand_count = 0
            if results.multi_hand_landmarks:
                hand_count = len(results.multi_hand_landmarks)
                
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Hand landmarks çiz
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
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
                        cv2.putText(frame, "Pinch Gesture", 
                                  (wrist_x, wrist_y + 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # FPS hesapla ve göster
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                fps_start = time.time()
                print(f"📊 FPS: {fps:.1f}, Frame: {frame_count}, Hands: {hand_count}")
            
            # Bilgi overlay
            cv2.putText(frame, f"FPS: {30/(time.time()-fps_start+0.001):.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Hands: {hand_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Görüntüyü göster
            cv2.imshow('MediaPipe Hand Detection (Ultra Fast)', frame)
            
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
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("🏁 Test tamamlandı")

if __name__ == "__main__":
    main()
