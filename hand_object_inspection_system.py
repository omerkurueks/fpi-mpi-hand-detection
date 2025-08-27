#!/usr/bin/env python3
"""
Elde Nesne İnceleme Tespiti - Tam Sistem
MediaPipe + YOLO + Motion Tracking + FSM ile inceleme süresi hesaplama
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import json
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
from ultralytics import YOLO
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HandDetection:
    bbox: List[int]  # [x1, y1, x2, y2]
    landmarks: any
    confidence: float
    hand_type: str  # "Left" or "Right"

@dataclass
class ObjectDetection:
    bbox: List[int]  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float

@dataclass
class HandObjectPair:
    hand: HandDetection
    object: ObjectDetection
    iou_score: float
    distance: float

@dataclass
class InspectionEvent:
    event_id: str
    start_time: float
    end_time: Optional[float]
    duration: Optional[float]
    hand_bbox: List[int]
    object_bbox: List[int]
    object_class: str
    hand_type: str
    motion_score: float
    is_active: bool

class MediaPipeHandDetector:
    """MediaPipe ile gelişmiş el tespiti"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
    def detect(self, frame) -> List[HandDetection]:
        """El tespiti yap"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hands = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # El bounding box hesapla
                h, w, _ = frame.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                
                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))
                
                # Padding ekle
                padding = 20
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                
                hand_type = handedness.classification[0].label
                confidence = handedness.classification[0].score
                
                hands.append(HandDetection(
                    bbox=[x1, y1, x2, y2],
                    landmarks=hand_landmarks,
                    confidence=confidence,
                    hand_type=hand_type
                ))
                
        return hands

class YOLOObjectDetector:
    """YOLO ile nesne tespiti"""
    
    def __init__(self, model_path="yolov8n.pt"):
        try:
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            logger.info(f"YOLO model loaded: {model_path}")
        except Exception as e:
            logger.error(f"YOLO model yüklenemedi: {e}")
            self.model = None
            self.class_names = {}
    
    def detect(self, frame, confidence_threshold=0.5) -> List[ObjectDetection]:
        """Nesne tespiti yap"""
        if self.model is None:
            return []
            
        try:
            results = self.model(frame, conf=confidence_threshold, verbose=False)
            
            objects = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        class_name = self.class_names.get(cls, f"class_{cls}")
                        
                        objects.append(ObjectDetection(
                            bbox=[int(x1), int(y1), int(x2), int(y2)],
                            class_id=cls,
                            class_name=class_name,
                            confidence=float(conf)
                        ))
            
            return objects
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []

class HandObjectMatcher:
    """El-nesne eşleştirme"""
    
    @staticmethod
    def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
        """IoU hesaplama"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def calculate_distance(bbox1: List[int], bbox2: List[int]) -> float:
        """Merkez nokta mesafesi"""
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2
        
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    def match_hands_objects(self, hands: List[HandDetection], 
                           objects: List[ObjectDetection],
                           iou_threshold: float = 0.1,
                           distance_threshold: float = 200) -> List[HandObjectPair]:
        """El-nesne eşleştirme"""
        pairs = []
        
        for hand in hands:
            best_match = None
            best_score = 0
            
            for obj in objects:
                # IoU hesapla
                iou = self.calculate_iou(hand.bbox, obj.bbox)
                
                # Mesafe hesapla
                distance = self.calculate_distance(hand.bbox, obj.bbox)
                
                # Skorlama: IoU öncelikli, mesafe de dikkate alınır
                if iou > iou_threshold or distance < distance_threshold:
                    score = iou + (1.0 / (1.0 + distance / 100))  # Normalize distance
                    
                    if score > best_score:
                        best_score = score
                        best_match = HandObjectPair(
                            hand=hand,
                            object=obj,
                            iou_score=iou,
                            distance=distance
                        )
            
            if best_match:
                pairs.append(best_match)
        
        return pairs

class MotionTracker:
    """Hareket takibi"""
    
    def __init__(self):
        self.prev_frame = None
        self.optical_flow = cv2.createOptFlow_DIS()
        
    def calculate_motion_score(self, frame: np.ndarray, bbox: List[int]) -> float:
        """Bounding box içindeki hareket skorunu hesapla"""
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return 0.0
        
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        try:
            # Optical flow hesapla
            flow = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, current_gray, None, None
            )
            
            # ROI içindeki flow
            x1, y1, x2, y2 = bbox
            roi_flow = flow[y1:y2, x1:x2] if flow is not None else None
            
            if roi_flow is not None and roi_flow.size > 0:
                magnitude = np.sqrt(roi_flow[..., 0]**2 + roi_flow[..., 1]**2)
                motion_score = np.mean(magnitude)
            else:
                motion_score = 0.0
                
        except Exception as e:
            logger.debug(f"Motion calculation error: {e}")
            motion_score = 0.0
        
        self.prev_frame = current_gray.copy()
        return float(motion_score)

class InspectionFSM:
    """Sonlu Durum Makinesi - İnceleme davranışı tespiti"""
    
    def __init__(self, 
                 motion_threshold: float = 1.0,
                 min_inspection_time: float = 2.0,
                 max_idle_time: float = 3.0):
        self.motion_threshold = motion_threshold
        self.min_inspection_time = min_inspection_time
        self.max_idle_time = max_idle_time
        
        self.active_events: Dict[str, InspectionEvent] = {}
        self.completed_events: List[InspectionEvent] = []
        self.event_counter = 0
        
    def update(self, pairs: List[HandObjectPair], motion_scores: List[float]) -> List[InspectionEvent]:
        """FSM güncelle"""
        current_time = time.time()
        
        # Aktif eşleştirmeler
        active_pairs = {}
        for i, pair in enumerate(pairs):
            motion_score = motion_scores[i] if i < len(motion_scores) else 0.0
            
            # Unique key oluştur
            key = f"{pair.hand.hand_type}_{pair.object.class_name}"
            active_pairs[key] = (pair, motion_score)
        
        # Mevcut event'leri kontrol et
        events_to_remove = []
        for event_id, event in self.active_events.items():
            if event_id not in active_pairs:
                # Event artık aktif değil
                if current_time - event.start_time >= self.min_inspection_time:
                    # Minimum süreyi geçti, tamamla
                    event.end_time = current_time
                    event.duration = event.end_time - event.start_time
                    event.is_active = False
                    self.completed_events.append(event)
                    logger.info(f"Inceleme tamamlandı: {event.duration:.1f} saniye - {event.object_class}")
                
                events_to_remove.append(event_id)
        
        # Tamamlanan event'leri kaldır
        for event_id in events_to_remove:
            del self.active_events[event_id]
        
        # Yeni event'leri kontrol et
        for key, (pair, motion_score) in active_pairs.items():
            if key not in self.active_events and motion_score > self.motion_threshold:
                # Yeni inceleme başlat
                self.event_counter += 1
                event = InspectionEvent(
                    event_id=f"inspection_{self.event_counter}_{int(current_time)}",
                    start_time=current_time,
                    end_time=None,
                    duration=None,
                    hand_bbox=pair.hand.bbox.copy(),
                    object_bbox=pair.object.bbox.copy(),
                    object_class=pair.object.class_name,
                    hand_type=pair.hand.hand_type,
                    motion_score=motion_score,
                    is_active=True
                )
                self.active_events[key] = event
                logger.info(f"Yeni inceleme başladı: {pair.object.class_name} - {pair.hand.hand_type} el")
        
        # Aktif event'lerin motion score'unu güncelle
        for key, (pair, motion_score) in active_pairs.items():
            if key in self.active_events:
                self.active_events[key].motion_score = motion_score
        
        return list(self.active_events.values())

class HandObjectInspectionSystem:
    """Ana sistem sınıfı"""
    
    def __init__(self):
        self.hand_detector = MediaPipeHandDetector()
        self.object_detector = YOLOObjectDetector()
        self.matcher = HandObjectMatcher()
        self.motion_tracker = MotionTracker()
        self.fsm = InspectionFSM()
        
        logger.info("El-Nesne İnceleme Sistemi başlatıldı")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[HandObjectPair], List[InspectionEvent]]:
        """Frame işle"""
        # El tespiti
        hands = self.hand_detector.detect(frame)
        
        # Nesne tespiti
        objects = self.object_detector.detect(frame)
        
        # El-nesne eşleştirme
        pairs = self.matcher.match_hands_objects(hands, objects)
        
        # Hareket skorları
        motion_scores = []
        for pair in pairs:
            # Combined bbox için hareket hesapla
            combined_bbox = [
                min(pair.hand.bbox[0], pair.object.bbox[0]),
                min(pair.hand.bbox[1], pair.object.bbox[1]),
                max(pair.hand.bbox[2], pair.object.bbox[2]),
                max(pair.hand.bbox[3], pair.object.bbox[3])
            ]
            motion_score = self.motion_tracker.calculate_motion_score(frame, combined_bbox)
            motion_scores.append(motion_score)
        
        # FSM güncelle
        active_events = self.fsm.update(pairs, motion_scores)
        
        return pairs, active_events
    
    def draw_results(self, frame: np.ndarray, pairs: List[HandObjectPair], 
                    events: List[InspectionEvent]) -> np.ndarray:
        """Sonuçları çiz"""
        # El-nesne çiftlerini çiz
        for i, pair in enumerate(pairs):
            hand = pair.hand
            obj = pair.object
            
            # El bbox
            cv2.rectangle(frame, (hand.bbox[0], hand.bbox[1]), 
                         (hand.bbox[2], hand.bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{hand.hand_type} Hand", 
                       (hand.bbox[0], hand.bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Nesne bbox
            cv2.rectangle(frame, (obj.bbox[0], obj.bbox[1]), 
                         (obj.bbox[2], obj.bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, f"{obj.class_name} ({obj.confidence:.2f})", 
                       (obj.bbox[0], obj.bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Bağlantı çizgisi
            hand_center = ((hand.bbox[0] + hand.bbox[2])//2, 
                          (hand.bbox[1] + hand.bbox[3])//2)
            obj_center = ((obj.bbox[0] + obj.bbox[2])//2, 
                         (obj.bbox[1] + obj.bbox[3])//2)
            cv2.line(frame, hand_center, obj_center, (0, 255, 255), 2)
            
            # IoU score
            cv2.putText(frame, f"IoU: {pair.iou_score:.2f}", 
                       (obj_center[0], obj_center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Aktif incelemeler
        y_offset = 30
        for event in events:
            if event.is_active:
                duration = time.time() - event.start_time
                text = f"İnceleme: {event.object_class} - {duration:.1f}s"
                cv2.putText(frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30
        
        # Tamamlanan incelemeler (son 5)
        recent_completed = self.fsm.completed_events[-5:]
        for i, event in enumerate(recent_completed):
            text = f"Tamamlanan: {event.object_class} - {event.duration:.1f}s"
            cv2.putText(frame, text, (10, frame.shape[0] - 150 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame

def main():
    """Ana fonksiyon"""
    # RTSP URL
    rtsp_url = "rtsp://admin:HeysemAI246@192.168.150.59"
    
    print(f"🔗 RTSP bağlantısı: {rtsp_url}")
    
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("❌ RTSP bağlantısı başarısız!")
        return
    
    print("✅ RTSP bağlantısı başarılı!")
    
    # Sistem başlat
    system = HandObjectInspectionSystem()
    
    # FPS sayacı
    fps_counter = 0
    start_time = time.time()
    
    print("🚀 El-Nesne İnceleme Tespiti başlatılıyor...")
    print("'q' ile çıkış, 's' ile screenshot")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("⚠️ Frame okunamadı!")
                break
            
            # Frame işle
            pairs, events = system.process_frame(frame)
            
            # Sonuçları çiz
            frame = system.draw_results(frame, pairs, events)
            
            # FPS hesapla
            fps_counter += 1
            elapsed = time.time() - start_time
            if elapsed > 1.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                start_time = time.time()
                
                # FPS ve istatistikler
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Konsol istatistikleri
                active_count = len(events)
                completed_count = len(system.fsm.completed_events)
                print(f"📊 FPS: {fps:.1f} | Aktif: {active_count} | Tamamlanan: {completed_count}")
            
            # Göster
            cv2.imshow('El-Nesne İnceleme Tespiti', frame)
            
            # Klavye kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"screenshot_{timestamp}.jpg", frame)
                print(f"📸 Screenshot kaydedildi: screenshot_{timestamp}.jpg")
    
    except KeyboardInterrupt:
        print("\n⚠️ Kullanıcı tarafından durduruldu")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Final istatistikler
        total_events = len(system.fsm.completed_events)
        if total_events > 0:
            total_time = sum(e.duration for e in system.fsm.completed_events)
            avg_time = total_time / total_events
            print(f"\n📈 Final İstatistikler:")
            print(f"   Toplam inceleme: {total_events}")
            print(f"   Toplam süre: {total_time:.1f} saniye")
            print(f"   Ortalama süre: {avg_time:.1f} saniye")
            
            # JSON olarak kaydet
            events_data = []
            for event in system.fsm.completed_events:
                events_data.append({
                    'event_id': event.event_id,
                    'start_time': event.start_time,
                    'duration': event.duration,
                    'object_class': event.object_class,
                    'hand_type': event.hand_type,
                    'motion_score': event.motion_score
                })
            
            with open('inspection_results.json', 'w') as f:
                json.dump(events_data, f, indent=2)
            print(f"💾 Sonuçlar kaydedildi: inspection_results.json")
        
        print("✅ Sistem kapatıldı!")

if __name__ == "__main__":
    main()
