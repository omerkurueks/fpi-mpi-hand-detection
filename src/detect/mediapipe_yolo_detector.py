"""
MediaPipe + YOLO Object Detection
SAM 2 yerine YOLO ile nesne tespiti
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import time
from ultralytics import YOLO

class MediaPipeYOLODetector:
    def __init__(self, 
                 yolo_model_path: str = "weights/yolov8n.pt",
                 device: str = "cpu"):
        """
        MediaPipe + YOLO Detector
        
        Args:
            yolo_model_path: YOLO model path
            device: Computation device (cuda/cpu)
        """
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"🚀 MediaPipeYOLODetector başlatılıyor...")
        self.logger.info(f"📁 YOLO Model: {yolo_model_path}")
        self.logger.info(f"🖥️ Device: {device}")
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # MediaPipe Hands modeli
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # YOLO setup
        self._init_yolo(yolo_model_path)
        
        # El landmark noktaları
        self.hand_landmarks_names = [
            'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]
        
    def _init_yolo(self, model_path):
        """YOLO modelini başlat"""
        try:
            if Path(model_path).exists():
                self.logger.info(f"YOLO model yükleniyor: {model_path}")
                self.yolo_model = YOLO(model_path)
                self.logger.info("✅ YOLO başarıyla başlatıldı")
            else:
                # Default model download
                self.logger.info("YOLO model indiriliyor...")
                self.yolo_model = YOLO('yolov8n.pt')
                self.logger.info("✅ YOLO default model yüklendi")
                
        except Exception as e:
            self.logger.error(f"❌ YOLO başlatma hatası: {e}")
            self.yolo_model = None
            
    def detect_hands_and_objects(self, image: np.ndarray) -> Dict[str, Any]:
        """
        El tespiti ve nesne tespiti
        
        Args:
            image: Giriş görüntüsü
            
        Returns:
            Detection sonuçları
        """
        results = {
            'hands': [],
            'objects': [],
            'hand_object_interactions': [],
            'annotated_image': image.copy()
        }
        
        # MediaPipe el tespiti
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb_image)
        
        if hand_results.multi_hand_landmarks:
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness)
            ):
                # El bilgileri
                hand_info = self._process_hand_landmarks(
                    hand_landmarks, handedness, image.shape, idx
                )
                results['hands'].append(hand_info)
        
        # YOLO nesne tespiti
        if self.yolo_model is not None:
            self.logger.info("🎯 YOLO nesne tespiti çalıştırılıyor...")
            objects = self._detect_objects_with_yolo(image)
            results['objects'].extend(objects)
            self.logger.info(f"✅ {len(objects)} nesne bulundu")
            
            # El-nesne etkileşimi analizi
            if results['hands'] and results['objects']:
                interactions = self._analyze_hand_object_interaction(
                    results['hands'], results['objects']
                )
                results['hand_object_interactions'].extend(interactions)
        else:
            self.logger.warning("⚠️ YOLO model yok, nesne tespiti atlanıyor")
        
        # Sonuçları görselleştir
        results['annotated_image'] = self._annotate_results(
            image, results['hands'], results['objects'], results['hand_object_interactions']
        )
        
        return results
    
    def _process_hand_landmarks(self, landmarks, handedness, image_shape, hand_id):
        """El landmark'larını işle"""
        h, w = image_shape[:2]
        original_label = handedness.classification[0].label
        confidence = handedness.classification[0].score
        
        # Landmark koordinatları
        landmarks_3d = []
        landmarks_2d = []
        
        for i, landmark in enumerate(landmarks.landmark):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            z = landmark.z
            
            landmarks_2d.append([x, y])
            landmarks_3d.append([x, y, z])
        
        # El sınıflandırmasını düzelt (başparmak pozisyonuna göre)
        thumb_tip = landmarks_2d[4]
        index_mcp = landmarks_2d[5]
        
        if thumb_tip[0] < index_mcp[0]:
            corrected_label = "Right"
        else:
            corrected_label = "Left"
            
        if original_label != corrected_label:
            self.logger.info(f"🔄 El sınıflandırma düzeltildi: {original_label} → {corrected_label}")
        
        # El bounding box
        x_coords = [lm[0] for lm in landmarks_2d]
        y_coords = [lm[1] for lm in landmarks_2d]
        
        bbox = {
            'x_min': min(x_coords) - 20,
            'y_min': min(y_coords) - 20,
            'x_max': max(x_coords) + 20,
            'y_max': max(y_coords) + 20
        }
        
        # El pose analizi
        pose_info = self._analyze_hand_pose(landmarks_2d, corrected_label)
        
        return {
            'hand_id': hand_id,
            'label': corrected_label,
            'original_label': original_label,
            'confidence': confidence,
            'landmarks_2d': landmarks_2d,
            'landmarks_3d': landmarks_3d,
            'bbox': bbox,
            'pose_info': pose_info,
            'center': [
                int(sum(x_coords) / len(x_coords)),
                int(sum(y_coords) / len(y_coords))
            ]
        }
    
    def _analyze_hand_pose(self, landmarks_2d, hand_label):
        """El pozunu analiz et"""
        fingers_up = []
        
        # Başparmak
        if hand_label == "Right":
            if landmarks_2d[4][0] > landmarks_2d[3][0]:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        else:
            if landmarks_2d[4][0] < landmarks_2d[3][0]:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
            
        # Diğer parmaklar
        for tip_id in [8, 12, 16, 20]:
            if landmarks_2d[tip_id][1] < landmarks_2d[tip_id - 2][1]:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        # El duruşu
        fingers_count = sum(fingers_up)
        if fingers_count == 0:
            pose = "fist"
        elif fingers_count == 5:
            pose = "open_hand"
        elif fingers_count == 1 and fingers_up[1]:
            pose = "pointing"
        elif fingers_count == 2 and fingers_up[1] and fingers_up[2]:
            pose = "peace"
        else:
            pose = "partial"
        
        return {
            'fingers_up': fingers_up,
            'fingers_count': fingers_count,
            'pose': pose
        }
    
    def _detect_objects_with_yolo(self, image):
        """YOLO ile nesne tespiti"""
        objects = []
        
        try:
            # YOLO prediction
            results = self.yolo_model(image, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Koordinatlar
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Class name
                        class_name = self.yolo_model.names[class_id]
                        
                        # Güven eşiği
                        if confidence > 0.5:
                            objects.append({
                                'class_id': class_id,
                                'class_name': class_name,
                                'confidence': float(confidence),
                                'bbox': {
                                    'x_min': int(x1),
                                    'y_min': int(y1),
                                    'x_max': int(x2),
                                    'y_max': int(y2)
                                },
                                'center': [
                                    int((x1 + x2) / 2),
                                    int((y1 + y2) / 2)
                                ],
                                'area': int((x2 - x1) * (y2 - y1))
                            })
                            
        except Exception as e:
            self.logger.error(f"❌ YOLO detection hatası: {e}")
        
        return objects
    
    def _analyze_hand_object_interaction(self, hands, objects):
        """El-nesne etkileşimi analizi"""
        interactions = []
        
        for hand in hands:
            hand_center = hand['center']
            
            for obj in objects:
                obj_center = obj['center']
                
                # Mesafe hesapla
                distance = np.sqrt(
                    (hand_center[0] - obj_center[0])**2 + 
                    (hand_center[1] - obj_center[1])**2
                )
                
                # Etkileşim türü
                if distance < 80:
                    interaction_type = "grasping"
                elif distance < 150:
                    interaction_type = "touching"
                elif distance < 250:
                    interaction_type = "near"
                else:
                    continue
                
                interactions.append({
                    'hand_id': hand['hand_id'],
                    'hand_label': hand['label'],
                    'object_class': obj['class_name'],
                    'object_confidence': obj['confidence'],
                    'distance': distance,
                    'interaction_type': interaction_type,
                    'hand_pose': hand['pose_info']['pose']
                })
        
        return interactions
    
    def _annotate_results(self, image, hands, objects, interactions):
        """Sonuçları görselleştir"""
        annotated = image.copy()
        
        # El landmark'larını çiz
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb_image)
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # El bilgilerini yaz
        for hand in hands:
            bbox = hand['bbox']
            label = hand['label']
            pose = hand['pose_info']['pose']
            confidence = hand['confidence']
            
            # Bounding box
            cv2.rectangle(annotated, 
                         (bbox['x_min'], bbox['y_min']), 
                         (bbox['x_max'], bbox['y_max']), 
                         (0, 255, 0), 2)
            
            # Label
            text = f"{label} ({confidence:.2f}) - {pose}"
            cv2.putText(annotated, text, 
                       (bbox['x_min'], bbox['y_min'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Nesneleri çiz
        for obj in objects:
            bbox = obj['bbox']
            class_name = obj['class_name']
            confidence = obj['confidence']
            
            # Bounding box
            cv2.rectangle(annotated,
                         (bbox['x_min'], bbox['y_min']),
                         (bbox['x_max'], bbox['y_max']),
                         (255, 0, 0), 2)
            
            # Label
            text = f"{class_name} ({confidence:.2f})"
            cv2.putText(annotated, text,
                       (bbox['x_min'], bbox['y_min'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Etkileşimleri çiz
        for interaction in interactions:
            # Find corresponding hand and object
            hand = next((h for h in hands if h['hand_id'] == interaction['hand_id']), None)
            obj = next((o for o in objects if o['class_name'] == interaction['object_class']), None)
            
            if hand and obj:
                hand_center = tuple(hand['center'])
                obj_center = tuple(obj['center'])
                
                # Etkileşim türüne göre renk
                color = {
                    'grasping': (0, 0, 255),  # Kırmızı
                    'touching': (0, 165, 255),  # Turuncu
                    'near': (0, 255, 255)   # Sarı
                }.get(interaction['interaction_type'], (128, 128, 128))
                
                # Çizgi çiz
                cv2.line(annotated, hand_center, obj_center, color, 2)
                
                # Etkileşim metni
                mid_point = (
                    (hand_center[0] + obj_center[0]) // 2,
                    (hand_center[1] + obj_center[1]) // 2
                )
                cv2.putText(annotated, interaction['interaction_type'],
                           mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return annotated
