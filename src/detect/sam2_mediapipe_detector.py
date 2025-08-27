"""
SAM 2 + MediaPipe El Tespiti ve Segmentasyon Sistemi
Zero-shot nesne segmentasyonu ve el eklem noktaları ile etkileşim analizi
"""

import cv2
import numpy as np
import torch
import mediapipe as mp
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import time

# SAM 2 imports
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2MediaPipeDetector:
    def __init__(self, 
                 sam2_checkpoint: str = "models/sam2/sam2.1_hiera_large.pt",
                 sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
                 device: str = "cuda"):
        """
        SAM 2 + MediaPipe Detector initialization
        """
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"🚀 SAM2MediaPipeDetector başlatılıyor...")
        self.logger.info(f"📁 Checkpoint: {sam2_checkpoint}")
        self.logger.info(f"⚙️ Config: {sam2_config}")
        self.logger.info(f"🖥️ Device: {device}")

class SAM2MediaPipeDetector:
    def __init__(self, 
                 sam2_checkpoint: str = "models/sam2/sam2.1_hiera_large.pt",
                 sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
                 device: str = "cpu"):
        """
        SAM 2 + MediaPipe Detector
        
        Args:
            sam2_checkpoint: SAM 2 model checkpoint yolu
            sam2_config: SAM 2 konfigürasyon dosyası yolu
            device: Hesaplama cihazı (cuda/cpu)
        """
        self.device = device
        self.logger = logging.getLogger(__name__)
        
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
        
        # SAM 2 setup
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.sam2_predictor = None
        self.video_predictor = None
        
        # Initialize SAM 2
        self._init_sam2()
        
        # El landmark noktaları
        self.hand_landmarks_names = [
            'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]
        
    def _init_sam2(self):
        """SAM 2 modelini başlat - Resmi API kullanarak"""
        try:
            if not Path(self.sam2_checkpoint).exists():
                self.logger.error(f"❌ SAM 2 checkpoint bulunamadı: {self.sam2_checkpoint}")
                self.sam2_predictor = None
                self.video_predictor = None
                return
                
            if not Path(self.sam2_config).exists():
                self.logger.error(f"❌ SAM 2 config bulunamadı: {self.sam2_config}")
                self.sam2_predictor = None
                self.video_predictor = None
                return
                
            self.logger.info(f"� SAM 2 model yükleniyor: {self.sam2_checkpoint}")
            self.logger.info(f"⚙️ Config: {self.sam2_config}")
            self.logger.info(f"🖥️ Device: {self.device}")
            
            # Device kontrolü ve otomatik CUDA detection
            if self.device == "cuda" and not torch.cuda.is_available():
                self.logger.warning("⚠️ CUDA istendi ama kullanılamıyor, CPU'ya geçiliyor")
                self.device = "cpu"
            elif self.device == "cuda" and torch.cuda.is_available():
                self.logger.info(f"✅ CUDA kullanılacak - GPU: {torch.cuda.get_device_name()}")
            
            # SAM 2 imports
            from sam2.build_sam import build_sam2, build_sam2_video_predictor
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self.logger.info("✅ SAM 2 imports başarılı")
            
            # Build SAM 2 model - Resmi API
            self.logger.info("🔨 SAM 2 model build ediliyor...")
            sam2_model = build_sam2(self.sam2_config, self.sam2_checkpoint)
            self.logger.info("✅ SAM 2 model başarıyla build edildi")
            
            # Create image predictor
            self.logger.info("🔮 SAM 2 Image Predictor oluşturuluyor...")
            self.sam2_predictor = SAM2ImagePredictor(sam2_model)
            self.logger.info("✅ SAM 2 Image Predictor oluşturuldu")
            
            # Create video predictor
            self.logger.info("📹 SAM 2 Video Predictor oluşturuluyor...")
            self.video_predictor = build_sam2_video_predictor(
                self.sam2_config, self.sam2_checkpoint
            )
            self.logger.info("✅ SAM 2 Video Predictor oluşturuldu")
            
            self.logger.info("🎉 SAM 2 başarıyla başlatıldı!")
            
        except Exception as e:
            self.logger.error(f"❌ SAM 2 başlatma hatası: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback: SAM 2 olmadan devam et
            self.sam2_predictor = None
            self.video_predictor = None
            self.logger.info("⚠️ SAM 2 olmadan sadece MediaPipe ile devam ediliyor")
            
    def detect_hands_and_segments(self, image: np.ndarray) -> Dict[str, Any]:
        """
        El tespiti ve segmentasyon analizi
        
        Args:
            image: Giriş görüntüsü
            
        Returns:
            Detection sonuçları
        """
        results = {
            'hands': [],
            'segments': [],
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
                
                # El bölgesinde SAM 2 segmentasyon
                if self.sam2_predictor is not None:
                    self.logger.info(f"🎯 El {idx+1} için SAM 2 segmentasyon çalıştırılıyor...")
                    segments = self._segment_objects_in_hand_region(image, hand_info)
                    results['segments'].extend(segments)
                    self.logger.info(f"✅ {len(segments)} segment bulundu")
                    
                    # El-nesne etkileşimi analizi
                    interactions = self._analyze_hand_object_interaction(
                        hand_info, segments
                    )
                    results['hand_object_interactions'].extend(interactions)
                else:
                    self.logger.warning("⚠️ SAM 2 predictor başlatılamadı, segmentasyon atlanıyor")
        
        # Sonuçları görselleştir
        results['annotated_image'] = self._annotate_results(
            image, results['hands'], results['segments'], results['hand_object_interactions']
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
        # Başparmak (4) ve işaret parmağı (8) pozisyonlarına göre
        thumb_tip = landmarks_2d[4]      # Başparmak ucu
        index_mcp = landmarks_2d[5]      # İşaret parmağı MCP
        
        # Başparmak solda ise sağ el, sağda ise sol el (kamera görünümü)
        if thumb_tip[0] < index_mcp[0]:
            corrected_label = "Right"
        else:
            corrected_label = "Left"
            
        # Log if correction was made
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
            'label': corrected_label,  # Düzeltilmiş label
            'original_label': original_label,  # Orijinal MediaPipe label
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
        # Parmak durumları (açık/kapalı)
        fingers_up = []
        
        # Başparmak (el türüne göre)
        if hand_label == "Right":
            # Sağ el için: başparmak sağda olmalı
            if landmarks_2d[4][0] > landmarks_2d[3][0]:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        else:
            # Sol el için: başparmak solda olmalı
            if landmarks_2d[4][0] < landmarks_2d[3][0]:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
            
        # Diğer parmaklar (yukarı/aşağı)
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
    
    def _segment_objects_in_hand_region(self, image, hand_info):
        """El bölgesinde SAM 2 ile nesne segmentasyonu"""
        segments = []
        
        # SAM 2 predictor kontrolü
        if self.sam2_predictor is None:
            self.logger.warning("⚠️ SAM 2 predictor yok, segmentasyon atlanıyor")
            return segments
        
        try:
            self.logger.info("🔍 SAM 2 segmentasyon başlatılıyor...")
            
            # El bounding box'ını genişlet
            bbox = hand_info['bbox']
            margin = 50
            
            x1 = max(0, bbox['x_min'] - margin)
            y1 = max(0, bbox['y_min'] - margin)
            x2 = min(image.shape[1], bbox['x_max'] + margin)
            y2 = min(image.shape[0], bbox['y_max'] + margin)
            
            self.logger.info(f"📦 Segmentasyon bölgesi: ({x1},{y1}) - ({x2},{y2})")
            
            # SAM 2 ile otomatik segmentasyon
            self.sam2_predictor.set_image(image)
            self.logger.info("🖼️ Image SAM 2'ye set edildi")
            
            # El merkezi etrafında multiple point prompts
            center_x, center_y = hand_info['center']
            self.logger.info(f"📍 El merkezi: ({center_x}, {center_y})")
            
            # Grid pattern ile point prompts
            points = []
            for dx in [-30, 0, 30]:
                for dy in [-30, 0, 30]:
                    px = center_x + dx
                    py = center_y + dy
                    if x1 <= px <= x2 and y1 <= py <= y2:
                        points.append([px, py])
            
            self.logger.info(f"🎯 {len(points)} prompt point oluşturuldu")
            
            if points:
                input_points = np.array(points)
                input_labels = np.ones(len(points))
                
                self.logger.info("🔮 SAM 2 prediction çalıştırılıyor...")
                masks, scores, _ = self.sam2_predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True
                )
                
                self.logger.info(f"🎭 {len(masks)} mask, {len(scores)} score döndü")
                
                # En iyi maskeleri seç
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    self.logger.info(f"🎭 Mask {i}: score={score:.3f}")
                    if score > 0.5:  # Güven eşiği
                        segments.append({
                            'mask': mask,
                            'score': score,
                            'bbox': self._mask_to_bbox(mask),
                            'area': np.sum(mask),
                            'hand_id': hand_info['hand_id']
                        })
                        self.logger.info(f"✅ Mask {i} kabul edildi (score: {score:.3f})")
                    else:
                        self.logger.info(f"❌ Mask {i} reddedildi (score: {score:.3f})")
            else:
                self.logger.warning("⚠️ Hiç prompt point oluşturulamadı")
                        
        except Exception as e:
            self.logger.error(f"❌ SAM 2 segmentasyon hatası: {e}")
            import traceback
            traceback.print_exc()
        
        return segments
    
    def _mask_to_bbox(self, mask):
        """Mask'tan bounding box çıkar"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return {
            'x_min': int(cmin),
            'y_min': int(rmin),
            'x_max': int(cmax),
            'y_max': int(rmax)
        }
    
    def _analyze_hand_object_interaction(self, hand_info, segments):
        """El-nesne etkileşimi analizi"""
        interactions = []
        
        for segment in segments:
            if segment['hand_id'] == hand_info['hand_id']:
                # Mesafe analizi
                hand_center = hand_info['center']
                
                if segment['bbox']:
                    obj_center = [
                        (segment['bbox']['x_min'] + segment['bbox']['x_max']) // 2,
                        (segment['bbox']['y_min'] + segment['bbox']['y_max']) // 2
                    ]
                    
                    distance = np.sqrt(
                        (hand_center[0] - obj_center[0])**2 + 
                        (hand_center[1] - obj_center[1])**2
                    )
                    
                    # Etkileşim türü
                    if distance < 50:
                        interaction_type = "grasping"
                    elif distance < 100:
                        interaction_type = "touching"
                    else:
                        interaction_type = "near"
                    
                    interactions.append({
                        'hand_id': hand_info['hand_id'],
                        'segment': segment,
                        'distance': distance,
                        'interaction_type': interaction_type,
                        'hand_pose': hand_info['pose_info']['pose']
                    })
        
        return interactions
    
    def _annotate_results(self, image, hands, segments, interactions):
        """Sonuçları görselleştir"""
        annotated = image.copy()
        
        # El landmark'larını çiz
        for hand in hands:
            # Landmark noktaları
            for i, (x, y) in enumerate(hand['landmarks_2d']):
                cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(annotated, str(i), (x+5, y+5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # El bağlantıları çiz (MediaPipe connections)
            connections = self.mp_hands.HAND_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                start_point = hand['landmarks_2d'][start_idx]
                end_point = hand['landmarks_2d'][end_idx]
                cv2.line(annotated, tuple(start_point), tuple(end_point), (0, 255, 0), 2)
            
            # El bilgileri
            cv2.putText(annotated, 
                       f"{hand['label']} ({hand['confidence']:.2f})", 
                       (hand['bbox']['x_min'], hand['bbox']['y_min'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # El pozu
            cv2.putText(annotated, 
                       f"Pose: {hand['pose_info']['pose']}", 
                       (hand['bbox']['x_min'], hand['bbox']['y_max'] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Segmentasyon maskelerini çiz
        for segment in segments:
            if segment['bbox']:
                bbox = segment['bbox']
                cv2.rectangle(annotated, 
                            (bbox['x_min'], bbox['y_min']),
                            (bbox['x_max'], bbox['y_max']),
                            (255, 0, 255), 2)
                
                cv2.putText(annotated, 
                           f"Obj ({segment['score']:.2f})", 
                           (bbox['x_min'], bbox['y_min'] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Etkileşimleri çiz
        for interaction in interactions:
            hand_id = interaction['hand_id']
            hand = hands[hand_id]
            segment = interaction['segment']
            
            if segment['bbox']:
                # Etkileşim çizgisi
                hand_center = tuple(hand['center'])
                obj_center = (
                    (segment['bbox']['x_min'] + segment['bbox']['x_max']) // 2,
                    (segment['bbox']['y_min'] + segment['bbox']['y_max']) // 2
                )
                
                color = (0, 255, 255) if interaction['interaction_type'] == "grasping" else (0, 165, 255)
                cv2.line(annotated, hand_center, obj_center, color, 2)
                
                # Etkileşim bilgisi
                mid_point = (
                    (hand_center[0] + obj_center[0]) // 2,
                    (hand_center[1] + obj_center[1]) // 2
                )
                cv2.putText(annotated, 
                           interaction['interaction_type'], 
                           mid_point,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return annotated
