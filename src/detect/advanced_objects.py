"""Advanced object detection with zero-shot capabilities."""

import cv2
import numpy as np
import torch
import open_clip
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)


class AdvancedObjectDetector:
    """Advanced object detector with zero-shot and segmentation capabilities."""
    
    def __init__(
        self,
        segmentation_model_path: str = "models/advanced/yolov8l-seg.pt",
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        device: str = "auto"
    ):
        """
        Initialize advanced object detector.
        
        Args:
            segmentation_model_path: Path to YOLOv8 segmentation model
            clip_model_name: CLIP model architecture
            clip_pretrained: CLIP pretrained weights
            device: Device to use ("cpu", "cuda", "auto")
        """
        self.device = self._setup_device(device)
        
        # Initialize YOLOv8 segmentation model
        logger.info(f"Loading YOLOv8 segmentation model: {segmentation_model_path}")
        self.yolo_seg = YOLO(segmentation_model_path)
        
        # Initialize CLIP model for zero-shot detection
        logger.info(f"Loading CLIP model: {clip_model_name}")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained, device=self.device
        )
        self.clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
        
        logger.info("✅ Advanced object detector initialized successfully!")
        
    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using device: {device}")
        return device
    
    def detect_in_hand_objects(
        self,
        image: np.ndarray,
        hand_bbox: Tuple[int, int, int, int],
        expansion_factor: float = 1.3,
        conf_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Detect objects specifically in hand region.
        
        Args:
            image: Input image
            hand_bbox: Hand bounding box (x, y, w, h)
            expansion_factor: Factor to expand hand region
            conf_threshold: Confidence threshold
            
        Returns:
            List of detected objects in hand
        """
        # Expand hand region
        x, y, w, h = hand_bbox
        center_x, center_y = x + w // 2, y + h // 2
        
        new_w = int(w * expansion_factor)
        new_h = int(h * expansion_factor)
        
        crop_x = max(0, center_x - new_w // 2)
        crop_y = max(0, center_y - new_h // 2)
        crop_x2 = min(image.shape[1], crop_x + new_w)
        crop_y2 = min(image.shape[0], crop_y + new_h)
        
        # Crop hand region
        hand_crop = image[crop_y:crop_y2, crop_x:crop_x2]
        
        if hand_crop.size == 0:
            return []
        
        # Run YOLOv8 segmentation on hand crop
        results = self.yolo_seg(hand_crop, conf=conf_threshold, verbose=False)
        
        detections = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                # Get masks if available
                masks = None
                if result.masks is not None:
                    masks = result.masks.xy  # Polygon format
                
                for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                    if conf >= conf_threshold:
                        # Convert coordinates back to full image
                        x1, y1, x2, y2 = box
                        full_x1 = int(crop_x + x1)
                        full_y1 = int(crop_y + y1)
                        full_x2 = int(crop_x + x2)
                        full_y2 = int(crop_y + y2)
                        
                        detection = {
                            'bbox': (full_x1, full_y1, full_x2 - full_x1, full_y2 - full_y1),
                            'confidence': float(conf),
                            'class_id': int(cls),
                            'class_name': self.yolo_seg.names[int(cls)],
                            'mask': masks[i] if masks is not None else None,
                            'in_hand': True
                        }
                        
                        detections.append(detection)
        
        return detections
    
    def zero_shot_detect(
        self,
        image: np.ndarray,
        text_queries: List[str],
        region_bbox: Optional[Tuple[int, int, int, int]] = None,
        similarity_threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Zero-shot object detection using CLIP.
        
        Args:
            image: Input image
            text_queries: List of text descriptions to search for
            region_bbox: Optional region to search in (x, y, w, h)
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of detected objects with similarity scores
        """
        # Use region if specified
        if region_bbox is not None:
            x, y, w, h = region_bbox
            search_image = image[y:y+h, x:x+w]
            offset_x, offset_y = x, y
        else:
            search_image = image
            offset_x, offset_y = 0, 0
        
        # Preprocess image for CLIP
        rgb_image = cv2.cvtColor(search_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Apply CLIP preprocessing
        processed_image = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Tokenize text queries
        text_tokens = self.clip_tokenizer(text_queries).to(self.device)
        
        # Compute similarities
        with torch.no_grad():
            image_features = self.clip_model.encode_image(processed_image)
            text_features = self.clip_model.encode_text(text_tokens)
            
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarities
            similarities = (image_features @ text_features.T).squeeze(0)
        
        detections = []
        
        for i, (query, similarity) in enumerate(zip(text_queries, similarities)):
            if similarity >= similarity_threshold:
                # For CLIP, we return the whole search region as detection
                # In a more sophisticated implementation, you would use
                # techniques like gradient-based attention maps
                detection = {
                    'bbox': (offset_x, offset_y, search_image.shape[1], search_image.shape[0]),
                    'confidence': float(similarity),
                    'class_name': query,
                    'similarity': float(similarity),
                    'method': 'zero_shot_clip'
                }
                detections.append(detection)
        
        return detections
    
    def analyze_hand_object_interaction(
        self,
        image: np.ndarray,
        hand_landmarks: np.ndarray,
        hand_bbox: Tuple[int, int, int, int],
        objects: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze interaction between hand and detected objects.
        
        Args:
            image: Input image
            hand_landmarks: Hand landmarks (21x3)
            hand_bbox: Hand bounding box
            objects: Detected objects
            
        Returns:
            Interaction analysis results
        """
        analysis = {
            'holding_object': False,
            'interaction_type': 'none',
            'grip_strength': 0.0,
            'object_in_hand': None,
            'finger_positions': {},
            'hand_pose': 'unknown'
        }
        
        if len(objects) == 0:
            return analysis
        
        # Analyze finger positions
        fingertip_indices = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        fingertips = hand_landmarks[fingertip_indices]
        
        # Calculate grip strength based on finger closure
        palm_center = np.mean(hand_landmarks[[0, 5, 9, 13, 17]], axis=0)
        
        distances_to_palm = []
        for fingertip in fingertips:
            dist = np.linalg.norm(fingertip[:2] - palm_center[:2])
            distances_to_palm.append(dist)
        
        # Normalized grip strength (lower distances = stronger grip)
        avg_distance = np.mean(distances_to_palm)
        max_expected_distance = 80  # pixels, adjust based on hand size
        grip_strength = max(0.0, 1.0 - (avg_distance / max_expected_distance))
        
        analysis['grip_strength'] = float(grip_strength)
        
        # Find closest object to hand
        closest_object = None
        min_distance = float('inf')
        
        hand_center = (hand_bbox[0] + hand_bbox[2]//2, hand_bbox[1] + hand_bbox[3]//2)
        
        for obj in objects:
            obj_bbox = obj['bbox']
            obj_center = (obj_bbox[0] + obj_bbox[2]//2, obj_bbox[1] + obj_bbox[3]//2)
            
            distance = np.sqrt(
                (hand_center[0] - obj_center[0])**2 + 
                (hand_center[1] - obj_center[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_object = obj
        
        # Determine interaction
        if closest_object and min_distance < 50:  # pixels
            analysis['holding_object'] = True
            analysis['object_in_hand'] = closest_object
            
            if grip_strength > 0.6:
                analysis['interaction_type'] = 'grasping'
            elif grip_strength > 0.3:
                analysis['interaction_type'] = 'holding'
            else:
                analysis['interaction_type'] = 'touching'
        
        # Basic hand pose classification
        if grip_strength > 0.7:
            analysis['hand_pose'] = 'closed_fist'
        elif grip_strength < 0.2:
            analysis['hand_pose'] = 'open_hand'
        else:
            analysis['hand_pose'] = 'partial_grip'
        
        return analysis
    
    def visualize_advanced_detection(
        self,
        image: np.ndarray,
        hand_detections: List[Any],
        object_detections: List[Dict[str, Any]],
        interaction_analysis: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Visualize advanced detection results.
        
        Args:
            image: Input image
            hand_detections: Hand detection results
            object_detections: Object detection results
            interaction_analysis: Interaction analysis results
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Draw hand detections
        for hand in hand_detections:
            x, y, w, h = hand.bbox
            
            # Hand bounding box (blue)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Hand landmarks if available
            if hand.landmarks:
                landmarks_px = hand.landmarks.landmarks
                
                # Draw landmarks
                for landmark in landmarks_px:
                    px, py = int(landmark[0]), int(landmark[1])
                    cv2.circle(annotated, (px, py), 3, (255, 255, 0), -1)
                
                # Draw connections
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
                    (0, 5), (5, 6), (6, 7), (7, 8),  # index
                    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
                    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
                    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
                    (5, 9), (9, 13), (13, 17)  # palm
                ]
                
                for start_idx, end_idx in connections:
                    if start_idx < len(landmarks_px) and end_idx < len(landmarks_px):
                        start_point = (int(landmarks_px[start_idx][0]), int(landmarks_px[start_idx][1]))
                        end_point = (int(landmarks_px[end_idx][0]), int(landmarks_px[end_idx][1]))
                        cv2.line(annotated, start_point, end_point, (0, 255, 255), 1)
        
        # Draw object detections
        for obj in object_detections:
            x, y, w, h = obj['bbox']
            conf = obj['confidence']
            class_name = obj['class_name']
            
            # Object bounding box (green for in-hand, red for general)
            color = (0, 255, 0) if obj.get('in_hand', False) else (0, 0, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Label
            label = f"{class_name}: {conf:.2f}"
            if 'similarity' in obj:
                label += f" (sim: {obj['similarity']:.2f})"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annotated, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            cv2.putText(annotated, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw mask if available
            if 'mask' in obj and obj['mask'] is not None:
                mask = obj['mask']
                if isinstance(mask, np.ndarray) and len(mask) > 0:
                    pts = mask.astype(np.int32)
                    cv2.polylines(annotated, [pts], True, color, 2)
        
        # Draw interaction analysis
        if interaction_analysis:
            y_offset = 30
            
            # Grip strength bar
            grip = interaction_analysis['grip_strength']
            bar_width = 200
            bar_height = 20
            bar_x, bar_y = 10, y_offset
            
            cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            cv2.rectangle(annotated, (bar_x, bar_y), 
                         (bar_x + int(bar_width * grip), bar_y + bar_height), (0, 255, 0), -1)
            cv2.putText(annotated, f"Grip: {grip:.2f}", (bar_x, bar_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 50
            
            # Interaction type
            interaction = interaction_analysis['interaction_type']
            cv2.putText(annotated, f"Interaction: {interaction}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            y_offset += 25
            
            # Hand pose
            pose = interaction_analysis['hand_pose']
            cv2.putText(annotated, f"Pose: {pose}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        return annotated


def create_advanced_detector(**kwargs) -> AdvancedObjectDetector:
    """Create advanced object detector with default parameters."""
    return AdvancedObjectDetector(**kwargs)
