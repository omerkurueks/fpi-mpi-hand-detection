"""MediaPipe-based hand detection."""

from typing import List, Optional, Tuple, NamedTuple
import cv2
import numpy as np
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)


class HandLandmarks(NamedTuple):
    """Hand landmarks data structure."""
    landmarks: np.ndarray  # 21x3 array (x, y, z)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    handedness: str  # "Left" or "Right"


class HandDetection(NamedTuple):
    """Hand detection result."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) 
    confidence: float
    landmarks: Optional[HandLandmarks] = None
    handedness: Optional[str] = None


class MediaPipeHands:
    """MediaPipe-based hand detector."""
    
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        return_landmarks: bool = True
    ):
        """
        Initialize MediaPipe hands detector.
        
        Args:
            static_image_mode: Treat each image as independent
            max_num_hands: Maximum number of hands to detect
            model_complexity: Model complexity (0, 1, or 2)
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence  
            return_landmarks: Whether to return landmark points
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.return_landmarks = return_landmarks
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        logger.info(f"MediaPipe hands initialized: max_hands={max_num_hands}, complexity={model_complexity}")
    
    def detect(self, image: np.ndarray) -> List[HandDetection]:
        """
        Detect hands in image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of hand detections
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.hands.process(rgb_image)
        
        detections = []
        
        if results.multi_hand_landmarks:
            h, w = image.shape[:2]
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get handedness
                handedness = "Unknown"
                confidence = 0.0
                
                if results.multi_handedness and idx < len(results.multi_handedness):
                    hand_info = results.multi_handedness[idx]
                    handedness = hand_info.classification[0].label
                    confidence = hand_info.classification[0].score
                
                # Convert landmarks to pixel coordinates
                landmarks_px = []
                x_coords, y_coords = [], []
                
                for landmark in hand_landmarks.landmark:
                    px = int(landmark.x * w)
                    py = int(landmark.y * h)
                    pz = landmark.z  # Relative depth
                    
                    landmarks_px.append([px, py, pz])
                    x_coords.append(px)
                    y_coords.append(py)
                
                # Calculate bounding box
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Add padding
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                
                # Create landmarks object if requested
                landmarks_obj = None
                if self.return_landmarks:
                    landmarks_obj = HandLandmarks(
                        landmarks=np.array(landmarks_px),
                        bbox=bbox,
                        confidence=confidence,
                        handedness=handedness
                    )
                
                # Create detection
                detection = HandDetection(
                    bbox=bbox,
                    confidence=confidence,
                    landmarks=landmarks_obj,
                    handedness=handedness
                )
                
                detections.append(detection)
        
        return detections
    
    def draw_landmarks(
        self,
        image: np.ndarray,
        detections: List[HandDetection],
        draw_bbox: bool = True,
        draw_landmarks: bool = True,
        draw_connections: bool = True
    ) -> np.ndarray:
        """
        Draw hand landmarks and bounding boxes on image.
        
        Args:
            image: Input image
            detections: Hand detections
            draw_bbox: Draw bounding boxes
            draw_landmarks: Draw landmark points
            draw_connections: Draw landmark connections
            
        Returns:
            Image with annotations
        """
        annotated_image = image.copy()
        
        for detection in detections:
            # Draw bounding box
            if draw_bbox:
                x, y, w, h = detection.bbox
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw confidence and handedness
                label = f"{detection.handedness}: {detection.confidence:.2f}"
                cv2.putText(
                    annotated_image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )
            
            # Draw landmarks
            if detection.landmarks and (draw_landmarks or draw_connections):
                # Convert landmarks for MediaPipe drawing
                landmarks_list = []
                for landmark in detection.landmarks.landmarks:
                    x_norm = landmark[0] / image.shape[1]
                    y_norm = landmark[1] / image.shape[0]
                    z_norm = landmark[2]
                    
                    landmark_proto = self.mp_hands.HandLandmark()
                    landmark_proto.x = x_norm
                    landmark_proto.y = y_norm  
                    landmark_proto.z = z_norm
                    landmarks_list.append(landmark_proto)
                
                # Create landmarks object for drawing
                hand_landmarks = type('HandLandmarks', (), {})()
                hand_landmarks.landmark = landmarks_list
                
                if draw_connections:
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                elif draw_landmarks:
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        None,
                        self.mp_drawing_styles.get_default_hand_landmarks_style()
                    )
        
        return annotated_image
    
    def get_fingertip_positions(self, landmarks: HandLandmarks) -> np.ndarray:
        """
        Get fingertip positions from landmarks.
        
        Args:
            landmarks: Hand landmarks
            
        Returns:
            Array of fingertip positions (5x3: thumb, index, middle, ring, pinky)
        """
        # Fingertip landmark indices
        fingertip_indices = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        
        fingertips = landmarks.landmarks[fingertip_indices]
        return fingertips
    
    def estimate_grip_strength(self, landmarks: HandLandmarks) -> float:
        """
        Estimate grip strength based on finger positions.
        
        Args:
            landmarks: Hand landmarks
            
        Returns:
            Grip strength score (0.0 to 1.0)
        """
        if landmarks.landmarks.shape[0] < 21:
            return 0.0
        
        # Palm center (approximate)
        palm_landmarks = [0, 1, 5, 9, 13, 17]  # Wrist and finger bases
        palm_center = np.mean(landmarks.landmarks[palm_landmarks], axis=0)
        
        # Fingertips
        fingertips = self.get_fingertip_positions(landmarks)
        
        # Calculate distances from palm center to fingertips
        distances = []
        for fingertip in fingertips:
            dist = np.linalg.norm(fingertip[:2] - palm_center[:2])
            distances.append(dist)
        
        # Normalize and invert (closer = stronger grip)
        avg_distance = np.mean(distances)
        max_distance = 100.0  # Approximate max distance for open hand
        
        grip_strength = max(0.0, 1.0 - (avg_distance / max_distance))
        return min(1.0, grip_strength)
    
    def is_hand_closed(self, landmarks: HandLandmarks, threshold: float = 0.6) -> bool:
        """
        Determine if hand appears to be in a closed/gripping position.
        
        Args:
            landmarks: Hand landmarks
            threshold: Grip strength threshold
            
        Returns:
            True if hand appears closed
        """
        grip_strength = self.estimate_grip_strength(landmarks)
        return grip_strength > threshold
    
    def close(self) -> None:
        """Close MediaPipe hands detector."""
        if self.hands:
            self.hands.close()
            logger.info("MediaPipe hands detector closed")


def create_mediapipe_detector(**kwargs) -> MediaPipeHands:
    """
    Factory function to create MediaPipe hands detector.
    
    Args:
        **kwargs: Arguments for MediaPipeHands
        
    Returns:
        MediaPipeHands instance
    """
    return MediaPipeHands(**kwargs)
