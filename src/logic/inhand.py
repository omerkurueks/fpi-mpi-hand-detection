"""In-hand object detection logic."""

from typing import Dict, List, Optional, Tuple, NamedTuple
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


class InHandMatch(NamedTuple):
    """In-hand matching result."""
    hand_bbox: Tuple[int, int, int, int]
    object_bbox: Optional[Tuple[int, int, int, int]]
    iou: float
    distance: float
    confidence: float
    is_in_hand: bool


class InHandDetector:
    """Detector for objects held in hands."""
    
    def __init__(
        self,
        iou_threshold: float = 0.20,
        distance_threshold: float = 50.0,
        center_inside: bool = True,
        confidence_weight: float = 0.3
    ):
        """
        Initialize in-hand detector.
        
        Args:
            iou_threshold: Minimum IoU for in-hand detection
            distance_threshold: Maximum distance threshold in pixels
            center_inside: Consider object center inside hand bbox
            confidence_weight: Weight for detection confidence
        """
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.center_inside = center_inside
        self.confidence_weight = confidence_weight
        
        logger.info(f"In-hand detector initialized: IoU={iou_threshold}, distance={distance_threshold}")
    
    def detect_in_hand(
        self,
        hand_detections: List[Tuple[Tuple[int, int, int, int], float]],
        object_detections: Optional[List[Tuple[Tuple[int, int, int, int], float]]] = None
    ) -> List[InHandMatch]:
        """
        Detect objects held in hands.
        
        Args:
            hand_detections: List of (bbox, confidence) for hands
            object_detections: List of (bbox, confidence) for objects
            
        Returns:
            List of in-hand matches
        """
        matches = []
        
        for hand_bbox, hand_conf in hand_detections:
            best_match = None
            best_score = 0.0
            
            if object_detections:
                # Find best matching object for this hand
                for obj_bbox, obj_conf in object_detections:
                    match = self._evaluate_match(hand_bbox, obj_bbox, hand_conf, obj_conf)
                    
                    # Compute overall score
                    score = self._compute_match_score(match)
                    
                    if score > best_score and match.is_in_hand:
                        best_match = match
                        best_score = score
            
            # If no object match found, create hand-only match
            if best_match is None:
                best_match = InHandMatch(
                    hand_bbox=hand_bbox,
                    object_bbox=None,
                    iou=0.0,
                    distance=float('inf'),
                    confidence=hand_conf,
                    is_in_hand=self._detect_hand_holding(hand_bbox)
                )
            
            matches.append(best_match)
        
        return matches
    
    def _evaluate_match(
        self,
        hand_bbox: Tuple[int, int, int, int],
        object_bbox: Tuple[int, int, int, int],
        hand_conf: float,
        obj_conf: float
    ) -> InHandMatch:
        """Evaluate hand-object match."""
        # Compute IoU
        iou = self._compute_iou(hand_bbox, object_bbox)
        
        # Compute distance between centers
        distance = self._compute_distance(hand_bbox, object_bbox)
        
        # Check if object center is inside hand bbox
        center_inside = False
        if self.center_inside:
            center_inside = self._is_center_inside(object_bbox, hand_bbox)
        
        # Determine if object is in hand
        is_in_hand = (
            iou >= self.iou_threshold or
            distance <= self.distance_threshold or
            center_inside
        )
        
        # Combined confidence
        combined_conf = (hand_conf + obj_conf) / 2.0
        
        return InHandMatch(
            hand_bbox=hand_bbox,
            object_bbox=object_bbox,
            iou=iou,
            distance=distance,
            confidence=combined_conf,
            is_in_hand=is_in_hand
        )
    
    def _compute_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Compute IoU between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert to (x1, y1, x2, y2) format
        box1 = (x1, y1, x1 + w1, y1 + h1)
        box2 = (x2, y2, x2 + w2, y2 + h2)
        
        # Compute intersection
        ix1 = max(box1[0], box2[0])
        iy1 = max(box1[1], box2[1])
        ix2 = min(box1[2], box2[2])
        iy2 = min(box1[3], box2[3])
        
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        
        intersection = (ix2 - ix1) * (iy2 - iy1)
        
        # Compute union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _compute_distance(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """Compute distance between bbox centers."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        center1 = (x1 + w1 / 2, y1 + h1 / 2)
        center2 = (x2 + w2 / 2, y2 + h2 / 2)
        
        dx = center1[0] - center2[0]
        dy = center1[1] - center2[1]
        
        return np.sqrt(dx**2 + dy**2)
    
    def _is_center_inside(
        self,
        object_bbox: Tuple[int, int, int, int],
        hand_bbox: Tuple[int, int, int, int]
    ) -> bool:
        """Check if object center is inside hand bbox."""
        obj_x, obj_y, obj_w, obj_h = object_bbox
        hand_x, hand_y, hand_w, hand_h = hand_bbox
        
        # Object center
        obj_center_x = obj_x + obj_w / 2
        obj_center_y = obj_y + obj_h / 2
        
        # Check if center is inside hand bbox
        return (
            hand_x <= obj_center_x <= hand_x + hand_w and
            hand_y <= obj_center_y <= hand_y + hand_h
        )
    
    def _compute_match_score(self, match: InHandMatch) -> float:
        """Compute overall match score."""
        if not match.is_in_hand:
            return 0.0
        
        # Combine IoU, distance, and confidence
        iou_score = match.iou
        
        # Distance score (inverse normalized)
        distance_score = max(0.0, 1.0 - (match.distance / self.distance_threshold))
        
        # Confidence score
        conf_score = match.confidence
        
        # Weighted combination
        score = (
            0.4 * iou_score +
            0.3 * distance_score +
            self.confidence_weight * conf_score
        )
        
        return score
    
    def _detect_hand_holding(self, hand_bbox: Tuple[int, int, int, int]) -> bool:
        """
        Detect if hand appears to be holding something (even without object detection).
        
        This is a simple heuristic that could be enhanced with:
        - Hand pose analysis
        - Finger position analysis
        - Motion patterns
        
        Args:
            hand_bbox: Hand bounding box
            
        Returns:
            True if hand appears to be holding something
        """
        # Simple heuristic: assume hands with sufficient size might be holding objects
        x, y, w, h = hand_bbox
        area = w * h
        
        # Minimum area threshold for potential object holding
        min_area = 2000  # pixels
        
        return area >= min_area
    
    def create_roi_from_match(
        self,
        match: InHandMatch,
        expansion_factor: float = 1.2
    ) -> Tuple[int, int, int, int]:
        """
        Create expanded ROI from hand-object match.
        
        Args:
            match: In-hand match
            expansion_factor: Factor to expand ROI
            
        Returns:
            Expanded ROI (x, y, w, h)
        """
        if match.object_bbox is not None:
            # Use combined hand-object bbox
            hand_x, hand_y, hand_w, hand_h = match.hand_bbox
            obj_x, obj_y, obj_w, obj_h = match.object_bbox
            
            # Combined bbox
            min_x = min(hand_x, obj_x)
            min_y = min(hand_y, obj_y)
            max_x = max(hand_x + hand_w, obj_x + obj_w)
            max_y = max(hand_y + hand_h, obj_y + obj_h)
            
            combined_w = max_x - min_x
            combined_h = max_y - min_y
            combined_bbox = (min_x, min_y, combined_w, combined_h)
        else:
            # Use hand bbox only
            combined_bbox = match.hand_bbox
        
        # Expand ROI
        x, y, w, h = combined_bbox
        
        center_x = x + w // 2
        center_y = y + h // 2
        
        new_w = int(w * expansion_factor)
        new_h = int(h * expansion_factor)
        
        new_x = center_x - new_w // 2
        new_y = center_y - new_h // 2
        
        return (new_x, new_y, new_w, new_h)
    
    def update_thresholds(
        self,
        iou_threshold: Optional[float] = None,
        distance_threshold: Optional[float] = None
    ) -> None:
        """Update detection thresholds."""
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
            
        if distance_threshold is not None:
            self.distance_threshold = distance_threshold
        
        logger.info(f"Thresholds updated: IoU={self.iou_threshold}, distance={self.distance_threshold}")


def create_inhand_detector(**kwargs) -> InHandDetector:
    """
    Factory function to create in-hand detector.
    
    Args:
        **kwargs: Arguments for InHandDetector
        
    Returns:
        InHandDetector instance
    """
    return InHandDetector(**kwargs)


def compute_hand_object_overlap(
    hand_bbox: Tuple[int, int, int, int],
    object_bbox: Tuple[int, int, int, int]
) -> Dict[str, float]:
    """
    Compute detailed overlap metrics between hand and object.
    
    Args:
        hand_bbox: Hand bounding box
        object_bbox: Object bounding box
        
    Returns:
        Dictionary with overlap metrics
    """
    # Basic IoU
    iou = InHandDetector(0.0, 0.0)._compute_iou(hand_bbox, object_bbox)
    
    # Distance between centers
    distance = InHandDetector(0.0, 0.0)._compute_distance(hand_bbox, object_bbox)
    
    # Area overlap ratio
    x1, y1, w1, h1 = hand_bbox
    x2, y2, w2, h2 = object_bbox
    
    hand_area = w1 * h1
    object_area = w2 * h2
    
    # Intersection area
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)
    
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    
    # Overlap ratios
    hand_overlap_ratio = intersection / hand_area if hand_area > 0 else 0.0
    object_overlap_ratio = intersection / object_area if object_area > 0 else 0.0
    
    return {
        'iou': iou,
        'distance': distance,
        'intersection_area': intersection,
        'hand_overlap_ratio': hand_overlap_ratio,
        'object_overlap_ratio': object_overlap_ratio,
        'hand_area': hand_area,
        'object_area': object_area
    }
