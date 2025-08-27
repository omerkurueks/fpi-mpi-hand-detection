"""Motion metrics computation for inspection detection."""

from typing import Dict, List, Optional, Tuple, NamedTuple
import numpy as np
import cv2
from collections import deque
import logging

logger = logging.getLogger(__name__)


class MotionMetrics(NamedTuple):
    """Motion metrics data structure."""
    flow_magnitude: float
    centroid_movement: float
    area_ratio: float
    bbox_change: float
    timestamp: float


class MotionTracker:
    """Track motion metrics for regions of interest."""
    
    def __init__(
        self,
        history_size: int = 30,
        smoothing_window: int = 5
    ):
        """
        Initialize motion tracker.
        
        Args:
            history_size: Number of frames to keep in history
            smoothing_window: Window size for smoothing metrics
        """
        self.history_size = history_size
        self.smoothing_window = smoothing_window
        
        # Motion history per track ID
        self.motion_history: Dict[int, deque] = {}
        self.previous_bboxes: Dict[int, Tuple[int, int, int, int]] = {}
        self.previous_centroids: Dict[int, Tuple[float, float]] = {}
        
        logger.info(f"Motion tracker initialized: history={history_size}, smoothing={smoothing_window}")
    
    def update(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        flow_field: Optional[np.ndarray] = None,
        timestamp: float = 0.0
    ) -> MotionMetrics:
        """
        Update motion metrics for a track.
        
        Args:
            track_id: Track identifier
            bbox: Bounding box (x, y, w, h)
            flow_field: Optical flow field
            timestamp: Current timestamp
            
        Returns:
            Motion metrics for this update
        """
        # Initialize history for new tracks
        if track_id not in self.motion_history:
            self.motion_history[track_id] = deque(maxlen=self.history_size)
            self.previous_bboxes[track_id] = bbox
            self.previous_centroids[track_id] = self._get_centroid(bbox)
        
        # Compute metrics
        flow_magnitude = self._compute_flow_magnitude(bbox, flow_field)
        centroid_movement = self._compute_centroid_movement(track_id, bbox)
        area_ratio = self._compute_area_ratio(track_id, bbox)
        bbox_change = self._compute_bbox_change(track_id, bbox)
        
        # Create metrics object
        metrics = MotionMetrics(
            flow_magnitude=flow_magnitude,
            centroid_movement=centroid_movement,
            area_ratio=area_ratio,
            bbox_change=bbox_change,
            timestamp=timestamp
        )
        
        # Add to history
        self.motion_history[track_id].append(metrics)
        
        # Update previous values
        self.previous_bboxes[track_id] = bbox
        self.previous_centroids[track_id] = self._get_centroid(bbox)
        
        return metrics
    
    def _get_centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Get centroid of bounding box."""
        x, y, w, h = bbox
        return (x + w / 2.0, y + h / 2.0)
    
    def _compute_flow_magnitude(
        self,
        bbox: Tuple[int, int, int, int],
        flow_field: Optional[np.ndarray]
    ) -> float:
        """Compute average flow magnitude within bounding box."""
        if flow_field is None:
            return 0.0
        
        x, y, w, h = bbox
        
        # Ensure bbox is within flow field bounds
        h_max, w_max = flow_field.shape[:2]
        x = max(0, min(x, w_max - 1))
        y = max(0, min(y, h_max - 1))
        w = max(1, min(w, w_max - x))
        h = max(1, min(h, h_max - y))
        
        # Extract ROI flow
        roi_flow = flow_field[y:y+h, x:x+w]
        
        if roi_flow.size == 0:
            return 0.0
        
        # Compute magnitude
        magnitude = np.sqrt(roi_flow[..., 0]**2 + roi_flow[..., 1]**2)
        
        return float(np.mean(magnitude))
    
    def _compute_centroid_movement(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int]
    ) -> float:
        """Compute centroid movement from previous frame."""
        current_centroid = self._get_centroid(bbox)
        
        if track_id not in self.previous_centroids:
            return 0.0
        
        prev_centroid = self.previous_centroids[track_id]
        
        # Euclidean distance
        dx = current_centroid[0] - prev_centroid[0]
        dy = current_centroid[1] - prev_centroid[1]
        
        movement = np.sqrt(dx**2 + dy**2)
        
        return float(movement)
    
    def _compute_area_ratio(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int]
    ) -> float:
        """Compute area change ratio from previous frame."""
        x, y, w, h = bbox
        current_area = w * h
        
        if track_id not in self.previous_bboxes:
            return 1.0
        
        prev_x, prev_y, prev_w, prev_h = self.previous_bboxes[track_id]
        prev_area = prev_w * prev_h
        
        if prev_area == 0:
            return 1.0
        
        ratio = current_area / prev_area
        
        return float(ratio)
    
    def _compute_bbox_change(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int]
    ) -> float:
        """Compute overall bounding box change metric."""
        if track_id not in self.previous_bboxes:
            return 0.0
        
        prev_bbox = self.previous_bboxes[track_id]
        
        # Compute IoU between current and previous bbox
        iou = self._compute_bbox_iou(bbox, prev_bbox)
        
        # Change metric is 1 - IoU
        change = 1.0 - iou
        
        return float(change)
    
    def _compute_bbox_iou(
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
    
    def get_smoothed_metrics(
        self,
        track_id: int,
        metric_name: str
    ) -> Optional[float]:
        """
        Get smoothed metric value.
        
        Args:
            track_id: Track identifier
            metric_name: Name of metric to smooth
            
        Returns:
            Smoothed metric value or None if insufficient history
        """
        if track_id not in self.motion_history:
            return None
        
        history = list(self.motion_history[track_id])
        
        if len(history) < self.smoothing_window:
            return None
        
        # Get recent values
        recent_values = []
        for metrics in history[-self.smoothing_window:]:
            value = getattr(metrics, metric_name, None)
            if value is not None:
                recent_values.append(value)
        
        if not recent_values:
            return None
        
        return float(np.mean(recent_values))
    
    def get_motion_score(
        self,
        track_id: int,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute combined motion score.
        
        Args:
            track_id: Track identifier
            weights: Weights for different metrics
            
        Returns:
            Combined motion score
        """
        if weights is None:
            weights = {
                'flow_magnitude': 0.4,
                'centroid_movement': 0.3,
                'area_ratio': 0.2,
                'bbox_change': 0.1
            }
        
        score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in weights.items():
            value = self.get_smoothed_metrics(track_id, metric_name)
            if value is not None:
                # Normalize metrics to 0-1 range
                if metric_name == 'area_ratio':
                    # Area ratio: closer to 1.0 means less change
                    normalized_value = abs(1.0 - value)
                else:
                    # Other metrics: higher values mean more motion
                    normalized_value = min(1.0, value / 10.0)  # Scale factor
                
                score += weight * normalized_value
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return score / total_weight
    
    def get_history(self, track_id: int) -> List[MotionMetrics]:
        """Get motion history for a track."""
        if track_id not in self.motion_history:
            return []
        
        return list(self.motion_history[track_id])
    
    def remove_track(self, track_id: int) -> None:
        """Remove track from motion tracking."""
        self.motion_history.pop(track_id, None)
        self.previous_bboxes.pop(track_id, None)
        self.previous_centroids.pop(track_id, None)
    
    def get_active_tracks(self) -> List[int]:
        """Get list of active track IDs."""
        return list(self.motion_history.keys())
    
    def clear(self) -> None:
        """Clear all motion tracking data."""
        self.motion_history.clear()
        self.previous_bboxes.clear()
        self.previous_centroids.clear()


def compute_motion_intensity(
    flow_field: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None
) -> float:
    """
    Compute motion intensity in region.
    
    Args:
        flow_field: Optical flow field
        roi: Region of interest (x, y, w, h). If None, use entire field
        
    Returns:
        Motion intensity score
    """
    if roi is not None:
        x, y, w, h = roi
        h_max, w_max = flow_field.shape[:2]
        
        # Clip ROI to valid bounds
        x = max(0, min(x, w_max - 1))
        y = max(0, min(y, h_max - 1))
        w = max(1, min(w, w_max - x))
        h = max(1, min(h, h_max - y))
        
        flow_roi = flow_field[y:y+h, x:x+w]
    else:
        flow_roi = flow_field
    
    if flow_roi.size == 0:
        return 0.0
    
    # Compute magnitude
    magnitude = np.sqrt(flow_roi[..., 0]**2 + flow_roi[..., 1]**2)
    
    # Use 95th percentile to avoid outliers
    intensity = float(np.percentile(magnitude, 95))
    
    return intensity


def create_motion_tracker(**kwargs) -> MotionTracker:
    """
    Factory function to create motion tracker.
    
    Args:
        **kwargs: Arguments for MotionTracker
        
    Returns:
        MotionTracker instance
    """
    return MotionTracker(**kwargs)
