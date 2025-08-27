"""Visualization and overlay rendering."""

from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class OverlayRenderer:
    """Renderer for visualization overlays."""
    
    def __init__(
        self,
        show_fps: bool = True,
        show_confidence: bool = True,
        show_landmarks: bool = False,
        hand_color: Tuple[int, int, int] = (0, 255, 0),
        object_color: Tuple[int, int, int] = (255, 0, 0),
        track_color: Tuple[int, int, int] = (0, 255, 255),
        inspecting_color: Tuple[int, int, int] = (0, 0, 255),
        font_scale: float = 0.6,
        line_thickness: int = 2
    ):
        """
        Initialize overlay renderer.
        
        Args:
            show_fps: Show FPS counter
            show_confidence: Show detection confidence
            show_landmarks: Show hand landmarks
            hand_color: Color for hand bounding boxes (BGR)
            object_color: Color for object bounding boxes (BGR)
            track_color: Color for tracking information (BGR)
            inspecting_color: Color for active inspection state (BGR)
            font_scale: Font scale for text
            line_thickness: Line thickness for drawings
        """
        self.show_fps = show_fps
        self.show_confidence = show_confidence
        self.show_landmarks = show_landmarks
        self.hand_color = hand_color
        self.object_color = object_color
        self.track_color = track_color
        self.inspecting_color = inspecting_color
        self.font_scale = font_scale
        self.line_thickness = line_thickness
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        logger.info("Overlay renderer initialized")
    
    def draw_hand(self, image: np.ndarray, hand_data: Dict[str, Any]) -> None:
        """
        Draw hand detection on image.
        
        Args:
            image: Image to draw on
            hand_data: Hand detection data
        """
        bbox = hand_data['bbox']
        confidence = hand_data.get('confidence', 0.0)
        source = hand_data.get('source', 'unknown')
        handedness = hand_data.get('handedness', '')
        
        x, y, w, h = bbox
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), self.hand_color, self.line_thickness)
        
        # Draw label
        label_parts = ['Hand']
        if handedness:
            label_parts.append(handedness)
        if self.show_confidence:
            label_parts.append(f"{confidence:.2f}")
        if source != 'unknown':
            label_parts.append(f"({source})")
        
        label = ' '.join(label_parts)
        
        # Calculate text size and position
        (text_width, text_height), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, 1
        )
        
        # Draw background for text
        text_y = y - 10 if y - 10 > text_height else y + h + text_height + 10
        cv2.rectangle(
            image,
            (x, text_y - text_height - baseline),
            (x + text_width, text_y + baseline),
            self.hand_color,
            -1
        )
        
        # Draw text
        cv2.putText(
            image, label, (x, text_y - baseline),
            self.font, self.font_scale, (255, 255, 255), 1
        )
        
        # Draw landmarks if available and enabled
        if self.show_landmarks and 'landmarks' in hand_data:
            self._draw_hand_landmarks(image, hand_data['landmarks'])
    
    def draw_object(self, image: np.ndarray, object_data: Dict[str, Any]) -> None:
        """
        Draw object detection on image.
        
        Args:
            image: Image to draw on
            object_data: Object detection data
        """
        bbox = object_data['bbox']
        confidence = object_data.get('confidence', 0.0)
        class_name = object_data.get('class_name', 'object')
        source = object_data.get('source', 'unknown')
        
        x, y, w, h = bbox
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), self.object_color, self.line_thickness)
        
        # Draw label
        label_parts = [class_name.capitalize()]
        if self.show_confidence:
            label_parts.append(f"{confidence:.2f}")
        if source != 'unknown':
            label_parts.append(f"({source})")
        
        label = ' '.join(label_parts)
        
        # Calculate text size and position
        (text_width, text_height), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, 1
        )
        
        # Draw background for text
        text_y = y - 10 if y - 10 > text_height else y + h + text_height + 10
        cv2.rectangle(
            image,
            (x, text_y - text_height - baseline),
            (x + text_width, text_y + baseline),
            self.object_color,
            -1
        )
        
        # Draw text
        cv2.putText(
            image, label, (x, text_y - baseline),
            self.font, self.font_scale, (255, 255, 255), 1
        )
    
    def draw_track(self, image: np.ndarray, track_data: Dict[str, Any]) -> None:
        """
        Draw track information on image.
        
        Args:
            image: Image to draw on
            track_data: Track data
        """
        track_id = track_data['track_id']
        roi = track_data.get('roi', (0, 0, 0, 0))
        motion_score = track_data.get('motion_score', 0.0)
        state = track_data.get('state', 'idle')
        is_inspecting = track_data.get('is_inspecting', False)
        
        x, y, w, h = roi
        
        # Choose color based on state
        color = self.inspecting_color if is_inspecting else self.track_color
        
        # Draw ROI
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
        
        # Draw track info
        label = f"Track {track_id}: {state.upper()}"
        if motion_score > 0:
            label += f" ({motion_score:.2f})"
        
        # Position text inside ROI
        text_x = x + 5
        text_y = y + 20
        
        # Draw text background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, self.font, self.font_scale * 0.8, 1
        )
        
        cv2.rectangle(
            image,
            (text_x - 2, text_y - text_height - 2),
            (text_x + text_width + 2, text_y + baseline + 2),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            image, label, (text_x, text_y),
            self.font, self.font_scale * 0.8, (255, 255, 255), 1
        )
        
        # Draw motion indicator
        if is_inspecting:
            self._draw_motion_indicator(image, (x + w//2, y + h//2), motion_score)
    
    def draw_statistics(self, image: np.ndarray, stats: Dict[str, Any]) -> None:
        """
        Draw statistics on image.
        
        Args:
            image: Image to draw on
            stats: Statistics dictionary
        """
        h, w = image.shape[:2]
        
        # Prepare stats text
        text_lines = []
        
        if self.show_fps:
            fps = stats.get('fps', 0.0)
            text_lines.append(f"FPS: {fps:.1f}")
        
        frame_count = stats.get('frame_count', 0)
        text_lines.append(f"Frame: {frame_count}")
        
        active_tracks = stats.get('active_tracks', 0)
        text_lines.append(f"Tracks: {active_tracks}")
        
        inspecting_tracks = stats.get('inspecting_tracks', 0)
        if inspecting_tracks > 0:
            text_lines.append(f"Inspecting: {inspecting_tracks}")
        
        total_events = stats.get('completed_events', 0)
        if total_events > 0:
            text_lines.append(f"Events: {total_events}")
        
        # Draw background panel
        panel_width = 200
        panel_height = len(text_lines) * 25 + 20
        panel_x = w - panel_width - 10
        panel_y = 10
        
        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw border
        cv2.rectangle(image, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (255, 255, 255), 1)
        
        # Draw text lines
        for i, line in enumerate(text_lines):
            text_x = panel_x + 10
            text_y = panel_y + 20 + i * 25
            
            cv2.putText(
                image, line, (text_x, text_y),
                self.font, self.font_scale * 0.7, (255, 255, 255), 1
            )
    
    def draw_flow_visualization(
        self,
        image: np.ndarray,
        flow_field: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        step: int = 16,
        scale: float = 3.0
    ) -> None:
        """
        Draw optical flow visualization.
        
        Args:
            image: Image to draw on
            flow_field: Optical flow field
            roi: Region of interest to limit visualization
            step: Step size for flow vectors
            scale: Scale factor for arrows
        """
        if flow_field is None:
            return
        
        h, w = flow_field.shape[:2]
        
        # Apply ROI if specified
        if roi:
            x, y, rw, rh = roi
            x = max(0, min(x, w))
            y = max(0, min(y, h))
            rw = max(1, min(rw, w - x))
            rh = max(1, min(rh, h - y))
            
            y_coords, x_coords = np.mgrid[y:y+rh:step, x:x+rw:step]
        else:
            y_coords, x_coords = np.mgrid[0:h:step, 0:w:step]
        
        y_coords = y_coords.flatten()
        x_coords = x_coords.flatten()
        
        # Get flow vectors at sample points
        fx = flow_field[y_coords, x_coords, 0]
        fy = flow_field[y_coords, x_coords, 1]
        
        # Draw flow arrows
        for i in range(len(x_coords)):
            if np.sqrt(fx[i]**2 + fy[i]**2) > 1.0:  # Minimum threshold
                end_x = int(x_coords[i] + fx[i] * scale)
                end_y = int(y_coords[i] + fy[i] * scale)
                
                cv2.arrowedLine(
                    image,
                    (x_coords[i], y_coords[i]),
                    (end_x, end_y),
                    (0, 255, 255),
                    1,
                    tipLength=0.3
                )
    
    def _draw_hand_landmarks(self, image: np.ndarray, landmarks_data: Any) -> None:
        """Draw hand landmarks if available."""
        # This would integrate with MediaPipe landmarks drawing
        # Implementation depends on the landmarks data structure
        pass
    
    def _draw_motion_indicator(
        self,
        image: np.ndarray,
        center: Tuple[int, int],
        motion_score: float
    ) -> None:
        """
        Draw motion intensity indicator.
        
        Args:
            image: Image to draw on
            center: Center point for indicator
            motion_score: Motion intensity score
        """
        x, y = center
        
        # Scale radius based on motion score
        max_radius = 30
        radius = int(min(max_radius, motion_score * 10))
        
        if radius > 5:
            # Pulsing circle effect
            alpha = 0.6 + 0.4 * np.sin(time.time() * 10)  # Pulsing effect
            
            # Create overlay for transparency
            overlay = image.copy()
            cv2.circle(overlay, (x, y), radius, self.inspecting_color, -1)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            
            # Draw border
            cv2.circle(image, (x, y), radius, self.inspecting_color, 2)


def create_overlay_renderer(**kwargs) -> OverlayRenderer:
    """
    Factory function to create overlay renderer.
    
    Args:
        **kwargs: Arguments for OverlayRenderer
        
    Returns:
        OverlayRenderer instance
    """
    return OverlayRenderer(**kwargs)


def draw_detection_results(
    image: np.ndarray,
    hands: List[Dict[str, Any]],
    objects: List[Dict[str, Any]],
    tracks: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Quick function to draw detection results on image.
    
    Args:
        image: Input image
        hands: Hand detections
        objects: Object detections  
        tracks: Track information
        
    Returns:
        Image with detections drawn
    """
    renderer = create_overlay_renderer()
    result = image.copy()
    
    # Draw all detections
    for hand in hands:
        renderer.draw_hand(result, hand)
    
    for obj in objects:
        renderer.draw_object(result, obj)
    
    for track in tracks:
        renderer.draw_track(result, track)
    
    return result
