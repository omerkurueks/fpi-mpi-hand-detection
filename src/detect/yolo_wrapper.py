"""YOLO-based object and hand detection wrapper."""

from pathlib import Path
from typing import List, Optional, Tuple, Union, NamedTuple
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Optional imports - graceful degradation if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    logger.warning("Ultralytics YOLO not available. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False


class YOLODetection(NamedTuple):
    """YOLO detection result."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    class_id: int
    class_name: str


class YOLOWrapper:
    """Wrapper for YOLO-based detection."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize YOLO wrapper.
        
        Args:
            model_path: Path to YOLO model weights
            device: Device to use ('cpu', 'cuda', or 'auto')
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            img_size: Input image size
            class_names: List of class names
        """
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO not available. Install ultralytics: pip install ultralytics")
        
        self.model_path = Path(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.class_names = class_names or []
        
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load YOLO model."""
        try:
            if not self.model_path.exists():
                logger.warning(f"Model weights not found: {self.model_path}")
                logger.info("Using YOLOv8n as fallback")
                self.model = YOLO('yolov8n.pt')
            else:
                self.model = YOLO(str(self.model_path))
            
            # Set device
            if self.device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            self.model.to(device)
            
            # Get class names from model if not provided
            if not self.class_names and hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
            
            logger.info(f"YOLO model loaded: {self.model_path} on {device}")
            logger.info(f"Classes: {self.class_names}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
    
    def detect(self, image: np.ndarray) -> List[YOLODetection]:
        """
        Detect objects in image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detections
        """
        if self.model is None:
            return []
        
        try:
            # Run inference
            results = self.model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                verbose=False
            )
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]  # First image
                
                if result.boxes is not None:
                    boxes = result.boxes.cpu().numpy()
                    
                    for box in boxes:
                        # Extract box data
                        x1, y1, x2, y2 = box.xyxy[0].astype(int)
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Convert to (x, y, w, h) format
                        bbox = (x1, y1, x2 - x1, y2 - y1)
                        
                        # Get class name
                        class_name = "unknown"
                        if class_id < len(self.class_names):
                            class_name = self.class_names[class_id]
                        
                        detection = YOLODetection(
                            bbox=bbox,
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name
                        )
                        
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []
    
    def detect_hands(self, image: np.ndarray) -> List[YOLODetection]:
        """
        Detect hands specifically.
        
        Args:
            image: Input image
            
        Returns:
            List of hand detections
        """
        all_detections = self.detect(image)
        
        # Filter for hand class
        hand_detections = []
        for det in all_detections:
            if det.class_name.lower() in ['hand', 'hands']:
                hand_detections.append(det)
        
        return hand_detections
    
    def detect_objects(self, image: np.ndarray) -> List[YOLODetection]:
        """
        Detect generic objects.
        
        Args:
            image: Input image
            
        Returns:
            List of object detections
        """
        all_detections = self.detect(image)
        
        # Filter for object class or exclude hands
        object_detections = []
        for det in all_detections:
            if det.class_name.lower() in ['object', 'objects'] or \
               det.class_name.lower() not in ['hand', 'hands', 'person']:
                object_detections.append(det)
        
        return object_detections
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[YOLODetection],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detections on image.
        
        Args:
            image: Input image
            detections: List of detections
            color: Bounding box color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with detections drawn
        """
        annotated_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Draw background for text
            cv2.rectangle(
                annotated_image,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                annotated_image,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return annotated_image


class HandYOLO(YOLOWrapper):
    """Specialized YOLO wrapper for hand detection."""
    
    def __init__(self, model_path: Union[str, Path], **kwargs):
        """Initialize hand-specific YOLO detector."""
        super().__init__(model_path, class_names=["hand"], **kwargs)
    
    def detect(self, image: np.ndarray) -> List[YOLODetection]:
        """Detect hands in image."""
        return self.detect_hands(image)


class ObjectYOLO(YOLOWrapper):
    """Specialized YOLO wrapper for generic object detection."""
    
    def __init__(self, model_path: Union[str, Path], **kwargs):
        """Initialize object-specific YOLO detector."""
        super().__init__(model_path, class_names=["object"], **kwargs)
    
    def detect(self, image: np.ndarray) -> List[YOLODetection]:
        """Detect objects in image."""
        return self.detect_objects(image)


def create_yolo_detector(
    model_path: Union[str, Path],
    detector_type: str = "generic",
    **kwargs
) -> YOLOWrapper:
    """
    Factory function to create YOLO detector.
    
    Args:
        model_path: Path to model weights
        detector_type: Type of detector ('generic', 'hand', 'object')
        **kwargs: Additional arguments
        
    Returns:
        YOLO detector instance
    """
    if not YOLO_AVAILABLE:
        logger.error("YOLO not available")
        return None
    
    if detector_type == "hand":
        return HandYOLO(model_path, **kwargs)
    elif detector_type == "object":
        return ObjectYOLO(model_path, **kwargs)
    else:
        return YOLOWrapper(model_path, **kwargs)


def is_yolo_available() -> bool:
    """Check if YOLO is available."""
    return YOLO_AVAILABLE
