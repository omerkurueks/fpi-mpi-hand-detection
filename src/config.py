"""Configuration management for the hand inspection detection system."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, Field, validator


class InHandConfig(BaseModel):
    """Configuration for in-hand detection logic."""
    iou_min: float = Field(0.20, ge=0.0, le=1.0, description="Minimum IoU threshold for hand-object overlap")
    center_inside: bool = Field(True, description="Consider object center inside hand bbox")
    distance_threshold: float = Field(50.0, ge=0.0, description="Maximum distance threshold in pixels")


class MotionConfig(BaseModel):
    """Configuration for motion analysis."""
    flow_method: str = Field("DIS", description="Optical flow method: DIS or Farneback")
    start_flow_mag: float = Field(1.5, ge=0.0, description="Flow magnitude threshold to start inspection")
    stop_flow_mag: float = Field(0.8, ge=0.0, description="Flow magnitude threshold to stop inspection")
    start_centroid_px: float = Field(6.0, ge=0.0, description="Centroid movement threshold to start inspection")
    stop_centroid_px: float = Field(3.0, ge=0.0, description="Centroid movement threshold to stop inspection")
    min_frames_start: int = Field(6, ge=1, description="Minimum consecutive frames to start inspection")
    min_frames_stop: int = Field(8, ge=1, description="Minimum consecutive frames to stop inspection")
    grace_frames: int = Field(10, ge=0, description="Grace period frames for lost detections")
    area_ratio_threshold: float = Field(0.15, ge=0.0, description="Area change ratio threshold")

    @validator('flow_method')
    def validate_flow_method(cls, v):
        if v not in ['DIS', 'Farneback']:
            raise ValueError('flow_method must be either "DIS" or "Farneback"')
        return v


class TrackingConfig(BaseModel):
    """Configuration for object tracking."""
    method: str = Field("bytetrack", description="Tracking method: bytetrack, deepsort, or csrt")
    max_age: int = Field(30, ge=1, description="Maximum age for tracks")
    track_thresh: float = Field(0.5, ge=0.0, le=1.0, description="Tracking threshold")
    track_buffer: int = Field(30, ge=1, description="Track buffer size")
    match_thresh: float = Field(0.8, ge=0.0, le=1.0, description="Matching threshold")

    @validator('method')
    def validate_method(cls, v):
        if v not in ['bytetrack', 'deepsort', 'csrt']:
            raise ValueError('method must be one of: bytetrack, deepsort, csrt')
        return v


class DetectorConfig(BaseModel):
    """Configuration for object detection."""
    enable_yolo_object: bool = Field(False, description="Enable YOLO object detection")
    enable_yolo_hand: bool = Field(False, description="Enable YOLO hand detection")
    mediapipe_hands: bool = Field(True, description="Enable MediaPipe hands detection")
    conf_thres: float = Field(0.25, ge=0.0, le=1.0, description="Confidence threshold")
    iou_nms: float = Field(0.45, ge=0.0, le=1.0, description="IoU threshold for NMS")
    img_size: int = Field(640, ge=64, description="Input image size")
    max_num_hands: int = Field(2, ge=1, description="Maximum number of hands to detect")


class RuntimeConfig(BaseModel):
    """Configuration for runtime behavior."""
    device: str = Field("auto", description="Device to use: cpu, cuda, or auto")
    draw_overlay: bool = Field(True, description="Draw visualization overlay")
    show_fps: bool = Field(True, description="Show FPS counter")
    show_confidence: bool = Field(True, description="Show detection confidence")

    @validator('device')
    def validate_device(cls, v):
        if v not in ['cpu', 'cuda', 'auto']:
            raise ValueError('device must be one of: cpu, cuda, auto')
        return v


class LoggingConfig(BaseModel):
    """Configuration for logging and output."""
    out_dir: str = Field("runs/events", description="Output directory for events")
    write_jsonl: bool = Field(True, description="Write events to JSONL format")
    write_csv: bool = Field(True, description="Write events to CSV format")
    log_level: str = Field("INFO", description="Logging level")
    max_log_files: int = Field(10, ge=1, description="Maximum number of log files to keep")

    @validator('log_level')
    def validate_log_level(cls, v):
        if v not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            raise ValueError('log_level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL')
        return v


class Config(BaseModel):
    """Main configuration class for the hand inspection detection system."""
    fps_target: int = Field(25, ge=1, description="Target FPS for processing")
    inhand: InHandConfig = Field(default_factory=InHandConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, allow_unicode=True)

    def dict(self, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values by default."""
        return super().dict(exclude_none=True, **kwargs)


class ModelConfig(BaseModel):
    """Configuration for model weights and parameters."""
    
    class YOLOConfig(BaseModel):
        weights_hand: Optional[str] = Field(None, description="Path to YOLO hand detection weights")
        weights_object: Optional[str] = Field(None, description="Path to YOLO object detection weights")
        names: List[str] = Field(["hand", "object"], description="Class names")
        imgsz: int = Field(640, ge=64, description="Input image size")
        conf: float = Field(0.25, ge=0.0, le=1.0, description="Confidence threshold")
        iou: float = Field(0.45, ge=0.0, le=1.0, description="IoU threshold")

    class MediaPipeConfig(BaseModel):
        static_image_mode: bool = Field(False, description="Static image mode")
        max_num_hands: int = Field(2, ge=1, description="Maximum number of hands")
        model_complexity: int = Field(1, ge=0, le=2, description="Model complexity")
        min_detection_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Min detection confidence")
        min_tracking_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Min tracking confidence")

    class TrackerConfig(BaseModel):
        class ByteTrackConfig(BaseModel):
            track_thresh: float = Field(0.5, ge=0.0, le=1.0)
            track_buffer: int = Field(30, ge=1)
            match_thresh: float = Field(0.8, ge=0.0, le=1.0)
            mot20: bool = Field(False)

        class DeepSORTConfig(BaseModel):
            model_path: Optional[str] = Field(None, description="Path to DeepSORT model")
            max_dist: float = Field(0.2, ge=0.0)
            min_confidence: float = Field(0.3, ge=0.0, le=1.0)
            nms_max_overlap: float = Field(1.0, ge=0.0)
            max_iou_distance: float = Field(0.7, ge=0.0, le=1.0)
            max_age: int = Field(70, ge=1)
            n_init: int = Field(3, ge=1)

        class CSRTConfig(BaseModel):
            use_hog: bool = Field(True)
            use_color_names: bool = Field(True)
            use_gray: bool = Field(True)
            use_rgb: bool = Field(False)
            window_function: str = Field("hann")

        bytetrack: ByteTrackConfig = Field(default_factory=ByteTrackConfig)
        deepsort: DeepSORTConfig = Field(default_factory=DeepSORTConfig)
        csrt: CSRTConfig = Field(default_factory=CSRTConfig)

    yolo: YOLOConfig = Field(default_factory=YOLOConfig)
    mediapipe: MediaPipeConfig = Field(default_factory=MediaPipeConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ModelConfig":
        """Load model configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model config file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)


def load_config(logic_path: Union[str, Path], model_path: Optional[Union[str, Path]] = None) -> tuple[Config, ModelConfig]:
    """Load both logic and model configurations."""
    config = Config.from_yaml(logic_path)
    
    if model_path is None:
        model_path = Path(logic_path).parent / "model.yaml"
    
    model_config = ModelConfig.from_yaml(model_path) if Path(model_path).exists() else ModelConfig()
    
    return config, model_config
