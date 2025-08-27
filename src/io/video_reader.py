"""Video reader for various input sources (RTSP, USB, file)."""

import time
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union
import cv2
import numpy as np
from loguru import logger


class VideoReader:
    """
    Video reader supporting multiple input sources.
    
    Supports:
    - RTSP streams
    - USB cameras  
    - Video files (MP4, AVI, etc.)
    - Image sequences
    """
    
    def __init__(
        self,
        source: Union[str, int, Path],
        target_fps: Optional[float] = None,
        resize: Optional[Tuple[int, int]] = None,
        auto_reconnect: bool = True,
        reconnect_delay: float = 5.0,
        buffer_size: int = 1
    ):
        """
        Initialize video reader.
        
        Args:
            source: Video source (file path, RTSP URL, or camera index)
            target_fps: Target FPS for playback (None for natural speed)
            resize: Resize frames to (width, height) if specified
            auto_reconnect: Automatically reconnect on RTSP failures
            reconnect_delay: Delay between reconnection attempts
            buffer_size: OpenCV buffer size for real-time streams
        """
        self.source = source
        self.target_fps = target_fps
        self.resize = resize
        self.auto_reconnect = auto_reconnect
        self.reconnect_delay = reconnect_delay
        self.buffer_size = buffer_size
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        self.frame_count = 0
        self.last_frame_time = 0.0
        self.source_fps = 30.0  # Default fallback
        
        self._initialize_capture()
    
    def _initialize_capture(self) -> bool:
        """Initialize video capture."""
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False
            
            # Set buffer size for real-time streams
            if isinstance(self.source, (str, int)) and (
                str(self.source).startswith(('rtsp://', 'rtmp://')) or isinstance(self.source, int)
            ):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            # Get source properties
            self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video source opened: {self.source}")
            logger.info(f"Source properties: {width}x{height} @ {self.source_fps:.1f} FPS")
            
            self.is_opened = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing video capture: {e}")
            return False
    
    def _reconnect(self) -> bool:
        """Attempt to reconnect to video source."""
        if self.cap:
            self.cap.release()
        
        logger.info(f"Attempting to reconnect to {self.source}...")
        time.sleep(self.reconnect_delay)
        
        return self._initialize_capture()
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame from video source.
        
        Returns:
            Tuple of (success, frame) where frame is None if read failed
        """
        if not self.cap or not self.is_opened:
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            if self.auto_reconnect and isinstance(self.source, str) and self.source.startswith(('rtsp://', 'rtmp://')):
                logger.warning("Frame read failed, attempting reconnection...")
                if self._reconnect():
                    ret, frame = self.cap.read()
                else:
                    return False, None
            else:
                return False, None
        
        if frame is None:
            return False, None
        
        # Resize if requested
        if self.resize:
            frame = cv2.resize(frame, self.resize)
        
        # FPS limiting
        if self.target_fps:
            current_time = time.time()
            if self.last_frame_time > 0:
                frame_interval = 1.0 / self.target_fps
                elapsed = current_time - self.last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
            self.last_frame_time = time.time()
        
        self.frame_count += 1
        return True, frame
    
    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over video frames."""
        while True:
            ret, frame = self.read()
            if not ret or frame is None:
                break
            yield frame
    
    def get_properties(self) -> dict:
        """Get video source properties."""
        if not self.cap:
            return {}
        
        return {
            'fps': self.source_fps,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'current_frame': self.frame_count,
            'source': str(self.source)
        }
    
    def seek(self, frame_number: int) -> bool:
        """Seek to specific frame (for video files only)."""
        if not self.cap:
            return False
        
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    def release(self) -> None:
        """Release video capture resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_opened = False
        logger.info(f"Video reader released for source: {self.source}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


class VideoWriter:
    """Video writer for saving processed frames."""
    
    def __init__(
        self,
        output_path: Union[str, Path],
        fps: float = 30.0,
        frame_size: Optional[Tuple[int, int]] = None,
        fourcc: str = 'mp4v'
    ):
        """
        Initialize video writer.
        
        Args:
            output_path: Output video file path
            fps: Output video FPS
            frame_size: Frame size (width, height). Auto-detected from first frame if None
            fourcc: Video codec fourcc code
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.frame_size = frame_size
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        
        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_count = 0
        
        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def write(self, frame: np.ndarray) -> bool:
        """
        Write frame to video file.
        
        Args:
            frame: Frame to write
            
        Returns:
            True if successful, False otherwise
        """
        if self.writer is None:
            # Initialize writer with first frame
            if self.frame_size is None:
                h, w = frame.shape[:2]
                self.frame_size = (w, h)
            
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                self.fourcc,
                self.fps,
                self.frame_size
            )
            
            if not self.writer.isOpened():
                logger.error(f"Failed to open video writer: {self.output_path}")
                return False
            
            logger.info(f"Video writer initialized: {self.output_path} @ {self.fps} FPS")
        
        # Resize frame if needed
        if frame.shape[:2][::-1] != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        
        self.writer.write(frame)
        self.frame_count += 1
        return True
    
    def release(self) -> None:
        """Release video writer resources."""
        if self.writer:
            self.writer.release()
            self.writer = None
        
        logger.info(f"Video writer released: {self.output_path} ({self.frame_count} frames)")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


def create_video_reader(source: Union[str, int, Path], **kwargs) -> VideoReader:
    """
    Factory function to create video reader.
    
    Args:
        source: Video source
        **kwargs: Additional arguments for VideoReader
        
    Returns:
        VideoReader instance
    """
    return VideoReader(source, **kwargs)


def is_valid_video_source(source: Union[str, int, Path]) -> bool:
    """
    Check if video source is valid.
    
    Args:
        source: Video source to check
        
    Returns:
        True if valid, False otherwise
    """
    try:
        cap = cv2.VideoCapture(source)
        is_valid = cap.isOpened()
        cap.release()
        return is_valid
    except Exception:
        return False
