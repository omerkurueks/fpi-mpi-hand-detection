"""Main processing pipeline for hand inspection detection."""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import cv2
import numpy as np
import logging

from .config import Config, ModelConfig
from .io.video_reader import VideoReader, create_video_reader
from .io.sink import EventLogger, InspectionEvent
from .detect.hands_mediapipe import MediaPipeHands, create_mediapipe_detector
from .detect.yolo_wrapper import create_yolo_detector, is_yolo_available
from .motion.optical_flow import OpticalFlowComputer, create_flow_computer
from .motion.metrics import MotionTracker, create_motion_tracker
from .logic.inhand import InHandDetector, create_inhand_detector
from .logic.fsm import InspectionFSM, create_inspection_fsm, InspectionState
from .viz.overlay import OverlayRenderer, create_overlay_renderer

logger = logging.getLogger(__name__)


class Pipeline:
    """Main processing pipeline for hand inspection detection."""
    
    def __init__(
        self,
        config: Config,
        model_config: Optional[ModelConfig] = None
    ):
        """
        Initialize processing pipeline.
        
        Args:
            config: Main configuration
            model_config: Model configuration
        """
        self.config = config
        self.model_config = model_config or ModelConfig()
        
        # Components
        self.video_reader: Optional[VideoReader] = None
        self.hand_detector: Optional[MediaPipeHands] = None
        self.object_detector = None
        self.hand_yolo_detector = None
        self.flow_computer: Optional[OpticalFlowComputer] = None
        self.motion_tracker: Optional[MotionTracker] = None
        self.inhand_detector: Optional[InHandDetector] = None
        self.fsm: Optional[InspectionFSM] = None
        self.event_logger: Optional[EventLogger] = None
        self.overlay_renderer: Optional[OverlayRenderer] = None
        
        # State
        self.is_running = False
        self.frame_count = 0
        self.start_time = 0.0
        self.last_fps_time = 0.0
        self.fps = 0.0
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Pipeline initialized successfully")
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        # Hand detector (MediaPipe)
        if self.config.detector.mediapipe_hands:
            self.hand_detector = create_mediapipe_detector(
                static_image_mode=self.model_config.mediapipe.static_image_mode,
                max_num_hands=self.model_config.mediapipe.max_num_hands,
                model_complexity=self.model_config.mediapipe.model_complexity,
                min_detection_confidence=self.model_config.mediapipe.min_detection_confidence,
                min_tracking_confidence=self.model_config.mediapipe.min_tracking_confidence
            )
            logger.info("MediaPipe hands detector initialized")
        
        # YOLO detectors (optional)
        if self.config.detector.enable_yolo_hand and is_yolo_available():
            if self.model_config.yolo.weights_hand:
                self.hand_yolo_detector = create_yolo_detector(
                    self.model_config.yolo.weights_hand,
                    detector_type="hand",
                    device=self.config.runtime.device,
                    conf_threshold=self.config.detector.conf_thres,
                    iou_threshold=self.config.detector.iou_nms
                )
                logger.info("YOLO hand detector initialized")
        
        if self.config.detector.enable_yolo_object and is_yolo_available():
            if self.model_config.yolo.weights_object:
                self.object_detector = create_yolo_detector(
                    self.model_config.yolo.weights_object,
                    detector_type="object",
                    device=self.config.runtime.device,
                    conf_threshold=self.config.detector.conf_thres,
                    iou_threshold=self.config.detector.iou_nms
                )
                logger.info("YOLO object detector initialized")
        
        # Optical flow
        self.flow_computer = create_flow_computer(
            method=self.config.motion.flow_method
        )
        
        # Motion tracker
        self.motion_tracker = create_motion_tracker(
            history_size=30,
            smoothing_window=self.config.motion.min_frames_start
        )
        
        # In-hand detector
        self.inhand_detector = create_inhand_detector(
            iou_threshold=self.config.inhand.iou_min,
            distance_threshold=self.config.inhand.distance_threshold,
            center_inside=self.config.inhand.center_inside
        )
        
        # FSM
        self.fsm = create_inspection_fsm(
            start_threshold=self.config.motion.start_flow_mag,
            stop_threshold=self.config.motion.stop_flow_mag,
            min_frames_start=self.config.motion.min_frames_start,
            min_frames_stop=self.config.motion.min_frames_stop,
            grace_frames=self.config.motion.grace_frames,
            confidence_threshold=self.config.detector.conf_thres
        )
        
        # Event logger
        self.event_logger = EventLogger(
            output_dir=self.config.logging.out_dir,
            write_jsonl=self.config.logging.write_jsonl,
            write_csv=self.config.logging.write_csv
        )
        
        # Overlay renderer
        if self.config.runtime.draw_overlay:
            self.overlay_renderer = create_overlay_renderer(
                show_fps=self.config.runtime.show_fps,
                show_confidence=self.config.runtime.show_confidence
            )
    
    def start(self, video_source: Union[str, int, Path]) -> None:
        """
        Start processing pipeline.
        
        Args:
            video_source: Video input source
        """
        # Initialize video reader
        self.video_reader = create_video_reader(
            video_source,
            target_fps=self.config.fps_target,
            auto_reconnect=True
        )
        
        if not self.video_reader.is_opened:
            raise RuntimeError(f"Failed to open video source: {video_source}")
        
        logger.info(f"Started pipeline with source: {video_source}")
        
        self.is_running = True
        self.start_time = time.time()
        self.last_fps_time = self.start_time
        
        # Main processing loop
        try:
            self._process_loop()
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
        finally:
            self.stop()
    
    def _process_loop(self) -> None:
        """Main processing loop."""
        for frame in self.video_reader:
            if not self.is_running:
                break
            
            self.frame_count += 1
            current_time = time.time()
            
            # Process frame
            result = self.process_frame(frame, current_time)
            
            # Display result if overlay enabled
            if self.overlay_renderer and result.get('overlay') is not None:
                cv2.imshow('Hand Inspection Detection', result['overlay'])
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('r'):  # Reset
                    self._reset_tracking()
            
            # Update FPS
            self._update_fps(current_time)
            
            # Log events
            self._log_events()
    
    def process_frame(self, frame: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """
        Process single frame.
        
        Args:
            frame: Input frame
            timestamp: Frame timestamp
            
        Returns:
            Processing results dictionary
        """
        results = {
            'frame': frame,
            'timestamp': timestamp,
            'hands': [],
            'objects': [],
            'matches': [],
            'tracks': [],
            'states': {},
            'overlay': None
        }
        
        # Hand detection
        hands = self._detect_hands(frame)
        results['hands'] = hands
        
        # Object detection (optional)
        objects = self._detect_objects(frame)
        results['objects'] = objects
        
        # In-hand matching
        hand_boxes = [(h['bbox'], h['confidence']) for h in hands]
        object_boxes = [(o['bbox'], o['confidence']) for o in objects] if objects else None
        
        matches = self.inhand_detector.detect_in_hand(hand_boxes, object_boxes)
        results['matches'] = [m._asdict() for m in matches]
        
        # Optical flow computation
        flow_field = self.flow_computer.compute_flow(frame)
        
        # Motion tracking and FSM updates
        for i, match in enumerate(matches):
            if match.is_in_hand:
                track_id = i  # Simple track ID assignment
                
                # Create ROI for motion analysis
                roi = self.inhand_detector.create_roi_from_match(match)
                
                # Update motion tracker
                motion_metrics = self.motion_tracker.update(
                    track_id, roi, flow_field, timestamp
                )
                
                # Get motion score
                motion_score = self.motion_tracker.get_motion_score(track_id)
                
                # Update FSM
                transition = self.fsm.update(
                    track_id, motion_score, match.confidence, True, timestamp
                )
                
                # Store results
                results['tracks'].append({
                    'track_id': track_id,
                    'roi': roi,
                    'motion_metrics': motion_metrics._asdict(),
                    'motion_score': motion_score,
                    'state': self.fsm.get_state(track_id).value,
                    'is_inspecting': self.fsm.is_inspecting(track_id)
                })
                
                results['states'][track_id] = self.fsm.get_state(track_id).value
                
                if transition:
                    logger.debug(f"State transition: {transition}")
        
        # Create overlay visualization
        if self.overlay_renderer:
            overlay = self._create_overlay(frame, results)
            results['overlay'] = overlay
        
        return results
    
    def _detect_hands(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect hands in frame."""
        hands = []
        
        # MediaPipe detection
        if self.hand_detector:
            mp_hands = self.hand_detector.detect(frame)
            for detection in mp_hands:
                hands.append({
                    'bbox': detection.bbox,
                    'confidence': detection.confidence,
                    'source': 'mediapipe',
                    'landmarks': detection.landmarks,
                    'handedness': detection.handedness
                })
        
        # YOLO detection (if enabled)
        if self.hand_yolo_detector:
            yolo_hands = self.hand_yolo_detector.detect(frame)
            for detection in yolo_hands:
                hands.append({
                    'bbox': detection.bbox,
                    'confidence': detection.confidence,
                    'source': 'yolo',
                    'class_name': detection.class_name
                })
        
        return hands
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in frame."""
        objects = []
        
        if self.object_detector:
            detections = self.object_detector.detect(frame)
            for detection in detections:
                objects.append({
                    'bbox': detection.bbox,
                    'confidence': detection.confidence,
                    'source': 'yolo',
                    'class_name': detection.class_name
                })
        
        return objects
    
    def _create_overlay(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Create visualization overlay."""
        overlay = frame.copy()
        
        # Draw hands
        for hand in results['hands']:
            self.overlay_renderer.draw_hand(overlay, hand)
        
        # Draw objects
        for obj in results['objects']:
            self.overlay_renderer.draw_object(overlay, obj)
        
        # Draw tracks and states
        for track in results['tracks']:
            self.overlay_renderer.draw_track(overlay, track)
        
        # Draw FPS and statistics
        stats = {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'active_tracks': len(results['tracks']),
            'inspecting_tracks': sum(1 for t in results['tracks'] if t['is_inspecting'])
        }
        
        self.overlay_renderer.draw_statistics(overlay, stats)
        
        return overlay
    
    def _update_fps(self, current_time: float) -> None:
        """Update FPS calculation."""
        if current_time - self.last_fps_time >= 1.0:
            elapsed = current_time - self.last_fps_time
            self.fps = 1.0 / elapsed if elapsed > 0 else 0.0
            self.last_fps_time = current_time
    
    def _log_events(self) -> None:
        """Log completed inspection events."""
        completed_events = self.fsm.get_completed_events()
        
        # Convert to event logger format and log new events
        for event in completed_events:
            if event.end_time is not None:  # Only log completed events
                log_event = InspectionEvent.create_new(
                    track_id=event.track_id,
                    start_ts=event.start_time
                )
                log_event.finish(
                    end_ts=event.end_time,
                    avg_flow=event.avg_motion_score,
                    min_conf=event.confidence,
                    max_conf=event.confidence
                )
                
                self.event_logger.log_event(log_event)
        
        # Clear completed events from FSM
        self.fsm.completed_events.clear()
    
    def _reset_tracking(self) -> None:
        """Reset all tracking state."""
        if self.motion_tracker:
            self.motion_tracker.clear()
        
        if self.fsm:
            # Remove all tracks
            for track_id in list(self.fsm.states.keys()):
                self.fsm.remove_track(track_id)
        
        if self.flow_computer:
            self.flow_computer.reset()
        
        logger.info("Tracking state reset")
    
    def stop(self) -> None:
        """Stop processing pipeline."""
        self.is_running = False
        
        # Close video reader
        if self.video_reader:
            self.video_reader.release()
        
        # Close detectors
        if self.hand_detector:
            self.hand_detector.close()
        
        # Close event logger
        if self.event_logger:
            self.event_logger.close()
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        # Log final statistics
        if self.fsm:
            stats = self.fsm.get_statistics()
            logger.info(f"Final statistics: {stats}")
        
        logger.info("Pipeline stopped")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            'frame_count': self.frame_count,
            'fps': self.fps,
            'runtime': time.time() - self.start_time if self.start_time > 0 else 0,
            'is_running': self.is_running
        }
        
        if self.fsm:
            stats.update(self.fsm.get_statistics())
        
        if self.event_logger:
            stats.update(self.event_logger.get_statistics())
        
        return stats
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def create_pipeline(config_path: Union[str, Path], model_config_path: Optional[Union[str, Path]] = None) -> Pipeline:
    """
    Factory function to create pipeline from configuration files.
    
    Args:
        config_path: Path to main configuration file
        model_config_path: Path to model configuration file
        
    Returns:
        Pipeline instance
    """
    from .config import load_config
    
    config, model_config = load_config(config_path, model_config_path)
    
    return Pipeline(config, model_config)
