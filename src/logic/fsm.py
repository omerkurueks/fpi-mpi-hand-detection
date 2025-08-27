"""Finite State Machine for inspection detection."""

from enum import Enum
from typing import Dict, List, Optional, NamedTuple
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)


class InspectionState(Enum):
    """States for inspection detection."""
    IDLE = "idle"
    HOLDING = "holding"
    INSPECTING = "inspecting"


class StateTransition(NamedTuple):
    """State transition data."""
    from_state: InspectionState
    to_state: InspectionState
    timestamp: float
    trigger: str
    motion_score: float


class InspectionEvent(NamedTuple):
    """Inspection event data."""
    track_id: int
    start_time: float
    end_time: Optional[float]
    duration: Optional[float]
    avg_motion_score: float
    max_motion_score: float
    confidence: float


class InspectionFSM:
    """Finite State Machine for inspection detection."""
    
    def __init__(
        self,
        start_threshold: float = 1.5,
        stop_threshold: float = 0.8,
        min_frames_start: int = 6,
        min_frames_stop: int = 8,
        grace_frames: int = 10,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize inspection FSM.
        
        Args:
            start_threshold: Motion score threshold to start inspection
            stop_threshold: Motion score threshold to stop inspection
            min_frames_start: Minimum consecutive frames above threshold to start
            min_frames_stop: Minimum consecutive frames below threshold to stop
            grace_frames: Grace period for temporary detection loss
            confidence_threshold: Minimum detection confidence
        """
        self.start_threshold = start_threshold
        self.stop_threshold = stop_threshold
        self.min_frames_start = min_frames_start
        self.min_frames_stop = min_frames_stop
        self.grace_frames = grace_frames
        self.confidence_threshold = confidence_threshold
        
        # State tracking per track ID
        self.states: Dict[int, InspectionState] = {}
        self.frame_counters: Dict[int, int] = {}
        self.grace_counters: Dict[int, int] = {}
        self.motion_history: Dict[int, deque] = {}
        self.confidence_history: Dict[int, deque] = {}
        
        # Event tracking
        self.current_events: Dict[int, InspectionEvent] = {}
        self.completed_events: List[InspectionEvent] = []
        self.state_transitions: List[StateTransition] = []
        
        # Statistics
        self.frame_count = 0
        
        logger.info(f"Inspection FSM initialized: start={start_threshold}, stop={stop_threshold}")
    
    def update(
        self,
        track_id: int,
        motion_score: float,
        confidence: float = 1.0,
        is_detected: bool = True,
        timestamp: Optional[float] = None
    ) -> Optional[StateTransition]:
        """
        Update FSM state for a track.
        
        Args:
            track_id: Track identifier
            motion_score: Current motion score
            confidence: Detection confidence
            is_detected: Whether object is currently detected
            timestamp: Current timestamp (uses current time if None)
            
        Returns:
            State transition if state changed, None otherwise
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.frame_count += 1
        
        # Initialize track if new
        if track_id not in self.states:
            self._initialize_track(track_id)
        
        current_state = self.states[track_id]
        
        # Update history
        self._update_history(track_id, motion_score, confidence)
        
        # Handle detection loss
        if not is_detected:
            return self._handle_detection_loss(track_id, timestamp)
        
        # Reset grace counter on successful detection
        self.grace_counters[track_id] = 0
        
        # State machine logic
        transition = None
        
        if current_state == InspectionState.IDLE:
            transition = self._handle_idle_state(track_id, motion_score, confidence, timestamp)
        
        elif current_state == InspectionState.HOLDING:
            transition = self._handle_holding_state(track_id, motion_score, confidence, timestamp)
        
        elif current_state == InspectionState.INSPECTING:
            transition = self._handle_inspecting_state(track_id, motion_score, confidence, timestamp)
        
        return transition
    
    def _initialize_track(self, track_id: int) -> None:
        """Initialize tracking for new track ID."""
        self.states[track_id] = InspectionState.IDLE
        self.frame_counters[track_id] = 0
        self.grace_counters[track_id] = 0
        self.motion_history[track_id] = deque(maxlen=30)
        self.confidence_history[track_id] = deque(maxlen=30)
        
        logger.debug(f"Initialized tracking for track {track_id}")
    
    def _update_history(self, track_id: int, motion_score: float, confidence: float) -> None:
        """Update motion and confidence history."""
        self.motion_history[track_id].append(motion_score)
        self.confidence_history[track_id].append(confidence)
    
    def _handle_detection_loss(self, track_id: int, timestamp: float) -> Optional[StateTransition]:
        """Handle temporary detection loss."""
        self.grace_counters[track_id] += 1
        
        # If grace period exceeded, stop any ongoing inspection
        if self.grace_counters[track_id] > self.grace_frames:
            if self.states[track_id] == InspectionState.INSPECTING:
                return self._transition_to_idle(track_id, timestamp, "detection_lost")
        
        return None
    
    def _handle_idle_state(
        self,
        track_id: int,
        motion_score: float,
        confidence: float,
        timestamp: float
    ) -> Optional[StateTransition]:
        """Handle IDLE state logic."""
        # Check if motion and confidence are sufficient to start inspection
        if motion_score >= self.start_threshold and confidence >= self.confidence_threshold:
            self.frame_counters[track_id] += 1
            
            if self.frame_counters[track_id] >= self.min_frames_start:
                return self._transition_to_inspecting(track_id, timestamp, "motion_detected")
        else:
            self.frame_counters[track_id] = 0
        
        return None
    
    def _handle_holding_state(
        self,
        track_id: int,
        motion_score: float,
        confidence: float,
        timestamp: float
    ) -> Optional[StateTransition]:
        """Handle HOLDING state logic (optional intermediate state)."""
        # Transition to inspecting if motion increases
        if motion_score >= self.start_threshold:
            return self._transition_to_inspecting(track_id, timestamp, "inspection_started")
        
        # Transition to idle if object no longer detected confidently
        if confidence < self.confidence_threshold:
            return self._transition_to_idle(track_id, timestamp, "low_confidence")
        
        return None
    
    def _handle_inspecting_state(
        self,
        track_id: int,
        motion_score: float,
        confidence: float,
        timestamp: float
    ) -> Optional[StateTransition]:
        """Handle INSPECTING state logic."""
        # Check if motion has decreased sufficiently to stop inspection
        if motion_score <= self.stop_threshold:
            self.frame_counters[track_id] += 1
            
            if self.frame_counters[track_id] >= self.min_frames_stop:
                return self._transition_to_idle(track_id, timestamp, "motion_stopped")
        else:
            self.frame_counters[track_id] = 0
        
        # Also check confidence
        if confidence < self.confidence_threshold:
            return self._transition_to_idle(track_id, timestamp, "low_confidence")
        
        return None
    
    def _transition_to_inspecting(
        self,
        track_id: int,
        timestamp: float,
        trigger: str
    ) -> StateTransition:
        """Transition to INSPECTING state."""
        old_state = self.states[track_id]
        self.states[track_id] = InspectionState.INSPECTING
        self.frame_counters[track_id] = 0
        
        # Start new inspection event
        motion_score = self._get_average_motion_score(track_id)
        confidence = self._get_average_confidence(track_id)
        
        event = InspectionEvent(
            track_id=track_id,
            start_time=timestamp,
            end_time=None,
            duration=None,
            avg_motion_score=motion_score,
            max_motion_score=motion_score,
            confidence=confidence
        )
        
        self.current_events[track_id] = event
        
        transition = StateTransition(
            from_state=old_state,
            to_state=InspectionState.INSPECTING,
            timestamp=timestamp,
            trigger=trigger,
            motion_score=motion_score
        )
        
        self.state_transitions.append(transition)
        
        logger.info(f"Track {track_id}: {old_state.value} -> {InspectionState.INSPECTING.value} ({trigger})")
        
        return transition
    
    def _transition_to_idle(
        self,
        track_id: int,
        timestamp: float,
        trigger: str
    ) -> StateTransition:
        """Transition to IDLE state."""
        old_state = self.states[track_id]
        self.states[track_id] = InspectionState.IDLE
        self.frame_counters[track_id] = 0
        
        # Complete inspection event if exists
        if track_id in self.current_events:
            event = self.current_events[track_id]
            
            # Update event with end information
            completed_event = InspectionEvent(
                track_id=event.track_id,
                start_time=event.start_time,
                end_time=timestamp,
                duration=timestamp - event.start_time,
                avg_motion_score=self._get_average_motion_score(track_id),
                max_motion_score=self._get_max_motion_score(track_id),
                confidence=self._get_average_confidence(track_id)
            )
            
            self.completed_events.append(completed_event)
            del self.current_events[track_id]
            
            logger.info(f"Inspection completed for track {track_id}: {completed_event.duration:.2f}s")
        
        motion_score = self._get_average_motion_score(track_id)
        
        transition = StateTransition(
            from_state=old_state,
            to_state=InspectionState.IDLE,
            timestamp=timestamp,
            trigger=trigger,
            motion_score=motion_score
        )
        
        self.state_transitions.append(transition)
        
        logger.info(f"Track {track_id}: {old_state.value} -> {InspectionState.IDLE.value} ({trigger})")
        
        return transition
    
    def _get_average_motion_score(self, track_id: int, window: int = 10) -> float:
        """Get average motion score from recent history."""
        if track_id not in self.motion_history:
            return 0.0
        
        history = list(self.motion_history[track_id])
        if not history:
            return 0.0
        
        recent = history[-window:] if len(history) >= window else history
        return sum(recent) / len(recent)
    
    def _get_max_motion_score(self, track_id: int, window: int = 30) -> float:
        """Get maximum motion score from recent history."""
        if track_id not in self.motion_history:
            return 0.0
        
        history = list(self.motion_history[track_id])
        if not history:
            return 0.0
        
        recent = history[-window:] if len(history) >= window else history
        return max(recent)
    
    def _get_average_confidence(self, track_id: int, window: int = 10) -> float:
        """Get average confidence from recent history."""
        if track_id not in self.confidence_history:
            return 0.0
        
        history = list(self.confidence_history[track_id])
        if not history:
            return 0.0
        
        recent = history[-window:] if len(history) >= window else history
        return sum(recent) / len(recent)
    
    def get_state(self, track_id: int) -> InspectionState:
        """Get current state for track."""
        return self.states.get(track_id, InspectionState.IDLE)
    
    def is_inspecting(self, track_id: int) -> bool:
        """Check if track is currently inspecting."""
        return self.get_state(track_id) == InspectionState.INSPECTING
    
    def get_current_events(self) -> List[InspectionEvent]:
        """Get currently active inspection events."""
        return list(self.current_events.values())
    
    def get_completed_events(self) -> List[InspectionEvent]:
        """Get completed inspection events."""
        return self.completed_events.copy()
    
    def get_all_events(self) -> List[InspectionEvent]:
        """Get all events (current + completed)."""
        return self.get_completed_events() + self.get_current_events()
    
    def remove_track(self, track_id: int) -> None:
        """Remove track from FSM (cleanup)."""
        # Complete any ongoing inspection
        if track_id in self.current_events:
            timestamp = time.time()
            self._transition_to_idle(track_id, timestamp, "track_removed")
        
        # Remove from all dictionaries
        self.states.pop(track_id, None)
        self.frame_counters.pop(track_id, None)
        self.grace_counters.pop(track_id, None)
        self.motion_history.pop(track_id, None)
        self.confidence_history.pop(track_id, None)
        
        logger.debug(f"Removed track {track_id} from FSM")
    
    def get_statistics(self) -> Dict[str, any]:
        """Get FSM statistics."""
        active_tracks = len(self.states)
        inspecting_tracks = sum(1 for state in self.states.values() 
                               if state == InspectionState.INSPECTING)
        
        total_events = len(self.completed_events)
        total_duration = sum(event.duration or 0 for event in self.completed_events)
        avg_duration = total_duration / total_events if total_events > 0 else 0
        
        return {
            'active_tracks': active_tracks,
            'inspecting_tracks': inspecting_tracks,
            'completed_events': total_events,
            'current_events': len(self.current_events),
            'total_duration': total_duration,
            'avg_duration': avg_duration,
            'frame_count': self.frame_count
        }
    
    def update_thresholds(
        self,
        start_threshold: Optional[float] = None,
        stop_threshold: Optional[float] = None,
        min_frames_start: Optional[int] = None,
        min_frames_stop: Optional[int] = None
    ) -> None:
        """Update FSM thresholds."""
        if start_threshold is not None:
            self.start_threshold = start_threshold
        if stop_threshold is not None:
            self.stop_threshold = stop_threshold
        if min_frames_start is not None:
            self.min_frames_start = min_frames_start
        if min_frames_stop is not None:
            self.min_frames_stop = min_frames_stop
        
        logger.info(f"FSM thresholds updated: start={self.start_threshold}, stop={self.stop_threshold}")


def create_inspection_fsm(**kwargs) -> InspectionFSM:
    """
    Factory function to create inspection FSM.
    
    Args:
        **kwargs: Arguments for InspectionFSM
        
    Returns:
        InspectionFSM instance
    """
    return InspectionFSM(**kwargs)
