"""Test suite for finite state machine (FSM) module."""

import pytest
import time
from src.logic.fsm import InspectionFSM, InspectionState, create_inspection_fsm


class TestInspectionFSM:
    """Test cases for InspectionFSM class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.fsm = create_inspection_fsm(
            start_threshold=2.0,
            stop_threshold=1.0,
            min_frames_start=3,
            min_frames_stop=3,
            grace_frames=5,
            confidence_threshold=0.7
        )
    
    def test_initialization(self):
        """Test FSM initialization."""
        assert self.fsm.start_threshold == 2.0
        assert self.fsm.stop_threshold == 1.0
        assert self.fsm.min_frames_start == 3
        assert self.fsm.min_frames_stop == 3
        assert self.fsm.grace_frames == 5
        assert self.fsm.confidence_threshold == 0.7
        
        assert len(self.fsm.states) == 0
        assert len(self.fsm.completed_events) == 0
    
    def test_track_initialization(self):
        """Test automatic track initialization."""
        track_id = 1
        
        # First update should initialize track
        transition = self.fsm.update(track_id, 0.5, 0.8, True)
        
        assert track_id in self.fsm.states
        assert self.fsm.get_state(track_id) == InspectionState.IDLE
        assert transition is None  # No state change on initialization
    
    def test_idle_to_inspecting_transition(self):
        """Test transition from IDLE to INSPECTING."""
        track_id = 1
        timestamp = time.time()
        
        # Send high motion scores for required frames
        transitions = []
        for i in range(self.fsm.min_frames_start):
            transition = self.fsm.update(track_id, 3.0, 0.8, True, timestamp + i)
            if transition:
                transitions.append(transition)
        
        # Should transition to INSPECTING after min_frames_start
        assert len(transitions) == 1
        assert transitions[0].to_state == InspectionState.INSPECTING
        assert self.fsm.is_inspecting(track_id)
        
        # Should have created inspection event
        current_events = self.fsm.get_current_events()
        assert len(current_events) == 1
        assert current_events[0].track_id == track_id
    
    def test_inspecting_to_idle_transition(self):
        """Test transition from INSPECTING to IDLE."""
        track_id = 1
        timestamp = time.time()
        
        # First transition to INSPECTING
        for i in range(self.fsm.min_frames_start):
            self.fsm.update(track_id, 3.0, 0.8, True, timestamp + i)
        
        assert self.fsm.is_inspecting(track_id)
        
        # Send low motion scores to transition back to IDLE
        transitions = []
        for i in range(self.fsm.min_frames_stop):
            transition = self.fsm.update(track_id, 0.5, 0.8, True, timestamp + 10 + i)
            if transition:
                transitions.append(transition)
        
        # Should transition back to IDLE
        assert len(transitions) == 1
        assert transitions[0].to_state == InspectionState.IDLE
        assert not self.fsm.is_inspecting(track_id)
        
        # Should have completed inspection event
        completed_events = self.fsm.get_completed_events()
        assert len(completed_events) == 1
        assert completed_events[0].track_id == track_id
        assert completed_events[0].duration is not None
    
    def test_hysteresis_behavior(self):
        """Test hysteresis (different start/stop thresholds)."""
        track_id = 1
        
        # Motion score between stop and start thresholds
        mid_score = (self.fsm.start_threshold + self.fsm.stop_threshold) / 2
        
        # Should not start inspection with mid-level motion
        for i in range(10):
            transition = self.fsm.update(track_id, mid_score, 0.8, True)
            assert transition is None
        
        assert self.fsm.get_state(track_id) == InspectionState.IDLE
        
        # Start inspection with high motion
        for i in range(self.fsm.min_frames_start):
            self.fsm.update(track_id, 3.0, 0.8, True)
        
        assert self.fsm.is_inspecting(track_id)
        
        # Mid-level motion should not stop inspection
        for i in range(10):
            transition = self.fsm.update(track_id, mid_score, 0.8, True)
            assert transition is None
        
        assert self.fsm.is_inspecting(track_id)
    
    def test_confidence_threshold(self):
        """Test confidence threshold behavior."""
        track_id = 1
        
        # High motion but low confidence should not start inspection
        for i in range(10):
            transition = self.fsm.update(track_id, 3.0, 0.5, True)  # Below threshold
            assert transition is None
        
        assert self.fsm.get_state(track_id) == InspectionState.IDLE
        
        # High motion and high confidence should start inspection
        for i in range(self.fsm.min_frames_start):
            self.fsm.update(track_id, 3.0, 0.8, True)
        
        assert self.fsm.is_inspecting(track_id)
        
        # Drop confidence should stop inspection
        transition = self.fsm.update(track_id, 3.0, 0.5, True)
        assert transition is not None
        assert transition.to_state == InspectionState.IDLE
    
    def test_detection_loss_grace_period(self):
        """Test grace period for detection loss."""
        track_id = 1
        
        # Start inspection
        for i in range(self.fsm.min_frames_start):
            self.fsm.update(track_id, 3.0, 0.8, True)
        
        assert self.fsm.is_inspecting(track_id)
        
        # Simulate detection loss within grace period
        for i in range(self.fsm.grace_frames - 1):
            transition = self.fsm.update(track_id, 0.0, 0.0, False)
            assert transition is None  # Should not transition yet
        
        assert self.fsm.is_inspecting(track_id)
        
        # Exceed grace period
        transition = self.fsm.update(track_id, 0.0, 0.0, False)
        assert transition is not None
        assert transition.to_state == InspectionState.IDLE
    
    def test_detection_recovery(self):
        """Test recovery from detection loss."""
        track_id = 1
        
        # Start inspection
        for i in range(self.fsm.min_frames_start):
            self.fsm.update(track_id, 3.0, 0.8, True)
        
        assert self.fsm.is_inspecting(track_id)
        
        # Brief detection loss
        for i in range(2):
            self.fsm.update(track_id, 0.0, 0.0, False)
        
        # Recovery - should reset grace counter
        transition = self.fsm.update(track_id, 3.0, 0.8, True)
        assert transition is None
        assert self.fsm.is_inspecting(track_id)
        assert self.fsm.grace_counters[track_id] == 0
    
    def test_multiple_tracks(self):
        """Test handling multiple tracks simultaneously."""
        track_ids = [1, 2, 3]
        
        # Start inspection for all tracks
        for track_id in track_ids:
            for i in range(self.fsm.min_frames_start):
                self.fsm.update(track_id, 3.0, 0.8, True)
        
        # All should be inspecting
        for track_id in track_ids:
            assert self.fsm.is_inspecting(track_id)
        
        # Stop inspection for one track
        for i in range(self.fsm.min_frames_stop):
            self.fsm.update(track_ids[0], 0.5, 0.8, True)
        
        # Only first track should stop
        assert not self.fsm.is_inspecting(track_ids[0])
        assert self.fsm.is_inspecting(track_ids[1])
        assert self.fsm.is_inspecting(track_ids[2])
    
    def test_remove_track(self):
        """Test track removal and cleanup."""
        track_id = 1
        
        # Start inspection
        for i in range(self.fsm.min_frames_start):
            self.fsm.update(track_id, 3.0, 0.8, True)
        
        assert self.fsm.is_inspecting(track_id)
        assert len(self.fsm.get_current_events()) == 1
        
        # Remove track
        self.fsm.remove_track(track_id)
        
        # Should be cleaned up
        assert track_id not in self.fsm.states
        assert len(self.fsm.get_current_events()) == 0
        assert len(self.fsm.get_completed_events()) == 1  # Should complete event
    
    def test_statistics(self):
        """Test statistics collection."""
        track_ids = [1, 2]
        
        # Initial statistics
        stats = self.fsm.get_statistics()
        assert stats['active_tracks'] == 0
        assert stats['inspecting_tracks'] == 0
        assert stats['completed_events'] == 0
        
        # Add some tracks
        for track_id in track_ids:
            self.fsm.update(track_id, 1.0, 0.8, True)
        
        stats = self.fsm.get_statistics()
        assert stats['active_tracks'] == 2
        assert stats['inspecting_tracks'] == 0
        
        # Start inspection for one track
        for i in range(self.fsm.min_frames_start):
            self.fsm.update(track_ids[0], 3.0, 0.8, True)
        
        stats = self.fsm.get_statistics()
        assert stats['active_tracks'] == 2
        assert stats['inspecting_tracks'] == 1
        assert stats['current_events'] == 1
    
    def test_threshold_updates(self):
        """Test dynamic threshold updates."""
        original_start = self.fsm.start_threshold
        original_stop = self.fsm.stop_threshold
        
        # Update thresholds
        new_start = 5.0
        new_stop = 2.0
        
        self.fsm.update_thresholds(
            start_threshold=new_start,
            stop_threshold=new_stop
        )
        
        assert self.fsm.start_threshold == new_start
        assert self.fsm.stop_threshold == new_stop
        
        # Partial update
        self.fsm.update_thresholds(start_threshold=original_start)
        assert self.fsm.start_threshold == original_start
        assert self.fsm.stop_threshold == new_stop  # Should remain unchanged


def test_create_inspection_fsm():
    """Test factory function."""
    fsm = create_inspection_fsm(
        start_threshold=1.5,
        stop_threshold=0.8,
        min_frames_start=5
    )
    
    assert isinstance(fsm, InspectionFSM)
    assert fsm.start_threshold == 1.5
    assert fsm.stop_threshold == 0.8
    assert fsm.min_frames_start == 5


# Additional integration test
def test_full_inspection_cycle():
    """Test complete inspection detection cycle."""
    fsm = create_inspection_fsm(
        start_threshold=2.0,
        stop_threshold=1.0,
        min_frames_start=3,
        min_frames_stop=3
    )
    
    track_id = 1
    timestamps = []
    
    # Simulate inspection sequence
    # 1. Start with low motion (idle)
    base_time = time.time()
    for i in range(5):
        timestamp = base_time + i * 0.1
        timestamps.append(timestamp)
        transition = fsm.update(track_id, 0.5, 0.8, True, timestamp)
        assert transition is None
    
    assert fsm.get_state(track_id) == InspectionState.IDLE
    
    # 2. Increase motion to start inspection
    for i in range(3):
        timestamp = base_time + (5 + i) * 0.1
        timestamps.append(timestamp)
        transition = fsm.update(track_id, 3.0, 0.8, True, timestamp)
        if transition:
            assert transition.to_state == InspectionState.INSPECTING
    
    assert fsm.is_inspecting(track_id)
    
    # 3. Continue with high motion
    for i in range(5):
        timestamp = base_time + (8 + i) * 0.1
        timestamps.append(timestamp)
        transition = fsm.update(track_id, 2.5, 0.8, True, timestamp)
        assert transition is None
    
    # 4. Decrease motion to stop inspection
    for i in range(3):
        timestamp = base_time + (13 + i) * 0.1
        timestamps.append(timestamp)
        transition = fsm.update(track_id, 0.5, 0.8, True, timestamp)
        if transition:
            assert transition.to_state == InspectionState.IDLE
    
    assert not fsm.is_inspecting(track_id)
    
    # Check completed event
    completed_events = fsm.get_completed_events()
    assert len(completed_events) == 1
    
    event = completed_events[0]
    assert event.track_id == track_id
    assert event.duration is not None
    assert event.duration > 0
