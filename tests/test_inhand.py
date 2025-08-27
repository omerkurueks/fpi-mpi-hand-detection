"""Test suite for in-hand logic module."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from src.logic.inhand import InHandLogic, create_inhand_logic
from src.detect.hands_mediapipe import HandDetection, HandLandmarks
from src.motion.metrics import MotionMetrics


class TestInHandLogic:
    """Test cases for InHandLogic class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Mock configuration
        self.config = Mock()
        self.config.min_confidence = 0.5
        self.config.iou_threshold = 0.3
        self.config.matching_strategy = 'center_iou'
        self.config.match_threshold = 0.5
        self.config.max_distance = 100.0
        
        # Create logic instance
        self.logic = InHandLogic(self.config)
    
    def test_initialization(self):
        """Test logic initialization."""
        assert self.logic.config == self.config
        assert len(self.logic.tracks) == 0
        assert self.logic.next_track_id == 1
    
    def test_bbox_to_center(self):
        """Test bounding box center calculation."""
        bbox = [10, 20, 50, 60]  # x1, y1, x2, y2
        center = self.logic._bbox_to_center(bbox)
        expected = [30.0, 40.0]  # (10+50)/2, (20+60)/2
        assert center == expected
    
    def test_calculate_iou(self):
        """Test IoU calculation."""
        bbox1 = [0, 0, 10, 10]
        bbox2 = [5, 5, 15, 15]
        
        iou = self.logic._calculate_iou(bbox1, bbox2)
        
        # Intersection: 5x5 = 25
        # Union: 100 + 100 - 25 = 175
        # IoU: 25/175 ≈ 0.143
        assert abs(iou - 0.142857) < 0.001
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU with no overlap."""
        bbox1 = [0, 0, 10, 10]
        bbox2 = [20, 20, 30, 30]
        
        iou = self.logic._calculate_iou(bbox1, bbox2)
        assert iou == 0.0
    
    def test_calculate_iou_complete_overlap(self):
        """Test IoU with complete overlap."""
        bbox = [0, 0, 10, 10]
        
        iou = self.logic._calculate_iou(bbox, bbox)
        assert iou == 1.0
    
    def test_calculate_distance(self):
        """Test Euclidean distance calculation."""
        point1 = [0, 0]
        point2 = [3, 4]
        
        distance = self.logic._calculate_distance(point1, point2)
        assert distance == 5.0  # 3-4-5 triangle
    
    def test_match_objects_center_strategy(self):
        """Test object matching with center strategy."""
        # Create hand detection
        hand_landmarks = HandLandmarks(
            landmarks=np.zeros((21, 3)),  # 21 landmarks with x,y,z
            bbox=[10, 10, 50, 50],
            confidence=0.8,
            handedness='Right',
            grip_strength=0.7
        )
        hand = HandDetection(
            landmarks=[hand_landmarks],
            frame_shape=(480, 640, 3)
        )
        
        # Create object detections (mock YOLO format)
        objects = [
            [15, 15, 45, 45, 0.9, 0],  # Close to hand
            [100, 100, 120, 120, 0.8, 1],  # Far from hand
        ]
        
        matches = self.logic._match_objects(hand, objects)
        
        # Should match first object (close to hand center)
        assert len(matches) == 1
        assert matches[0] == (0, 0)  # hand_idx=0, obj_idx=0
    
    def test_match_objects_iou_strategy(self):
        """Test object matching with IoU strategy."""
        self.config.matching_strategy = 'center_iou'
        
        # Create overlapping hand and object
        hand_landmarks = HandLandmarks(
            landmarks=np.zeros((21, 3)),
            bbox=[10, 10, 50, 50],
            confidence=0.8,
            handedness='Right',
            grip_strength=0.7
        )
        hand = HandDetection(
            landmarks=[hand_landmarks],
            frame_shape=(480, 640, 3)
        )
        
        objects = [
            [30, 30, 70, 70, 0.9, 0],  # Overlapping with hand
        ]
        
        matches = self.logic._match_objects(hand, objects)
        assert len(matches) == 1
    
    def test_match_objects_no_matches(self):
        """Test object matching when no objects are close enough."""
        hand_landmarks = HandLandmarks(
            landmarks=np.zeros((21, 3)),
            bbox=[10, 10, 50, 50],
            confidence=0.8,
            handedness='Right',
            grip_strength=0.7
        )
        hand = HandDetection(
            landmarks=[hand_landmarks],
            frame_shape=(480, 640, 3)
        )
        
        # Object too far away
        objects = [
            [200, 200, 220, 220, 0.9, 0],
        ]
        
        matches = self.logic._match_objects(hand, objects)
        assert len(matches) == 0
    
    def test_create_new_track(self):
        """Test creation of new track."""
        hand_landmarks = HandLandmarks(
            landmarks=np.zeros((21, 3)),
            bbox=[10, 10, 50, 50],
            confidence=0.8,
            handedness='Right',
            grip_strength=0.7
        )
        
        object_bbox = [15, 15, 45, 45]
        motion_metrics = MotionMetrics(
            flow_magnitude=2.5,
            centroid_movement=1.0,
            area_ratio=1.1,
            bbox_change=0.8
        )
        
        track = self.logic._create_new_track(
            hand_landmarks, object_bbox, motion_metrics, 0.9, 123.45
        )
        
        assert track.track_id == 1
        assert track.hand_bbox == [10, 10, 50, 50]
        assert track.object_bbox == object_bbox
        assert track.confidence == 0.9
        assert track.timestamp == 123.45
        assert track.motion_metrics == motion_metrics
        assert self.logic.next_track_id == 2
    
    def test_update_existing_track(self):
        """Test updating existing track."""
        # Create initial track
        hand_landmarks = HandLandmarks(
            landmarks=np.zeros((21, 3)),
            bbox=[10, 10, 50, 50],
            confidence=0.8,
            handedness='Right',
            grip_strength=0.7
        )
        
        motion_metrics = MotionMetrics(
            flow_magnitude=2.5,
            centroid_movement=1.0,
            area_ratio=1.1,
            bbox_change=0.8
        )
        
        track = self.logic._create_new_track(
            hand_landmarks, [15, 15, 45, 45], motion_metrics, 0.9, 100.0
        )
        self.logic.tracks[track.track_id] = track
        
        # Update track
        new_hand_landmarks = HandLandmarks(
            landmarks=np.zeros((21, 3)),
            bbox=[12, 12, 52, 52],
            confidence=0.85,
            handedness='Right',
            grip_strength=0.8
        )
        
        new_motion_metrics = MotionMetrics(
            flow_magnitude=3.0,
            centroid_movement=1.5,
            area_ratio=1.2,
            bbox_change=1.0
        )
        
        self.logic._update_track(
            track.track_id, new_hand_landmarks, [17, 17, 47, 47], 
            new_motion_metrics, 0.95, 200.0
        )
        
        updated_track = self.logic.tracks[track.track_id]
        assert updated_track.hand_bbox == [12, 12, 52, 52]
        assert updated_track.object_bbox == [17, 17, 47, 47]
        assert updated_track.confidence == 0.95
        assert updated_track.timestamp == 200.0
        assert updated_track.motion_metrics == new_motion_metrics
    
    def test_process_frame_new_detection(self):
        """Test processing frame with new hand-object detection."""
        # Mock dependencies
        motion_analyzer = Mock()
        motion_analyzer.analyze_region.return_value = MotionMetrics(
            flow_magnitude=2.5,
            centroid_movement=1.0,
            area_ratio=1.1,
            bbox_change=0.8
        )
        
        # Create hand detection
        hand_landmarks = HandLandmarks(
            landmarks=np.zeros((21, 3)),
            bbox=[10, 10, 50, 50],
            confidence=0.8,
            handedness='Right',
            grip_strength=0.7
        )
        hand_detection = HandDetection(
            landmarks=[hand_landmarks],
            frame_shape=(480, 640, 3)
        )
        
        object_detections = [
            [15, 15, 45, 45, 0.9, 0],
        ]
        
        tracks = self.logic.process_frame(
            hand_detection, object_detections, motion_analyzer, 100.0
        )
        
        assert len(tracks) == 1
        assert tracks[0].track_id == 1
        assert len(self.logic.tracks) == 1
    
    def test_process_frame_update_existing(self):
        """Test processing frame that updates existing track."""
        # Create existing track
        hand_landmarks = HandLandmarks(
            landmarks=np.zeros((21, 3)),
            bbox=[10, 10, 50, 50],
            confidence=0.8,
            handedness='Right',
            grip_strength=0.7
        )
        
        motion_metrics = MotionMetrics(
            flow_magnitude=2.5,
            centroid_movement=1.0,
            area_ratio=1.1,
            bbox_change=0.8
        )
        
        existing_track = self.logic._create_new_track(
            hand_landmarks, [15, 15, 45, 45], motion_metrics, 0.9, 100.0
        )
        self.logic.tracks[existing_track.track_id] = existing_track
        
        # Mock motion analyzer
        motion_analyzer = Mock()
        motion_analyzer.analyze_region.return_value = MotionMetrics(
            flow_magnitude=3.0,
            centroid_movement=1.5,
            area_ratio=1.2,
            bbox_change=1.0
        )
        
        # Create similar detection (should update existing track)
        new_hand_detection = HandDetection(
            landmarks=[hand_landmarks],
            frame_shape=(480, 640, 3)
        )
        
        object_detections = [
            [16, 16, 46, 46, 0.95, 0],  # Slightly moved
        ]
        
        tracks = self.logic.process_frame(
            new_hand_detection, object_detections, motion_analyzer, 200.0
        )
        
        assert len(tracks) == 1
        assert tracks[0].track_id == existing_track.track_id
        assert len(self.logic.tracks) == 1
        assert tracks[0].timestamp == 200.0
    
    def test_get_statistics(self):
        """Test statistics collection."""
        # Empty state
        stats = self.logic.get_statistics()
        assert stats['total_tracks'] == 0
        assert stats['active_tracks'] == 0
        
        # Add some tracks
        for i in range(3):
            hand_landmarks = HandLandmarks(
                landmarks=np.zeros((21, 3)),
                bbox=[i*10, i*10, i*10+40, i*10+40],
                confidence=0.8,
                handedness='Right',
                grip_strength=0.7
            )
            
            motion_metrics = MotionMetrics(
                flow_magnitude=2.5,
                centroid_movement=1.0,
                area_ratio=1.1,
                bbox_change=0.8
            )
            
            track = self.logic._create_new_track(
                hand_landmarks, [i*10+5, i*10+5, i*10+35, i*10+35], 
                motion_metrics, 0.9, 100.0
            )
            self.logic.tracks[track.track_id] = track
        
        stats = self.logic.get_statistics()
        assert stats['total_tracks'] == 3
        assert stats['active_tracks'] == 3
    
    def test_track_cleanup(self):
        """Test track cleanup functionality."""
        # Create some tracks
        track_ids = []
        for i in range(3):
            hand_landmarks = HandLandmarks(
                landmarks=np.zeros((21, 3)),
                bbox=[i*10, i*10, i*10+40, i*10+40],
                confidence=0.8,
                handedness='Right',
                grip_strength=0.7
            )
            
            motion_metrics = MotionMetrics(
                flow_magnitude=2.5,
                centroid_movement=1.0,
                area_ratio=1.1,
                bbox_change=0.8
            )
            
            track = self.logic._create_new_track(
                hand_landmarks, [i*10+5, i*10+5, i*10+35, i*10+35], 
                motion_metrics, 0.9, 100.0 + i
            )
            self.logic.tracks[track.track_id] = track
            track_ids.append(track.track_id)
        
        assert len(self.logic.tracks) == 3
        
        # Remove specific tracks
        removed = self.logic.cleanup_tracks(track_ids[1:])  # Remove last 2
        
        assert len(removed) == 2
        assert len(self.logic.tracks) == 1
        assert track_ids[0] in self.logic.tracks
        assert track_ids[1] not in self.logic.tracks
        assert track_ids[2] not in self.logic.tracks


def test_create_inhand_logic():
    """Test factory function."""
    config = Mock()
    config.min_confidence = 0.6
    config.iou_threshold = 0.4
    
    logic = create_inhand_logic(config)
    
    assert isinstance(logic, InHandLogic)
    assert logic.config == config


# Integration test
def test_full_tracking_cycle():
    """Test complete tracking cycle with multiple frames."""
    config = Mock()
    config.min_confidence = 0.5
    config.iou_threshold = 0.3
    config.matching_strategy = 'center_distance'
    config.match_threshold = 50.0
    config.max_distance = 100.0
    
    logic = InHandLogic(config)
    motion_analyzer = Mock()
    
    # Simulate tracking an object through multiple frames
    positions = [
        ([10, 10, 50, 50], [15, 15, 45, 45]),  # Initial position
        ([12, 12, 52, 52], [17, 17, 47, 47]),  # Slight movement
        ([15, 15, 55, 55], [20, 20, 50, 50]),  # More movement
    ]
    
    track_id = None
    
    for i, (hand_bbox, obj_bbox) in enumerate(positions):
        # Mock motion metrics
        motion_analyzer.analyze_region.return_value = MotionMetrics(
            flow_magnitude=2.0 + i * 0.5,
            centroid_movement=1.0 + i * 0.3,
            area_ratio=1.1,
            bbox_change=0.8 + i * 0.1
        )
        
        # Create hand detection
        hand_landmarks = HandLandmarks(
            landmarks=np.zeros((21, 3)),
            bbox=hand_bbox,
            confidence=0.8,
            handedness='Right',
            grip_strength=0.7
        )
        hand_detection = HandDetection(
            landmarks=[hand_landmarks],
            frame_shape=(480, 640, 3)
        )
        
        object_detections = [obj_bbox + [0.9, 0]]
        
        tracks = logic.process_frame(
            hand_detection, object_detections, motion_analyzer, 100.0 + i
        )
        
        assert len(tracks) == 1
        
        if track_id is None:
            track_id = tracks[0].track_id
        else:
            # Should be same track across frames
            assert tracks[0].track_id == track_id
        
        # Verify motion metrics are updated
        assert tracks[0].motion_metrics.flow_magnitude == 2.0 + i * 0.5
    
    # Verify final state
    assert len(logic.tracks) == 1
    final_track = logic.tracks[track_id]
    assert final_track.object_bbox == [20, 20, 50, 50]
    assert final_track.motion_metrics.flow_magnitude == 3.0
