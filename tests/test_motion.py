"""Test suite for motion analysis module."""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
from src.motion.optical_flow import OpticalFlowAnalyzer, create_optical_flow_analyzer
from src.motion.metrics import MotionMetrics


class TestOpticalFlowAnalyzer:
    """Test cases for OpticalFlowAnalyzer class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Mock configuration
        self.config = Mock()
        self.config.method = 'farneback'
        self.config.scale = 0.5
        self.config.levels = 3
        self.config.winsize = 15
        self.config.iterations = 3
        self.config.poly_n = 5
        self.config.poly_sigma = 1.2
        self.config.use_gpu = False
        
        self.analyzer = OpticalFlowAnalyzer(self.config)
    
    def test_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.config == self.config
        assert self.analyzer.prev_frame is None
        assert not self.analyzer.use_gpu
    
    def test_preprocess_frame(self):
        """Test frame preprocessing."""
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        processed = self.analyzer._preprocess_frame(frame)
        
        # Should be grayscale
        assert len(processed.shape) == 2
        
        # Should be scaled down
        expected_height = int(480 * self.config.scale)
        expected_width = int(640 * self.config.scale)
        assert processed.shape == (expected_height, expected_width)
    
    def test_preprocess_frame_already_gray(self):
        """Test preprocessing already grayscale frame."""
        # Create grayscale frame
        frame = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
        
        processed = self.analyzer._preprocess_frame(frame)
        
        assert len(processed.shape) == 2
        assert processed.shape == frame.shape  # No scaling if already small
    
    @patch('cv2.calcOpticalFlowPyrLK')
    def test_calculate_flow_lk(self, mock_lk):
        """Test Lucas-Kanade optical flow calculation."""
        self.config.method = 'lucas_kanade'
        analyzer = OpticalFlowAnalyzer(self.config)
        
        # Mock frames
        prev_frame = np.zeros((100, 100), dtype=np.uint8)
        curr_frame = np.ones((100, 100), dtype=np.uint8) * 255
        
        # Mock LK results
        mock_lk.return_value = (
            np.array([[10, 20]], dtype=np.float32),  # new points
            np.array([[1]], dtype=np.uint8),         # status
            np.array([[0.5]], dtype=np.float32)      # error
        )
        
        flow = analyzer._calculate_flow(prev_frame, curr_frame)
        
        # Should call LK method
        mock_lk.assert_called_once()
        assert flow is not None
    
    @patch('cv2.calcOpticalFlowPyrLK')
    def test_calculate_flow_lk_no_features(self, mock_lk):
        """Test LK with no good features to track."""
        self.config.method = 'lucas_kanade'
        analyzer = OpticalFlowAnalyzer(self.config)
        
        # Mock no features found
        with patch('cv2.goodFeaturesToTrack', return_value=None):
            prev_frame = np.zeros((100, 100), dtype=np.uint8)
            curr_frame = np.ones((100, 100), dtype=np.uint8) * 255
            
            flow = analyzer._calculate_flow(prev_frame, curr_frame)
            
            # Should return zeros
            assert flow is not None
            assert np.all(flow == 0)
    
    @patch('cv2.calcOpticalFlowFarneback')
    def test_calculate_flow_farneback(self, mock_farneback):
        """Test Farneback optical flow calculation."""
        # Mock flow result
        mock_flow = np.random.randn(100, 100, 2).astype(np.float32)
        mock_farneback.return_value = mock_flow
        
        prev_frame = np.zeros((100, 100), dtype=np.uint8)
        curr_frame = np.ones((100, 100), dtype=np.uint8) * 255
        
        flow = self.analyzer._calculate_flow(prev_frame, curr_frame)
        
        # Should call Farneback method
        mock_farneback.assert_called_once()
        assert np.array_equal(flow, mock_flow)
    
    def test_calculate_flow_dis(self):
        """Test DIS optical flow calculation."""
        self.config.method = 'dis'
        analyzer = OpticalFlowAnalyzer(self.config)
        
        # Create realistic test frames
        prev_frame = np.zeros((100, 100), dtype=np.uint8)
        curr_frame = np.zeros((100, 100), dtype=np.uint8)
        
        # Add a moving square
        prev_frame[40:60, 40:60] = 255
        curr_frame[42:62, 42:62] = 255
        
        flow = analyzer._calculate_flow(prev_frame, curr_frame)
        
        # Should return flow field
        assert flow is not None
        assert flow.shape == (100, 100, 2)
    
    def test_analyze_region_motion_metrics(self):
        """Test motion metrics calculation for a region."""
        # Create synthetic flow field with known motion
        flow = np.zeros((100, 100, 2), dtype=np.float32)
        
        # Add motion in a specific region
        flow[20:40, 30:50, 0] = 5.0  # x-direction
        flow[20:40, 30:50, 1] = 3.0  # y-direction
        
        bbox = [30, 20, 50, 40]  # x1, y1, x2, y2
        prev_bbox = [28, 18, 48, 38]  # Slightly different
        
        metrics = self.analyzer._analyze_region_motion(flow, bbox, prev_bbox)
        
        assert isinstance(metrics, MotionMetrics)
        assert metrics.flow_magnitude > 0
        assert metrics.centroid_movement > 0
        assert metrics.area_ratio != 1.0  # Boxes have different areas
        assert metrics.bbox_change > 0
    
    def test_analyze_region_no_motion(self):
        """Test metrics with no motion."""
        # Zero flow field
        flow = np.zeros((100, 100, 2), dtype=np.float32)
        
        bbox = [30, 20, 50, 40]
        
        metrics = self.analyzer._analyze_region_motion(flow, bbox, bbox)
        
        assert metrics.flow_magnitude == 0.0
        assert metrics.centroid_movement == 0.0
        assert metrics.area_ratio == 1.0
        assert metrics.bbox_change == 0.0
    
    def test_compute_frame_first_frame(self):
        """Test computing flow for first frame."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        flow = self.analyzer.compute_frame(frame)
        
        # First frame should return None
        assert flow is None
        assert self.analyzer.prev_frame is not None
    
    @patch('cv2.calcOpticalFlowFarneback')
    def test_compute_frame_subsequent_frames(self, mock_farneback):
        """Test computing flow for subsequent frames."""
        # Mock flow result
        mock_flow = np.random.randn(240, 320, 2).astype(np.float32)
        mock_farneback.return_value = mock_flow
        
        # First frame
        frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        flow1 = self.analyzer.compute_frame(frame1)
        assert flow1 is None
        
        # Second frame
        frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        flow2 = self.analyzer.compute_frame(frame2)
        
        # Should return flow
        assert flow2 is not None
        mock_farneback.assert_called_once()
    
    def test_analyze_region_with_flow(self):
        """Test region analysis with flow data."""
        # Setup analyzer with flow
        flow = np.random.randn(240, 320, 2).astype(np.float32) * 2.0
        self.analyzer.flow = flow
        
        # Define region in original coordinates
        bbox = [100, 80, 200, 160]  # Will be scaled down
        prev_bbox = [98, 78, 198, 158]
        
        metrics = self.analyzer.analyze_region(bbox, prev_bbox)
        
        assert isinstance(metrics, MotionMetrics)
        assert metrics.flow_magnitude >= 0
        assert metrics.centroid_movement >= 0
        assert metrics.area_ratio > 0
        assert metrics.bbox_change >= 0
    
    def test_analyze_region_no_flow(self):
        """Test region analysis without flow data."""
        # No flow computed yet
        assert self.analyzer.flow is None
        
        bbox = [100, 80, 200, 160]
        prev_bbox = [98, 78, 198, 158]
        
        metrics = self.analyzer.analyze_region(bbox, prev_bbox)
        
        # Should return zero metrics
        assert metrics.flow_magnitude == 0.0
        assert metrics.centroid_movement == 0.0
        assert metrics.area_ratio == 1.0
        assert metrics.bbox_change == 0.0
    
    def test_get_flow_visualization(self):
        """Test flow visualization generation."""
        # Create synthetic flow
        flow = np.zeros((100, 100, 2), dtype=np.float32)
        flow[40:60, 40:60, 0] = 5.0
        flow[40:60, 40:60, 1] = 3.0
        
        self.analyzer.flow = flow
        
        # Create base frame
        frame = np.zeros((200, 200, 3), dtype=np.uint8)  # Larger than flow
        
        vis = self.analyzer.get_flow_visualization(frame)
        
        assert vis is not None
        assert vis.shape == frame.shape
        assert vis.dtype == np.uint8
    
    def test_get_flow_visualization_no_flow(self):
        """Test visualization without flow data."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        
        vis = self.analyzer.get_flow_visualization(frame)
        
        # Should return original frame
        assert np.array_equal(vis, frame)
    
    def test_reset(self):
        """Test analyzer reset."""
        # Setup some state
        self.analyzer.prev_frame = np.zeros((100, 100), dtype=np.uint8)
        self.analyzer.flow = np.zeros((100, 100, 2), dtype=np.float32)
        
        self.analyzer.reset()
        
        assert self.analyzer.prev_frame is None
        assert self.analyzer.flow is None
    
    def test_get_statistics(self):
        """Test statistics collection."""
        # No flow computed
        stats = self.analyzer.get_statistics()
        
        expected_keys = ['has_flow', 'flow_shape', 'avg_magnitude', 'max_magnitude']
        for key in expected_keys:
            assert key in stats
        
        assert not stats['has_flow']
        assert stats['flow_shape'] is None
        
        # With flow
        flow = np.random.randn(100, 100, 2).astype(np.float32)
        self.analyzer.flow = flow
        
        stats = self.analyzer.get_statistics()
        assert stats['has_flow']
        assert stats['flow_shape'] == (100, 100, 2)
        assert stats['avg_magnitude'] >= 0
        assert stats['max_magnitude'] >= 0


class TestMotionMetrics:
    """Test cases for MotionMetrics data structure."""
    
    def test_initialization(self):
        """Test metrics initialization."""
        metrics = MotionMetrics(
            flow_magnitude=2.5,
            centroid_movement=1.0,
            area_ratio=1.1,
            bbox_change=0.8
        )
        
        assert metrics.flow_magnitude == 2.5
        assert metrics.centroid_movement == 1.0
        assert metrics.area_ratio == 1.1
        assert metrics.bbox_change == 0.8
    
    def test_default_values(self):
        """Test default metric values."""
        metrics = MotionMetrics()
        
        assert metrics.flow_magnitude == 0.0
        assert metrics.centroid_movement == 0.0
        assert metrics.area_ratio == 1.0
        assert metrics.bbox_change == 0.0
    
    def test_equality(self):
        """Test metrics equality comparison."""
        metrics1 = MotionMetrics(2.5, 1.0, 1.1, 0.8)
        metrics2 = MotionMetrics(2.5, 1.0, 1.1, 0.8)
        metrics3 = MotionMetrics(3.0, 1.0, 1.1, 0.8)
        
        assert metrics1 == metrics2
        assert metrics1 != metrics3
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = MotionMetrics(2.5, 1.0, 1.1, 0.8)
        
        data = metrics.to_dict()
        
        expected = {
            'flow_magnitude': 2.5,
            'centroid_movement': 1.0,
            'area_ratio': 1.1,
            'bbox_change': 0.8
        }
        
        assert data == expected
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'flow_magnitude': 2.5,
            'centroid_movement': 1.0,
            'area_ratio': 1.1,
            'bbox_change': 0.8
        }
        
        metrics = MotionMetrics.from_dict(data)
        
        assert metrics.flow_magnitude == 2.5
        assert metrics.centroid_movement == 1.0
        assert metrics.area_ratio == 1.1
        assert metrics.bbox_change == 0.8
    
    def test_overall_motion_score(self):
        """Test overall motion score calculation."""
        metrics = MotionMetrics(
            flow_magnitude=2.0,
            centroid_movement=1.5,
            area_ratio=1.2,
            bbox_change=0.8
        )
        
        # Test with default weights
        score = metrics.overall_motion_score()
        assert score > 0
        
        # Test with custom weights
        weights = {
            'flow_magnitude': 0.5,
            'centroid_movement': 0.3,
            'area_ratio': 0.1,
            'bbox_change': 0.1
        }
        
        custom_score = metrics.overall_motion_score(weights)
        assert custom_score > 0
        assert custom_score != score  # Should be different


def test_create_optical_flow_analyzer():
    """Test factory function."""
    config = Mock()
    config.method = 'dis'
    config.scale = 0.75
    
    analyzer = create_optical_flow_analyzer(config)
    
    assert isinstance(analyzer, OpticalFlowAnalyzer)
    assert analyzer.config == config


# Integration test
def test_optical_flow_integration():
    """Test complete optical flow analysis pipeline."""
    config = Mock()
    config.method = 'farneback'
    config.scale = 0.5
    config.levels = 3
    config.winsize = 15
    config.iterations = 3
    config.poly_n = 5
    config.poly_sigma = 1.2
    config.use_gpu = False
    
    analyzer = OpticalFlowAnalyzer(config)
    
    # Create sequence of frames with moving object
    frames = []
    for i in range(3):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Moving square
        x = 20 + i * 5
        y = 30 + i * 3
        frame[y:y+20, x:x+20] = 255
        frames.append(frame)
    
    # Process frames
    flows = []
    for frame in frames:
        flow = analyzer.compute_frame(frame)
        flows.append(flow)
    
    # First frame should return None
    assert flows[0] is None
    
    # Subsequent frames should have flow
    for flow in flows[1:]:
        assert flow is not None
        assert len(flow.shape) == 3
        assert flow.shape[2] == 2  # x, y components
    
    # Analyze motion in moving object region
    bbox = [20, 30, 40, 50]  # Initial position
    prev_bbox = [25, 33, 45, 53]  # Moved position
    
    metrics = analyzer.analyze_region(bbox, prev_bbox)
    
    # Should detect motion
    assert metrics.flow_magnitude > 0
    assert metrics.centroid_movement > 0
    assert metrics.bbox_change > 0
