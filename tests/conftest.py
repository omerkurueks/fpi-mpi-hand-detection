"""Test configuration for pytest."""

import sys
import os

# Add src directory to Python path for imports
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, os.path.abspath(src_path))

# Mock dependencies that might not be available
import unittest.mock as mock

# Mock heavy dependencies to allow tests to run without them
try:
    import mediapipe
except ImportError:
    sys.modules['mediapipe'] = mock.MagicMock()
    sys.modules['mediapipe.solutions'] = mock.MagicMock()
    sys.modules['mediapipe.solutions.hands'] = mock.MagicMock()

try:
    import ultralytics
except ImportError:
    sys.modules['ultralytics'] = mock.MagicMock()
    sys.modules['ultralytics.YOLO'] = mock.MagicMock()

try:
    import fastapi
except ImportError:
    sys.modules['fastapi'] = mock.MagicMock()
    sys.modules['fastapi.FastAPI'] = mock.MagicMock()
    sys.modules['fastapi.responses'] = mock.MagicMock()
    sys.modules['fastapi.middleware'] = mock.MagicMock()
    sys.modules['fastapi.middleware.cors'] = mock.MagicMock()

try:
    import jsonlines
except ImportError:
    sys.modules['jsonlines'] = mock.MagicMock()

try:
    import torch
except ImportError:
    sys.modules['torch'] = mock.MagicMock()
    sys.modules['torchvision'] = mock.MagicMock()

# Test fixtures and utilities
import pytest
import numpy as np


@pytest.fixture
def sample_frame():
    """Provide a sample frame for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_bbox():
    """Provide a sample bounding box."""
    return [100, 100, 200, 200]  # x1, y1, x2, y2


@pytest.fixture
def sample_hand_landmarks():
    """Provide sample hand landmarks."""
    return np.random.rand(21, 3).astype(np.float32)  # 21 landmarks with x,y,z


# Test markers
pytest_plugins = []

# Configure test discovery
collect_ignore = [
    "setup.py",
    "build",
    "dist",
    ".git",
    "__pycache__",
]
