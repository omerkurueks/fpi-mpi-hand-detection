"""Optical flow computation for motion analysis."""

from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class OpticalFlowComputer:
    """Optical flow computation for motion analysis."""
    
    def __init__(
        self,
        method: str = "DIS",
        flow_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize optical flow computer.
        
        Args:
            method: Flow method ('DIS' or 'Farneback')
            flow_params: Parameters for flow computation
        """
        self.method = method.upper()
        self.flow_params = flow_params or {}
        
        # Previous frame for flow computation
        self.prev_gray: Optional[np.ndarray] = None
        
        # Initialize flow computer
        if self.method == "DIS":
            self.flow_computer = cv2.DISOpticalFlow.create()
            self._setup_dis_params()
        elif self.method == "FARNEBACK":
            self._setup_farneback_params()
        else:
            raise ValueError(f"Unsupported flow method: {method}")
        
        logger.info(f"Optical flow initialized: {self.method}")
    
    def _setup_dis_params(self) -> None:
        """Setup DIS optical flow parameters."""
        default_params = {
            'finest_scale': 2,
            'patch_size': 8,
            'patch_stride': 4,
            'grad_descent_iter': 12,
            'variational_refinement_iter': 5
        }
        
        params = {**default_params, **self.flow_params}
        
        # Apply parameters to DIS
        if hasattr(self.flow_computer, 'setFinestScale'):
            self.flow_computer.setFinestScale(params['finest_scale'])
        if hasattr(self.flow_computer, 'setPatchSize'):
            self.flow_computer.setPatchSize(params['patch_size'])
        if hasattr(self.flow_computer, 'setPatchStride'):
            self.flow_computer.setPatchStride(params['patch_stride'])
        if hasattr(self.flow_computer, 'setGradientDescentIterations'):
            self.flow_computer.setGradientDescentIterations(params['grad_descent_iter'])
        if hasattr(self.flow_computer, 'setVariationalRefinementIterations'):
            self.flow_computer.setVariationalRefinementIterations(params['variational_refinement_iter'])
    
    def _setup_farneback_params(self) -> None:
        """Setup Farneback optical flow parameters."""
        self.farneback_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
        
        # Update with user parameters
        self.farneback_params.update(self.flow_params)
    
    def compute_flow(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute optical flow for current frame.
        
        Args:
            frame: Current frame (BGR)
            
        Returns:
            Flow field (H, W, 2) or None if no previous frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
        
        # Compute flow
        try:
            if self.method == "DIS":
                flow = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, None, None)
                if flow is not None and len(flow) >= 2:
                    # DIS returns tuple, we need the flow field
                    flow_field = self.flow_computer.calc(self.prev_gray, gray, None)
                else:
                    flow_field = self.flow_computer.calc(self.prev_gray, gray, None)
            
            elif self.method == "FARNEBACK":
                flow_field = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray,
                    gray,
                    **self.farneback_params
                )
                
                # Farneback might return tuple
                if isinstance(flow_field, tuple):
                    flow_field = flow_field[0] if len(flow_field) > 0 else None
            
            else:
                flow_field = None
            
            # Update previous frame
            self.prev_gray = gray
            
            return flow_field
            
        except Exception as e:
            logger.error(f"Flow computation failed: {e}")
            self.prev_gray = gray
            return None
    
    def compute_flow_magnitude(self, flow: np.ndarray) -> np.ndarray:
        """
        Compute flow magnitude.
        
        Args:
            flow: Flow field (H, W, 2)
            
        Returns:
            Flow magnitude (H, W)
        """
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return magnitude
    
    def compute_flow_direction(self, flow: np.ndarray) -> np.ndarray:
        """
        Compute flow direction in radians.
        
        Args:
            flow: Flow field (H, W, 2)
            
        Returns:
            Flow direction (H, W) in radians
        """
        direction = np.arctan2(flow[..., 1], flow[..., 0])
        return direction
    
    def visualize_flow(
        self,
        image: np.ndarray,
        flow: np.ndarray,
        step: int = 16,
        scale: float = 1.0,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Visualize optical flow as arrows.
        
        Args:
            image: Background image
            flow: Flow field
            step: Step size for flow vectors
            scale: Scale factor for arrows
            color: Arrow color (BGR)
            
        Returns:
            Image with flow visualization
        """
        h, w = flow.shape[:2]
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        
        fx, fy = flow[y, x].T
        
        # Create visualization
        vis = image.copy()
        
        # Draw arrows
        for i in range(len(x)):
            if np.sqrt(fx[i]**2 + fy[i]**2) > 1.0:  # Minimum threshold
                end_x = int(x[i] + fx[i] * scale)
                end_y = int(y[i] + fy[i] * scale)
                
                cv2.arrowedLine(vis, (x[i], y[i]), (end_x, end_y), color, 1, tipLength=0.3)
        
        return vis
    
    def create_flow_heatmap(self, flow: np.ndarray) -> np.ndarray:
        """
        Create flow magnitude heatmap.
        
        Args:
            flow: Flow field
            
        Returns:
            Heatmap image (BGR)
        """
        magnitude = self.compute_flow_magnitude(flow)
        
        # Normalize magnitude
        mag_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(mag_normalized, cv2.COLORMAP_JET)
        
        return heatmap
    
    def reset(self) -> None:
        """Reset flow computer (clear previous frame)."""
        self.prev_gray = None
        logger.debug("Optical flow computer reset")


def compute_roi_flow_statistics(
    flow: np.ndarray,
    roi: Tuple[int, int, int, int]
) -> Dict[str, float]:
    """
    Compute flow statistics within ROI.
    
    Args:
        flow: Flow field (H, W, 2)
        roi: Region of interest (x, y, w, h)
        
    Returns:
        Dictionary with flow statistics
    """
    x, y, w, h = roi
    
    # Extract ROI flow
    roi_flow = flow[y:y+h, x:x+w]
    
    if roi_flow.size == 0:
        return {
            'mean_magnitude': 0.0,
            'max_magnitude': 0.0,
            'mean_x': 0.0,
            'mean_y': 0.0,
            'std_magnitude': 0.0
        }
    
    # Compute magnitude
    magnitude = np.sqrt(roi_flow[..., 0]**2 + roi_flow[..., 1]**2)
    
    # Compute statistics
    stats = {
        'mean_magnitude': float(np.mean(magnitude)),
        'max_magnitude': float(np.max(magnitude)),
        'mean_x': float(np.mean(roi_flow[..., 0])),
        'mean_y': float(np.mean(roi_flow[..., 1])),
        'std_magnitude': float(np.std(magnitude))
    }
    
    return stats


def create_flow_computer(method: str = "DIS", **kwargs) -> OpticalFlowComputer:
    """
    Factory function to create optical flow computer.
    
    Args:
        method: Flow method
        **kwargs: Additional parameters
        
    Returns:
        OpticalFlowComputer instance
    """
    return OpticalFlowComputer(method=method, flow_params=kwargs)
