"""Convert EgoHands dataset to YOLO format."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import cv2
from scipy.io import loadmat


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert EgoHands dataset to YOLO format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--egohands-path",
        type=str,
        required=True,
        help="Path to EgoHands dataset directory"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output directory for YOLO dataset"
    )
    
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Training split ratio"
    )
    
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio"
    )
    
    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Minimum hand area to include"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()


def polygon_to_bbox(polygon: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Convert polygon to bounding box.
    
    Args:
        polygon: Polygon points (N, 2)
        
    Returns:
        Bounding box (x, y, w, h)
    """
    x_min = np.min(polygon[:, 0])
    y_min = np.min(polygon[:, 1])
    x_max = np.max(polygon[:, 0])
    y_max = np.max(polygon[:, 1])
    
    width = x_max - x_min
    height = y_max - y_min
    
    return int(x_min), int(y_min), int(width), int(height)


def load_egohands_annotations(mat_file: Path) -> List[Dict[str, Any]]:
    """
    Load annotations from EgoHands .mat file.
    
    Args:
        mat_file: Path to .mat annotation file
        
    Returns:
        List of frame annotations
    """
    logger = logging.getLogger(__name__)
    
    try:
        mat_data = loadmat(str(mat_file))
        
        # EgoHands specific structure
        polygons = mat_data['polygons'][0]
        
        annotations = []
        
        for frame_idx, frame_polygons in enumerate(polygons):
            frame_hands = []
            
            if frame_polygons.size > 0:
                # Each frame can have multiple hands
                for hand_idx in range(frame_polygons.shape[1]):
                    hand_polygon = frame_polygons[0, hand_idx]
                    
                    if hand_polygon.size > 0:
                        # Convert to numpy array
                        polygon = np.array(hand_polygon)
                        
                        if polygon.shape[0] >= 3:  # Valid polygon
                            # Convert to bounding box
                            bbox = polygon_to_bbox(polygon)
                            
                            frame_hands.append({
                                'bbox': bbox,
                                'polygon': polygon.tolist(),
                                'class': 'hand',
                                'class_id': 0
                            })
            
            annotations.append({
                'frame_idx': frame_idx,
                'hands': frame_hands
            })
        
        logger.info(f"Loaded annotations for {len(annotations)} frames from {mat_file.name}")
        return annotations
        
    except Exception as e:
        logger.error(f"Error loading annotations from {mat_file}: {e}")
        return []


def convert_to_yolo_format(
    bbox: Tuple[int, int, int, int],
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert bounding box to YOLO format.
    
    Args:
        bbox: Bounding box (x, y, w, h)
        img_width: Image width
        img_height: Image height
        
    Returns:
        YOLO format (center_x, center_y, width, height) normalized
    """
    x, y, w, h = bbox
    
    # Convert to center coordinates
    center_x = x + w / 2
    center_y = y + h / 2
    
    # Normalize
    center_x_norm = center_x / img_width
    center_y_norm = center_y / img_height
    width_norm = w / img_width
    height_norm = h / img_height
    
    return center_x_norm, center_y_norm, width_norm, height_norm


def process_video_folder(
    video_folder: Path,
    output_folder: Path,
    split: str,
    min_area: int = 100
) -> int:
    """
    Process single video folder from EgoHands.
    
    Args:
        video_folder: Path to video folder
        output_folder: Output folder for processed data
        split: Dataset split (train/val)
        min_area: Minimum hand area
        
    Returns:
        Number of processed frames
    """
    logger = logging.getLogger(__name__)
    
    # Find annotation file
    mat_files = list(video_folder.glob("*.mat"))
    if not mat_files:
        logger.warning(f"No .mat file found in {video_folder}")
        return 0
    
    mat_file = mat_files[0]
    
    # Load annotations
    annotations = load_egohands_annotations(mat_file)
    if not annotations:
        return 0
    
    # Find images
    image_files = sorted(video_folder.glob("*.jpg"))
    if not image_files:
        logger.warning(f"No image files found in {video_folder}")
        return 0
    
    # Create output directories
    images_dir = output_folder / "images" / split
    labels_dir = output_folder / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    for frame_annotation in annotations:
        frame_idx = frame_annotation['frame_idx']
        hands = frame_annotation['hands']
        
        # Skip if frame index is out of range
        if frame_idx >= len(image_files):
            continue
        
        image_file = image_files[frame_idx]
        
        # Load image to get dimensions
        try:
            img = cv2.imread(str(image_file))
            if img is None:
                continue
            
            img_height, img_width = img.shape[:2]
        except Exception as e:
            logger.warning(f"Error loading image {image_file}: {e}")
            continue
        
        # Filter hands by minimum area
        valid_hands = []
        for hand in hands:
            x, y, w, h = hand['bbox']
            area = w * h
            
            if area >= min_area:
                valid_hands.append(hand)
        
        # Create output filename
        video_name = video_folder.name
        output_name = f"{video_name}_frame_{frame_idx:06d}"
        
        # Copy image
        output_image_path = images_dir / f"{output_name}.jpg"
        import shutil
        shutil.copy2(image_file, output_image_path)
        
        # Create YOLO label file
        output_label_path = labels_dir / f"{output_name}.txt"
        
        with open(output_label_path, 'w') as f:
            for hand in valid_hands:
                bbox = hand['bbox']
                class_id = hand['class_id']
                
                # Convert to YOLO format
                yolo_bbox = convert_to_yolo_format(bbox, img_width, img_height)
                
                # Write YOLO annotation
                f.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} "
                       f"{yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
        
        processed_count += 1
    
    logger.info(f"Processed {processed_count} frames from {video_folder.name}")
    return processed_count


def create_dataset_yaml(output_path: Path, class_names: List[str]) -> None:
    """
    Create YOLO dataset configuration file.
    
    Args:
        output_path: Output dataset path
        class_names: List of class names
    """
    dataset_config = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    config_path = output_path / "dataset.yaml"
    
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)


def split_videos(
    video_folders: List[Path],
    train_split: float,
    val_split: float
) -> Tuple[List[Path], List[Path]]:
    """
    Split video folders into train and validation sets.
    
    Args:
        video_folders: List of video folder paths
        train_split: Training split ratio
        val_split: Validation split ratio
        
    Returns:
        Tuple of (train_folders, val_folders)
    """
    # Shuffle videos for random split
    import random
    random.shuffle(video_folders)
    
    total_videos = len(video_folders)
    train_count = int(total_videos * train_split)
    
    train_folders = video_folders[:train_count]
    val_folders = video_folders[train_count:]
    
    return train_folders, val_folders


def main() -> None:
    """Main conversion function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting EgoHands to YOLO conversion")
    
    try:
        # Validate input path
        egohands_path = Path(args.egohands_path)
        if not egohands_path.exists():
            logger.error(f"EgoHands dataset path not found: {egohands_path}")
            sys.exit(1)
        
        # Find video folders
        video_folders = [d for d in egohands_path.iterdir() if d.is_dir()]
        if not video_folders:
            logger.error(f"No video folders found in {egohands_path}")
            sys.exit(1)
        
        logger.info(f"Found {len(video_folders)} video folders")
        
        # Split into train/val
        train_folders, val_folders = split_videos(
            video_folders, args.train_split, args.val_split
        )
        
        logger.info(f"Split: {len(train_folders)} train, {len(val_folders)} validation")
        
        # Create output directory
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process training videos
        logger.info("Processing training videos...")
        train_count = 0
        for folder in train_folders:
            count = process_video_folder(folder, output_path, "train", args.min_area)
            train_count += count
        
        # Process validation videos
        logger.info("Processing validation videos...")
        val_count = 0
        for folder in val_folders:
            count = process_video_folder(folder, output_path, "val", args.min_area)
            val_count += count
        
        # Create dataset configuration
        create_dataset_yaml(output_path, ["hand"])
        
        logger.info("Conversion completed successfully")
        logger.info(f"Training frames: {train_count}")
        logger.info(f"Validation frames: {val_count}")
        logger.info(f"Total frames: {train_count + val_count}")
        logger.info(f"Output dataset saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
