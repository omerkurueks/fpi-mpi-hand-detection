"""Training script for YOLO models."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Config, ModelConfig


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
        description="Hand Inspection Detection - Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="configs/data.yaml",
        help="Path to data configuration file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="configs/model.yaml",
        help="Path to model configuration file"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        choices=["hand", "object", "both"],
        default="hand",
        help="Training task"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size"
    )
    
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size"
    )
    
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="Initial weights"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device (cpu, cuda, auto)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data loading workers"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Project directory"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        help="Experiment name"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume training from checkpoint"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()


def train_yolo_model(
    data_config: str,
    model_config: str,
    task: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    weights: str,
    device: str,
    workers: int,
    project: str,
    name: Optional[str] = None,
    resume: Optional[str] = None
) -> None:
    """
    Train YOLO model.
    
    Args:
        data_config: Path to data configuration
        model_config: Path to model configuration
        task: Training task (hand, object, both)
        epochs: Number of epochs
        batch_size: Batch size
        img_size: Input image size
        weights: Initial weights
        device: Training device
        workers: Number of workers
        project: Project directory
        name: Experiment name
        resume: Resume from checkpoint
    """
    logger = logging.getLogger(__name__)
    
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("YOLO not available. Install with: pip install ultralytics")
        sys.exit(1)
    
    # Load configurations
    data_config_path = Path(data_config)
    if not data_config_path.exists():
        logger.error(f"Data configuration not found: {data_config}")
        sys.exit(1)
    
    # Create YOLO dataset configuration
    dataset_yaml = create_yolo_dataset_config(data_config_path, task)
    
    # Initialize model
    model = YOLO(weights)
    
    # Generate experiment name if not provided
    if name is None:
        from datetime import datetime
        name = f"{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Training arguments
    train_args = {
        'data': dataset_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'workers': workers,
        'project': project,
        'name': name,
        'verbose': True,
        'save': True,
        'plots': True,
        'val': True
    }
    
    if resume:
        train_args['resume'] = resume
    
    logger.info(f"Starting training: {task} detection")
    logger.info(f"Dataset: {dataset_yaml}")
    logger.info(f"Model: {weights}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
    logger.info(f"Device: {device}")
    
    # Start training
    results = model.train(**train_args)
    
    # Save best model to weights directory
    best_path = Path(project) / name / "weights" / "best.pt"
    if best_path.exists():
        weights_dir = Path("weights")
        weights_dir.mkdir(exist_ok=True)
        
        output_name = f"{task}_best.pt"
        output_path = weights_dir / output_name
        
        import shutil
        shutil.copy2(best_path, output_path)
        logger.info(f"Best model saved to: {output_path}")
    
    logger.info("Training completed successfully")


def create_yolo_dataset_config(data_config_path: Path, task: str) -> str:
    """
    Create YOLO dataset configuration file.
    
    Args:
        data_config_path: Path to data configuration
        task: Training task
        
    Returns:
        Path to created YOLO dataset config
    """
    import yaml
    
    # Load data configuration
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Create YOLO format dataset config
    if task == "hand":
        classes = ["hand"]
        datasets = ["egohands", "custom"]
    elif task == "object":
        classes = ["object"]
        datasets = ["custom", "hoi"]
    else:  # both
        classes = ["hand", "object"]
        datasets = ["egohands", "custom", "hoi"]
    
    # Build paths
    train_paths = []
    val_paths = []
    
    for dataset_name in datasets:
        if dataset_name in data_config.get('datasets', {}):
            dataset_info = data_config['datasets'][dataset_name]
            root = Path(dataset_info['root'])
            
            train_paths.append(str(root / "images" / "train"))
            val_paths.append(str(root / "images" / "val"))
    
    yolo_config = {
        'path': str(Path.cwd()),
        'train': train_paths,
        'val': val_paths,
        'names': {i: name for i, name in enumerate(classes)}
    }
    
    # Save YOLO config
    output_path = Path(f"yolo_dataset_{task}.yaml")
    with open(output_path, 'w') as f:
        yaml.dump(yolo_config, f, default_flow_style=False)
    
    return str(output_path)


def main() -> None:
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Hand Inspection Detection - Training")
    
    try:
        train_yolo_model(
            data_config=args.data,
            model_config=args.model,
            task=args.task,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            weights=args.weights,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            resume=args.resume
        )
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    except Exception as e:
        logger.error(f"Training error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
