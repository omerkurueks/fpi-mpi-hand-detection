"""Inference script for hand inspection detection."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.pipeline import create_pipeline
from src.config import load_config


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
        description="Hand Inspection Detection - Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--src", "--source",
        type=str,
        required=True,
        help="Video source (file path, RTSP URL, or camera index)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/logic.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model.yaml",
        help="Path to model configuration file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for events (overrides config)"
    )
    
    parser.add_argument(
        "--save-video",
        type=str,
        help="Save processed video to file"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable video display"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--api",
        action="store_true",
        help="Start API server alongside inference"
    )
    
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="API server port"
    )
    
    return parser.parse_args()


def validate_source(source: str) -> str:
    """Validate and process video source."""
    # Check if it's a camera index
    try:
        camera_index = int(source)
        return camera_index
    except ValueError:
        pass
    
    # Check if it's a file path
    source_path = Path(source)
    if source_path.exists():
        return str(source_path.absolute())
    
    # Assume it's a URL (RTSP, HTTP, etc.)
    if source.startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
        return source
    
    # If none of the above, treat as file path (might not exist yet)
    return source


def main() -> None:
    """Main inference function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Hand Inspection Detection - Inference")
    logger.info(f"Source: {args.src}")
    logger.info(f"Config: {args.config}")
    
    try:
        # Validate source
        video_source = validate_source(args.src)
        
        # Load configuration
        config_path = Path(args.config)
        model_config_path = Path(args.model_config) if args.model_config else None
        
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        
        config, model_config = load_config(config_path, model_config_path)
        
        # Override output directory if specified
        if args.output_dir:
            config.logging.out_dir = args.output_dir
        
        # Disable display if requested
        if args.no_display:
            config.runtime.draw_overlay = False
        
        # Create pipeline
        logger.info("Initializing pipeline...")
        pipeline = create_pipeline(config_path, model_config_path)
        
        # Start API server if requested
        api_server = None
        if args.api:
            from src.api.server import create_api_server
            api_server = create_api_server(pipeline, port=args.api_port)
            logger.info(f"API server starting on port {args.api_port}")
        
        # Start processing
        logger.info("Starting video processing...")
        
        with pipeline:
            if api_server:
                # Run pipeline in separate thread if API is enabled
                import threading
                import uvicorn
                
                def run_pipeline():
                    pipeline.start(video_source)
                
                pipeline_thread = threading.Thread(target=run_pipeline)
                pipeline_thread.daemon = True
                pipeline_thread.start()
                
                # Run API server
                uvicorn.run(api_server, host="0.0.0.0", port=args.api_port)
            else:
                # Run pipeline directly
                pipeline.start(video_source)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        sys.exit(1)
    
    finally:
        logger.info("Inference completed")


if __name__ == "__main__":
    main()
