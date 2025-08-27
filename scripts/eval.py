"""Evaluation script for hand inspection detection."""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.io.sink import InspectionEvent


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
        description="Hand Inspection Detection - Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--gt", "--ground-truth",
        type=str,
        required=True,
        help="Path to ground truth events file (JSONL)"
    )
    
    parser.add_argument(
        "--pred", "--predictions",
        type=str,
        required=True,
        help="Path to predicted events file (JSONL)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for evaluation results (JSON)"
    )
    
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Time tolerance for event matching (seconds)"
    )
    
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum event duration to consider (seconds)"
    )
    
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for event overlap"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()


def load_events_from_jsonl(file_path: str) -> List[InspectionEvent]:
    """
    Load events from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of inspection events
    """
    events = []
    
    try:
        import jsonlines
        with jsonlines.open(file_path) as reader:
            for line in reader:
                event = InspectionEvent.from_dict(line)
                events.append(event)
    except ImportError:
        # Fallback to regular JSON lines
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    event = InspectionEvent.from_dict(data)
                    events.append(event)
    
    return events


def compute_temporal_iou(
    event1: InspectionEvent,
    event2: InspectionEvent
) -> float:
    """
    Compute temporal IoU between two events.
    
    Args:
        event1: First event
        event2: Second event
        
    Returns:
        Temporal IoU
    """
    if event1.end_ts is None or event2.end_ts is None:
        return 0.0
    
    # Intersection
    intersection_start = max(event1.start_ts, event2.start_ts)
    intersection_end = min(event1.end_ts, event2.end_ts)
    
    if intersection_end <= intersection_start:
        return 0.0
    
    intersection = intersection_end - intersection_start
    
    # Union
    union_start = min(event1.start_ts, event2.start_ts)
    union_end = max(event1.end_ts, event2.end_ts)
    union = union_end - union_start
    
    if union == 0:
        return 0.0
    
    return intersection / union


def match_events(
    gt_events: List[InspectionEvent],
    pred_events: List[InspectionEvent],
    tolerance: float = 1.0,
    iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match ground truth and predicted events.
    
    Args:
        gt_events: Ground truth events
        pred_events: Predicted events
        tolerance: Time tolerance for matching
        iou_threshold: IoU threshold for matching
        
    Returns:
        Tuple of (matches, unmatched_gt, unmatched_pred)
    """
    matches = []
    unmatched_gt = list(range(len(gt_events)))
    unmatched_pred = list(range(len(pred_events)))
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(gt_events), len(pred_events)))
    
    for i, gt_event in enumerate(gt_events):
        for j, pred_event in enumerate(pred_events):
            # Check track ID match (if available)
            if hasattr(gt_event, 'track_id') and hasattr(pred_event, 'track_id'):
                if gt_event.track_id != pred_event.track_id:
                    continue
            
            # Compute temporal IoU
            iou = compute_temporal_iou(gt_event, pred_event)
            iou_matrix[i, j] = iou
    
    # Find best matches using greedy approach
    while True:
        # Find best remaining match
        best_iou = 0.0
        best_match = None
        
        for i in unmatched_gt:
            for j in unmatched_pred:
                if iou_matrix[i, j] > best_iou and iou_matrix[i, j] >= iou_threshold:
                    best_iou = iou_matrix[i, j]
                    best_match = (i, j)
        
        if best_match is None:
            break
        
        # Add match and remove from unmatched
        matches.append(best_match)
        unmatched_gt.remove(best_match[0])
        unmatched_pred.remove(best_match[1])
    
    return matches, unmatched_gt, unmatched_pred


def compute_duration_metrics(
    gt_events: List[InspectionEvent],
    pred_events: List[InspectionEvent],
    matches: List[Tuple[int, int]]
) -> Dict[str, float]:
    """
    Compute duration-based metrics.
    
    Args:
        gt_events: Ground truth events
        pred_events: Predicted events
        matches: List of event matches
        
    Returns:
        Dictionary of duration metrics
    """
    duration_errors = []
    
    for gt_idx, pred_idx in matches:
        gt_event = gt_events[gt_idx]
        pred_event = pred_events[pred_idx]
        
        if gt_event.duration_s is not None and pred_event.duration_s is not None:
            error = abs(gt_event.duration_s - pred_event.duration_s)
            duration_errors.append(error)
    
    if not duration_errors:
        return {
            'mae': 0.0,
            'rmse': 0.0,
            'mean_relative_error': 0.0
        }
    
    mae = np.mean(duration_errors)
    rmse = np.sqrt(np.mean(np.square(duration_errors)))
    
    # Relative error
    relative_errors = []
    for gt_idx, pred_idx in matches:
        gt_event = gt_events[gt_idx]
        pred_event = pred_events[pred_idx]
        
        if gt_event.duration_s is not None and pred_event.duration_s is not None and gt_event.duration_s > 0:
            rel_error = abs(gt_event.duration_s - pred_event.duration_s) / gt_event.duration_s
            relative_errors.append(rel_error)
    
    mean_relative_error = np.mean(relative_errors) if relative_errors else 0.0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mean_relative_error': mean_relative_error
    }


def evaluate_events(
    gt_events: List[InspectionEvent],
    pred_events: List[InspectionEvent],
    tolerance: float = 1.0,
    iou_threshold: float = 0.5,
    min_duration: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate inspection events.
    
    Args:
        gt_events: Ground truth events
        pred_events: Predicted events
        tolerance: Time tolerance for matching
        iou_threshold: IoU threshold for matching
        min_duration: Minimum duration to consider
        
    Returns:
        Evaluation results
    """
    logger = logging.getLogger(__name__)
    
    # Filter events by minimum duration
    gt_events_filtered = [e for e in gt_events if e.duration_s and e.duration_s >= min_duration]
    pred_events_filtered = [e for e in pred_events if e.duration_s and e.duration_s >= min_duration]
    
    logger.info(f"Ground truth events: {len(gt_events)} -> {len(gt_events_filtered)} (after filtering)")
    logger.info(f"Predicted events: {len(pred_events)} -> {len(pred_events_filtered)} (after filtering)")
    
    # Match events
    matches, unmatched_gt, unmatched_pred = match_events(
        gt_events_filtered, pred_events_filtered, tolerance, iou_threshold
    )
    
    # Compute basic metrics
    tp = len(matches)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Compute duration metrics
    duration_metrics = compute_duration_metrics(gt_events_filtered, pred_events_filtered, matches)
    
    # Compute detailed statistics
    gt_durations = [e.duration_s for e in gt_events_filtered if e.duration_s is not None]
    pred_durations = [e.duration_s for e in pred_events_filtered if e.duration_s is not None]
    
    results = {
        'summary': {
            'gt_events': len(gt_events_filtered),
            'pred_events': len(pred_events_filtered),
            'matches': tp,
            'false_positives': fp,
            'false_negatives': fn
        },
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'duration_mae': duration_metrics['mae'],
            'duration_rmse': duration_metrics['rmse'],
            'duration_relative_error': duration_metrics['mean_relative_error']
        },
        'statistics': {
            'gt_duration_stats': {
                'mean': np.mean(gt_durations) if gt_durations else 0.0,
                'std': np.std(gt_durations) if gt_durations else 0.0,
                'min': np.min(gt_durations) if gt_durations else 0.0,
                'max': np.max(gt_durations) if gt_durations else 0.0
            },
            'pred_duration_stats': {
                'mean': np.mean(pred_durations) if pred_durations else 0.0,
                'std': np.std(pred_durations) if pred_durations else 0.0,
                'min': np.min(pred_durations) if pred_durations else 0.0,
                'max': np.max(pred_durations) if pred_durations else 0.0
            }
        },
        'parameters': {
            'tolerance': tolerance,
            'iou_threshold': iou_threshold,
            'min_duration': min_duration
        }
    }
    
    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Summary
    summary = results['summary']
    print(f"\nSUMMARY:")
    print(f"  Ground Truth Events: {summary['gt_events']}")
    print(f"  Predicted Events: {summary['pred_events']}")
    print(f"  Matches (TP): {summary['matches']}")
    print(f"  False Positives: {summary['false_positives']}")
    print(f"  False Negatives: {summary['false_negatives']}")
    
    # Metrics
    metrics = results['metrics']
    print(f"\nMETRICS:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1-Score: {metrics['f1_score']:.3f}")
    print(f"  Duration MAE: {metrics['duration_mae']:.3f} seconds")
    print(f"  Duration RMSE: {metrics['duration_rmse']:.3f} seconds")
    print(f"  Duration Relative Error: {metrics['duration_relative_error']:.3f}")
    
    # Statistics
    stats = results['statistics']
    print(f"\nDURATION STATISTICS:")
    print(f"  Ground Truth - Mean: {stats['gt_duration_stats']['mean']:.2f}s, "
          f"Std: {stats['gt_duration_stats']['std']:.2f}s")
    print(f"  Predictions - Mean: {stats['pred_duration_stats']['mean']:.2f}s, "
          f"Std: {stats['pred_duration_stats']['std']:.2f}s")


def main() -> None:
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Hand Inspection Detection - Evaluation")
    
    try:
        # Load events
        logger.info(f"Loading ground truth events from: {args.gt}")
        gt_events = load_events_from_jsonl(args.gt)
        
        logger.info(f"Loading predicted events from: {args.pred}")
        pred_events = load_events_from_jsonl(args.pred)
        
        # Evaluate
        logger.info("Computing evaluation metrics...")
        results = evaluate_events(
            gt_events=gt_events,
            pred_events=pred_events,
            tolerance=args.tolerance,
            iou_threshold=args.iou_threshold,
            min_duration=args.min_duration
        )
        
        # Print results
        print_results(results)
        
        # Save results if output specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
