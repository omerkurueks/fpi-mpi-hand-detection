"""Event logging and data persistence."""

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import jsonlines
from dataclasses import dataclass, asdict
from threading import Lock
import logging

logger = logging.getLogger(__name__)


@dataclass
class InspectionEvent:
    """Data class for inspection events."""
    event_id: str
    track_id: int
    start_ts: float
    end_ts: Optional[float] = None
    duration_s: Optional[float] = None
    avg_flow: Optional[float] = None
    avg_centroid_px: Optional[float] = None
    min_conf: Optional[float] = None
    max_conf: Optional[float] = None
    notes: str = "inspection"
    created_at: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        
        if self.end_ts is not None and self.start_ts is not None:
            self.duration_s = self.end_ts - self.start_ts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InspectionEvent":
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def create_new(
        cls,
        track_id: int,
        start_ts: float,
        notes: str = "inspection"
    ) -> "InspectionEvent":
        """Create new inspection event."""
        return cls(
            event_id=str(uuid.uuid4()),
            track_id=track_id,
            start_ts=start_ts,
            notes=notes
        )
    
    def finish(
        self,
        end_ts: float,
        avg_flow: Optional[float] = None,
        avg_centroid_px: Optional[float] = None,
        min_conf: Optional[float] = None,
        max_conf: Optional[float] = None
    ) -> None:
        """Complete the inspection event."""
        self.end_ts = end_ts
        self.duration_s = end_ts - self.start_ts
        self.avg_flow = avg_flow
        self.avg_centroid_px = avg_centroid_px
        self.min_conf = min_conf
        self.max_conf = max_conf


class EventLogger:
    """Logger for inspection events with multiple output formats."""
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        write_jsonl: bool = True,
        write_csv: bool = True,
        session_id: Optional[str] = None,
        max_files: int = 10
    ):
        """
        Initialize event logger.
        
        Args:
            output_dir: Directory to save event files
            write_jsonl: Enable JSONL output
            write_csv: Enable CSV output
            session_id: Session identifier (auto-generated if None)
            max_files: Maximum number of files to keep per format
        """
        self.output_dir = Path(output_dir)
        self.write_jsonl = write_jsonl
        self.write_csv = write_csv
        self.max_files = max_files
        
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.jsonl_path = self.output_dir / f"{self.session_id}_events.jsonl"
        self.csv_path = self.output_dir / f"{self.session_id}_events.csv"
        
        # Thread safety
        self._lock = Lock()
        
        # Event buffer
        self.events: List[InspectionEvent] = []
        
        # Initialize files
        self._initialize_files()
        
        logger.info(f"Event logger initialized: {self.output_dir}")
        if self.write_jsonl:
            logger.info(f"JSONL output: {self.jsonl_path}")
        if self.write_csv:
            logger.info(f"CSV output: {self.csv_path}")
    
    def _initialize_files(self) -> None:
        """Initialize output files."""
        # Initialize CSV with header
        if self.write_csv and not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'event_id', 'track_id', 'start_ts', 'end_ts', 'duration_s',
                    'avg_flow', 'avg_centroid_px', 'min_conf', 'max_conf',
                    'notes', 'created_at'
                ])
                writer.writeheader()
        
        # Clean up old files
        self._cleanup_old_files()
    
    def _cleanup_old_files(self) -> None:
        """Remove old files if limit exceeded."""
        for pattern in ['*_events.jsonl', '*_events.csv']:
            files = sorted(self.output_dir.glob(pattern))
            if len(files) > self.max_files:
                for old_file in files[:-self.max_files]:
                    try:
                        old_file.unlink()
                        logger.info(f"Removed old file: {old_file}")
                    except Exception as e:
                        logger.warning(f"Failed to remove old file {old_file}: {e}")
    
    def log_event(self, event: InspectionEvent) -> None:
        """
        Log inspection event to configured outputs.
        
        Args:
            event: Inspection event to log
        """
        with self._lock:
            # Add to buffer
            self.events.append(event)
            
            # Write to JSONL
            if self.write_jsonl:
                try:
                    with jsonlines.open(self.jsonl_path, mode='a') as writer:
                        writer.write(event.to_dict())
                except Exception as e:
                    logger.error(f"Failed to write JSONL: {e}")
            
            # Write to CSV
            if self.write_csv:
                try:
                    with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=[
                            'event_id', 'track_id', 'start_ts', 'end_ts', 'duration_s',
                            'avg_flow', 'avg_centroid_px', 'min_conf', 'max_conf',
                            'notes', 'created_at'
                        ])
                        writer.writerow(event.to_dict())
                except Exception as e:
                    logger.error(f"Failed to write CSV: {e}")
            
            logger.debug(f"Logged event: {event.event_id}")
    
    def get_events(
        self,
        since: Optional[float] = None,
        track_id: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[InspectionEvent]:
        """
        Retrieve events with optional filtering.
        
        Args:
            since: Return events after this timestamp
            track_id: Filter by track ID
            limit: Maximum number of events to return
            
        Returns:
            List of inspection events
        """
        with self._lock:
            filtered_events = self.events.copy()
            
            # Filter by timestamp
            if since is not None:
                filtered_events = [e for e in filtered_events if e.start_ts >= since]
            
            # Filter by track ID
            if track_id is not None:
                filtered_events = [e for e in filtered_events if e.track_id == track_id]
            
            # Apply limit
            if limit is not None:
                filtered_events = filtered_events[-limit:]
            
            return filtered_events
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about logged events."""
        with self._lock:
            if not self.events:
                return {
                    'total_events': 0,
                    'unique_tracks': 0,
                    'total_duration': 0.0,
                    'avg_duration': 0.0,
                    'session_id': self.session_id
                }
            
            completed_events = [e for e in self.events if e.duration_s is not None]
            durations = [e.duration_s for e in completed_events]
            
            return {
                'total_events': len(self.events),
                'completed_events': len(completed_events),
                'unique_tracks': len(set(e.track_id for e in self.events)),
                'total_duration': sum(durations) if durations else 0.0,
                'avg_duration': sum(durations) / len(durations) if durations else 0.0,
                'min_duration': min(durations) if durations else 0.0,
                'max_duration': max(durations) if durations else 0.0,
                'session_id': self.session_id
            }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert events to pandas DataFrame."""
        with self._lock:
            if not self.events:
                return pd.DataFrame()
            
            data = [event.to_dict() for event in self.events]
            df = pd.DataFrame(data)
            
            # Convert timestamps to datetime
            df['start_datetime'] = pd.to_datetime(df['start_ts'], unit='s')
            if 'end_ts' in df.columns:
                df['end_datetime'] = pd.to_datetime(df['end_ts'], unit='s')
            
            return df
    
    def export_summary(self, output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Export summary statistics and data.
        
        Args:
            output_path: Optional path to save summary JSON
            
        Returns:
            Summary dictionary
        """
        stats = self.get_statistics()
        
        if output_path:
            summary_path = Path(output_path)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Summary exported to: {summary_path}")
        
        return stats
    
    def close(self) -> None:
        """Close the logger and export final summary."""
        summary = self.export_summary(self.output_dir / f"{self.session_id}_summary.json")
        logger.info(f"Event logger closed. Total events: {summary['total_events']}")


class EventBuffer:
    """Thread-safe buffer for real-time event processing."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize event buffer.
        
        Args:
            max_size: Maximum buffer size
        """
        self.max_size = max_size
        self.buffer: List[InspectionEvent] = []
        self._lock = Lock()
    
    def add(self, event: InspectionEvent) -> None:
        """Add event to buffer."""
        with self._lock:
            self.buffer.append(event)
            
            # Remove oldest if buffer full
            if len(self.buffer) > self.max_size:
                self.buffer.pop(0)
    
    def get_recent(self, count: int = 10) -> List[InspectionEvent]:
        """Get most recent events."""
        with self._lock:
            return self.buffer[-count:] if self.buffer else []
    
    def clear(self) -> None:
        """Clear buffer."""
        with self._lock:
            self.buffer.clear()
    
    def size(self) -> int:
        """Get buffer size."""
        with self._lock:
            return len(self.buffer)
