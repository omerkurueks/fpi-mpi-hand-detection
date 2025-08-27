"""FastAPI server for hand inspection detection."""

import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    """Request model for inference."""
    source: str
    config_overrides: Optional[Dict[str, Any]] = None
    output_dir: Optional[str] = None
    save_video: Optional[str] = None


class InferenceResponse(BaseModel):
    """Response model for inference."""
    stream_id: str
    status: str
    message: str


class EventsResponse(BaseModel):
    """Response model for events."""
    events: List[Dict[str, Any]]
    total_count: int
    since: Optional[float] = None


class StatisticsResponse(BaseModel):
    """Response model for statistics."""
    active_tracks: int
    completed_events: int
    current_events: int
    total_duration: float
    avg_duration: float
    frame_count: int
    fps: float
    runtime: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    pipeline_running: bool
    components: Dict[str, bool]


def create_api_server(pipeline=None, port: int = 8000) -> FastAPI:
    """
    Create FastAPI server for hand inspection detection.
    
    Args:
        pipeline: Pipeline instance (optional)
        port: Server port
        
    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="Hand Inspection Detection API",
        description="API for real-time hand inspection detection system",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Store pipeline reference
    app.state.pipeline = pipeline
    app.state.inference_tasks = {}
    
    @app.get("/", response_class=JSONResponse)
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Hand Inspection Detection API",
            "version": "1.0.0",
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "health": "/health",
                "inference": "/infer",
                "events": "/events",
                "statistics": "/stats",
                "documentation": "/docs"
            }
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        pipeline = app.state.pipeline
        
        components = {
            "pipeline": pipeline is not None,
            "hand_detector": False,
            "flow_computer": False,
            "motion_tracker": False,
            "fsm": False,
            "event_logger": False
        }
        
        if pipeline:
            components.update({
                "hand_detector": pipeline.hand_detector is not None,
                "flow_computer": pipeline.flow_computer is not None,
                "motion_tracker": pipeline.motion_tracker is not None,
                "fsm": pipeline.fsm is not None,
                "event_logger": pipeline.event_logger is not None
            })
        
        return HealthResponse(
            status="healthy" if all(components.values()) else "degraded",
            timestamp=datetime.now().isoformat(),
            pipeline_running=pipeline.is_running if pipeline else False,
            components=components
        )
    
    @app.post("/infer", response_model=InferenceResponse)
    async def start_inference(
        request: InferenceRequest,
        background_tasks: BackgroundTasks
    ):
        """Start inference on video source."""
        if not app.state.pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not available")
        
        # Generate stream ID
        stream_id = f"stream_{int(time.time() * 1000)}"
        
        # Validate source
        try:
            # Basic validation - could be enhanced
            if not request.source:
                raise ValueError("Source cannot be empty")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid source: {e}")
        
        # Start inference as background task
        def run_inference():
            try:
                pipeline = app.state.pipeline
                
                # Apply config overrides if provided
                if request.config_overrides:
                    # Apply overrides to pipeline config
                    # This would need proper implementation
                    pass
                
                # Override output directory if specified
                if request.output_dir and pipeline.event_logger:
                    pipeline.event_logger.output_dir = request.output_dir
                
                # Start processing
                pipeline.start(request.source)
                
            except Exception as e:
                logger.error(f"Inference error for stream {stream_id}: {e}")
            finally:
                # Clean up
                app.state.inference_tasks.pop(stream_id, None)
        
        # Add to background tasks
        background_tasks.add_task(run_inference)
        app.state.inference_tasks[stream_id] = {
            "start_time": time.time(),
            "source": request.source,
            "status": "running"
        }
        
        return InferenceResponse(
            stream_id=stream_id,
            status="started",
            message=f"Inference started for source: {request.source}"
        )
    
    @app.get("/events", response_model=EventsResponse)
    async def get_events(
        since: Optional[float] = None,
        track_id: Optional[int] = None,
        limit: Optional[int] = 100
    ):
        """Get inspection events."""
        pipeline = app.state.pipeline
        
        if not pipeline or not pipeline.event_logger:
            raise HTTPException(status_code=503, detail="Event logger not available")
        
        try:
            # Get events from event logger
            events = pipeline.event_logger.get_events(
                since=since,
                track_id=track_id,
                limit=limit
            )
            
            # Convert to dictionaries
            event_dicts = [event.to_dict() for event in events]
            
            return EventsResponse(
                events=event_dicts,
                total_count=len(event_dicts),
                since=since
            )
            
        except Exception as e:
            logger.error(f"Error retrieving events: {e}")
            raise HTTPException(status_code=500, detail="Error retrieving events")
    
    @app.get("/events/export")
    async def export_events(format: str = "json"):
        """Export events in specified format."""
        pipeline = app.state.pipeline
        
        if not pipeline or not pipeline.event_logger:
            raise HTTPException(status_code=503, detail="Event logger not available")
        
        try:
            if format.lower() == "csv":
                # Export as CSV
                df = pipeline.event_logger.to_dataframe()
                csv_data = df.to_csv(index=False)
                
                return StreamingResponse(
                    iter([csv_data]),
                    media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=events.csv"}
                )
            
            elif format.lower() == "json":
                # Export as JSON
                events = pipeline.event_logger.get_events()
                event_dicts = [event.to_dict() for event in events]
                
                return JSONResponse(content=event_dicts)
            
            else:
                raise HTTPException(status_code=400, detail="Unsupported format")
                
        except Exception as e:
            logger.error(f"Error exporting events: {e}")
            raise HTTPException(status_code=500, detail="Error exporting events")
    
    @app.get("/stats", response_model=StatisticsResponse)
    async def get_statistics():
        """Get pipeline statistics."""
        pipeline = app.state.pipeline
        
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not available")
        
        try:
            stats = pipeline.get_statistics()
            
            return StatisticsResponse(
                active_tracks=stats.get('active_tracks', 0),
                completed_events=stats.get('completed_events', 0),
                current_events=stats.get('current_events', 0),
                total_duration=stats.get('total_duration', 0.0),
                avg_duration=stats.get('avg_duration', 0.0),
                frame_count=stats.get('frame_count', 0),
                fps=stats.get('fps', 0.0),
                runtime=stats.get('runtime', 0.0)
            )
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            raise HTTPException(status_code=500, detail="Error getting statistics")
    
    @app.post("/control/stop")
    async def stop_inference():
        """Stop current inference."""
        pipeline = app.state.pipeline
        
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not available")
        
        try:
            pipeline.stop()
            
            return JSONResponse(content={
                "status": "stopped",
                "message": "Inference stopped successfully"
            })
            
        except Exception as e:
            logger.error(f"Error stopping inference: {e}")
            raise HTTPException(status_code=500, detail="Error stopping inference")
    
    @app.post("/control/reset")
    async def reset_tracking():
        """Reset tracking state."""
        pipeline = app.state.pipeline
        
        if not pipeline:
            raise HTTPException(status_code=503, detail="Pipeline not available")
        
        try:
            pipeline._reset_tracking()
            
            return JSONResponse(content={
                "status": "reset",
                "message": "Tracking state reset successfully"
            })
            
        except Exception as e:
            logger.error(f"Error resetting tracking: {e}")
            raise HTTPException(status_code=500, detail="Error resetting tracking")
    
    @app.get("/streams")
    async def get_active_streams():
        """Get information about active inference streams."""
        return JSONResponse(content={
            "active_streams": app.state.inference_tasks,
            "count": len(app.state.inference_tasks)
        })
    
    # Add CORS middleware if needed
    from fastapi.middleware.cors import CORSMiddleware
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure as needed for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    logger.info(f"API server created on port {port}")
    
    return app


def run_server(pipeline=None, host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Run the API server.
    
    Args:
        pipeline: Pipeline instance
        host: Server host
        port: Server port
    """
    app = create_api_server(pipeline, port)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    # Run server standalone for testing
    run_server()
