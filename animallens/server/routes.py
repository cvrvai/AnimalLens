"""
REST API and WebSocket routes for AnimalLens server.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from animallens.core.config import settings
from animallens.core.schemas import AnalysisResult, BehaviorEvent
from animallens.models.registry import model_registry
from animallens.reasoning.ollama import OllamaClient
from animallens.sdk import AnimalLens
from animallens.server.websocket import ws_manager
from animallens.species.registry import species_registry

router = APIRouter(prefix="/v1", tags=["AnimalLens API v1"])


class AnalyzeImageRequest(BaseModel):
    image_url: Optional[str] = None
    species: str = "redclaw"
    reasoning: Optional[str] = None


class AnalyzeVideoRequest(BaseModel):
    video_url: Optional[str] = None
    species: str = "redclaw"
    reasoning: Optional[str] = None
    sample_fps: Optional[float] = 5.0
    max_duration_seconds: Optional[float] = None


class ModelPullRequest(BaseModel):
    model_name: str


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint returning device backend and installed models."""
    device_info = "cpu"
    cuda_available = False
    try:
        import torch
        if torch.cuda.is_available():
            device_info = f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})"
            cuda_available = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_info = "mps (Apple Silicon)"
    except Exception:
        pass

    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "device": device_info,
        "cuda_available": cuda_available,
        "installed_models": model_registry.list_installed(),
        "available_species": [s["id"] for s in species_registry.list_species()],
    }


@router.get("/species")
async def list_species() -> List[Dict[str, Any]]:
    """List all supported species and their behavior taxonomies."""
    return species_registry.list_species()


@router.get("/species/{species_id}")
async def get_species_details(species_id: str) -> Dict[str, Any]:
    """Get detailed taxonomy and config for a specific species."""
    try:
        adapter = species_registry.get(species_id)
        return {
            "config": adapter.config.model_dump(),
            "taxonomy": adapter.taxonomy.model_dump(),
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/models")
async def list_models() -> Dict[str, Any]:
    """List installed models and available official models."""
    return {
        "installed": model_registry.list_installed(),
        "available": model_registry.list_available(),
    }


@router.post("/models/pull")
async def pull_model(req: ModelPullRequest) -> Dict[str, Any]:
    """Pull a model package from the registry into local cache."""
    try:
        path = model_registry.pull(req.model_name)
        return {
            "status": "success",
            "model_name": req.model_name,
            "installed_path": str(path),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to pull model: {e}")


@router.delete("/models/{model_name}")
async def remove_model(model_name: str) -> Dict[str, Any]:
    """Remove a cached model package."""
    try:
        success = model_registry.remove(model_name)
        return {"status": "removed" if success else "failed", "model_name": model_name}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/ollama/models")
async def list_ollama_models() -> Dict[str, Any]:
    """Discover installed Ollama LLM models on the configured endpoint."""
    client = OllamaClient()
    models = await client.list_models()
    return {
        "ollama_base_url": client.base_url,
        "connected": len(models) > 0,
        "models": models,
    }


@router.post("/analyze/image", response_model=AnalysisResult)
async def analyze_image_endpoint(
    species: str = Form("dog"),
    reasoning: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
) -> AnalysisResult:
    """Analyze an uploaded image for animal behaviors."""
    if not file:
        raise HTTPException(status_code=400, detail="Image file must be uploaded.")

    content = await file.read()
    lens = AnimalLens(species=species, reasoning=reasoning)
    result = lens.analyze_image(content)

    # Broadcast events to connected WebSockets
    for event in result.behaviors:
        asyncio.create_task(
            ws_manager.broadcast_event(
                "behavior.detected",
                {
                    "species": event.species.id,
                    "behavior": event.behavior.label,
                    "confidence": event.behavior.confidence,
                    "event_id": event.event_id,
                },
            )
        )

    return result


@router.post("/analyze/video", response_model=AnalysisResult)
async def analyze_video_endpoint(
    species: str = Form("dog"),
    reasoning: Optional[str] = Form(None),
    sample_fps: float = Form(5.0),
    max_duration_seconds: Optional[float] = Form(None),
    file: Optional[UploadFile] = File(None),
) -> AnalysisResult:
    """Analyze an uploaded video file, generating a behavior timeline."""
    if not file:
        raise HTTPException(status_code=400, detail="Video file must be uploaded.")

    suffix = os.path.splitext(file.filename or "video.mp4")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        lens = AnimalLens(species=species, reasoning=reasoning)
        result = lens.analyze_video(
            tmp_path,
            sample_fps=sample_fps,
            max_duration_seconds=max_duration_seconds,
        )
        return result
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.websocket("/events")
async def websocket_events_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for receiving real-time behavior events."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep-alive receive loop
            data = await websocket.receive_text()
            # Handle client ping
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# MongoDB Storage & Ethological Analytics Endpoints
# ---------------------------------------------------------------------------

@router.get("/storage/status")
async def storage_status() -> Dict[str, Any]:
    """Check MongoDB storage connection status."""
    from animallens.storage import get_storage
    storage = get_storage()
    connected = storage.is_connected()
    return {
        "status": "connected" if connected else "disconnected",
        "database": storage.config.db_name,
        "collections": {
            "events": storage.config.events_collection,
            "sessions": storage.config.sessions_collection,
            "uncertainty": storage.config.uncertainty_collection,
        },
    }


@router.get("/storage/events")
async def query_stored_events(
    species_id: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(50, le=500),
) -> List[Dict[str, Any]]:
    """Query stored BehaviorEvents from MongoDB."""
    from animallens.storage import get_storage
    storage = get_storage()
    return storage.get_events(species_id=species_id, session_id=session_id, category=category, limit=limit)


@router.get("/analytics/transitions")
async def get_transition_matrix_endpoint(
    session_id: Optional[str] = Query(None),
    species_id: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Retrieve Markov behavior state transition matrix computed by MongoDB aggregation."""
    from animallens.storage import get_storage
    storage = get_storage()
    return storage.get_transition_matrix(session_id=session_id, species_id=species_id)


@router.get("/analytics/circadian")
async def get_circadian_budget_endpoint(
    species_id: Optional[str] = Query(None),
) -> List[Dict[str, Any]]:
    """Retrieve 24h circadian activity time budget computed by MongoDB aggregation."""
    from animallens.storage import get_storage
    storage = get_storage()
    return storage.get_circadian_budget(species_id=species_id)


@router.get("/uncertainty")
async def get_uncertainty_queue_endpoint(
    verified: bool = Query(False),
    limit: int = Query(50, le=200),
) -> List[Dict[str, Any]]:
    """Retrieve active learning review items from MongoDB uncertainty queue."""
    from animallens.storage import get_storage
    storage = get_storage()
    return storage.get_uncertainty_queue(verified=verified, limit=limit)


class VerifyUncertaintyRequest(BaseModel):
    verified_label: str
    verified_by: str = "human_expert"


@router.post("/uncertainty/{unc_id}/verify")
async def verify_uncertainty_endpoint(
    unc_id: str,
    req: VerifyUncertaintyRequest,
) -> Dict[str, Any]:
    """Verify an active learning candidate with human biologist label."""
    from animallens.storage import get_storage
    storage = get_storage()
    success = storage.verify_uncertainty(
        unc_id=unc_id,
        verified_label=req.verified_label,
        verified_by=req.verified_by,
    )
    if not success:
        raise HTTPException(status_code=400, detail="Failed to verify uncertainty item.")
    return {"status": "verified", "unc_id": unc_id, "verified_label": req.verified_label}


# ---------------------------------------------------------------------------
# Live RTSP Camera Streaming & Monitoring Endpoints
# ---------------------------------------------------------------------------

class StartStreamRequest(BaseModel):
    camera_id: str
    rtsp_url: str
    species: str = "redclaw"
    save_to_db: bool = True
    target_fps: float = 15.0


@router.post("/stream/start")
async def start_camera_stream(req: StartStreamRequest) -> Dict[str, Any]:
    """Start background real-time RTSP stream analysis and broadcasting worker."""
    from animallens.server.stream_manager import live_stream_manager
    return live_stream_manager.start_stream(
        camera_id=req.camera_id,
        rtsp_url=req.rtsp_url,
        species=req.species,
        save_to_db=req.save_to_db,
        target_fps=req.target_fps,
    )


@router.post("/stream/stop/{camera_id}")
async def stop_camera_stream(camera_id: str) -> Dict[str, Any]:
    """Stop background RTSP camera stream worker."""
    from animallens.server.stream_manager import live_stream_manager
    success = live_stream_manager.stop_stream(camera_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"No active stream found for camera '{camera_id}'")
    return {"status": "stopped", "camera_id": camera_id}


@router.get("/stream/active")
async def list_active_streams() -> Dict[str, Any]:
    """List all currently active camera streams and their real-time latency metrics."""
    from animallens.server.stream_manager import live_stream_manager
    return live_stream_manager.list_streams()


# ---------------------------------------------------------------------------
# Re-Identification (ReID) & Vector Gallery Endpoints
# ---------------------------------------------------------------------------

class RegisterIndividualRequest(BaseModel):
    name: str
    species: str = "dog"
    metadata: Dict[str, Any] = Field(default_factory=dict)


_reid_gallery_instance = None

def _get_gallery():
    global _reid_gallery_instance
    if _reid_gallery_instance is None:
        from animallens.reid import ReIDGallery
        _reid_gallery_instance = ReIDGallery()
    return _reid_gallery_instance


@router.get("/reid/gallery")
async def get_reid_gallery() -> Dict[str, Any]:
    """List all registered animal profiles in the ReID gallery."""
    return _get_gallery().to_dict()


@router.post("/reid/register")
async def register_individual_endpoint(req: RegisterIndividualRequest) -> Dict[str, Any]:
    """Register a named animal in the persistent ReID gallery."""
    import numpy as np
    dummy_vec = np.zeros(512, dtype=np.float32)
    dummy_vec[0] = 1.0
    prof = _get_gallery().register(req.name, dummy_vec, species=req.species, metadata=req.metadata)
    return {
        "status": "registered",
        "name": prof.name,
        "species": prof.species,
        "metadata": prof.metadata,
    }


# ---------------------------------------------------------------------------
# 1-Click Model Training REST Endpoints
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    video_path: Optional[str] = None
    dataset_yaml: Optional[str] = None
    base_model: str = "yolov8s.pt"
    epochs: int = 20
    batch: int = 16
    device: str = "cpu"


@router.post("/train")
async def trigger_training_endpoint(req: TrainRequest) -> Dict[str, Any]:
    """Trigger 1-click model fine-tuning and return training report."""
    from animallens.training import VideoDatasetBuilder, ModelTrainer
    
    if req.video_path:
        builder = VideoDatasetBuilder()
        builder.extract_keyframes(req.video_path, sample_fps=2.0)
        builder.generate_pseudo_labels()
        dataset_yaml = builder.write_yaml_config()
    elif req.dataset_yaml:
        dataset_yaml = req.dataset_yaml
    else:
        raise HTTPException(status_code=400, detail="Provide either video_path or dataset_yaml.")

    trainer = ModelTrainer(base_model=req.base_model)
    report = trainer.train(
        dataset_yaml=dataset_yaml,
        epochs=req.epochs,
        batch=req.batch,
        device=req.device,
    )
    return report.to_dict()


