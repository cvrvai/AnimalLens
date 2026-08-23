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
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
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
    species: str = Form("redclaw"),
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
    species: str = Form("redclaw"),
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


@router.get("/events/sse")
async def sse_events_endpoint():
    """Server-Sent Events (SSE) stream endpoint for behavior events."""
    async def event_generator():
        # Heartbeat SSE stream
        yield f"data: {json.dumps({'type': 'stream.connected'})}\n\n"
        while True:
            await asyncio.sleep(5.0)
            yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
