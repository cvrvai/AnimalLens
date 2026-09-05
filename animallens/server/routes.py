"""
REST API and WebSocket routes for AnimalLens server.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from animallens.core.config import settings
from animallens.core.exceptions import SourceError
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


def _resolve_param(query_val: Optional[str], form_val: Optional[str], default: Optional[str] = None) -> Optional[str]:
    """Resolve parameter checking query string first, then form-data, falling back to default."""
    if query_val is not None and query_val.strip() != "":
        return query_val.strip()
    if form_val is not None and form_val.strip() != "":
        return form_val.strip()
    return default


@router.post("/analyze/image", response_model=AnalysisResult)
async def analyze_image_endpoint(
    species: Optional[str] = Query(None),
    species_form: Optional[str] = Form(None, alias="species"),
    reasoning: Optional[str] = Query(None),
    reasoning_form: Optional[str] = Form(None, alias="reasoning"),
    file: Optional[UploadFile] = File(None),
) -> AnalysisResult:
    """Analyze an uploaded image for animal behaviors."""
    if not file:
        raise HTTPException(status_code=400, detail="Image file must be uploaded.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image file is empty (0 bytes).")

    target_species = _resolve_param(species, species_form, default="redclaw")
    target_reasoning = _resolve_param(reasoning, reasoning_form, default=None)

    lens = AnimalLens(species=target_species, reasoning=target_reasoning)
    try:
        result = lens.analyze_image(content)
    except HTTPException:
        raise
    except (SourceError, ValueError, Exception) as err:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {err}")

    if target_species.lower() in ("redclaw", "cherax_quadricarinatus"):
        result.species = "redclaw"
    else:
        result.species = lens.species_name

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
    species: Optional[str] = Query(None),
    species_form: Optional[str] = Form(None, alias="species"),
    reasoning: Optional[str] = Query(None),
    reasoning_form: Optional[str] = Form(None, alias="reasoning"),
    sample_fps: float = Form(5.0),
    max_duration_seconds: Optional[float] = Form(None),
    file: Optional[UploadFile] = File(None),
) -> AnalysisResult:
    """Analyze an uploaded video file, generating a behavior timeline."""
    if not file:
        raise HTTPException(status_code=400, detail="Video file must be uploaded.")

    target_species = _resolve_param(species, species_form, default="redclaw")
    target_reasoning = _resolve_param(reasoning, reasoning_form, default=None)

    suffix = os.path.splitext(file.filename or "video.mp4")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        lens = AnimalLens(species=target_species, reasoning=target_reasoning)
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
    """WebSocket endpoint for receiving real-time behavior and telemetry events."""
    from animallens.server.websocket import authenticate_websocket, ws_manager

    # Enforce API Key authentication when ANIMALLENS_API_KEY is configured
    if not await authenticate_websocket(websocket):
        return

    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep-alive receive loop
            data = await websocket.receive_text()
            # Handle client ping
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except (WebSocketDisconnect, Exception) as e:
        logger.debug(f"WebSocket client disconnected or closed: {e}")
    finally:
        ws_manager.disconnect(websocket)


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
# Dataset Management & Ingestion REST Endpoints
# ---------------------------------------------------------------------------

class UpdateAnnotationRequest(BaseModel):
    image_name: str
    split: str = "train"  # train or val
    bboxes: List[Dict[str, float]] = Field(
        default_factory=list,
        description="List of YOLO format dicts: {class_id: 0, x_center: 0.5, y_center: 0.5, width: 0.2, height: 0.2}"
    )


@router.get("/datasets")
async def list_datasets_endpoint() -> List[Dict[str, Any]]:
    """List all available datasets with image counts, partition breakdown, and classes."""
    import yaml
    results = []
    base_dirs = [Path("datasets"), Path("models/trained")]

    for base in base_dirs:
        if not base.exists():
            continue
        for candidate in base.glob("**"):
            yaml_file = candidate / "dataset.yaml"
            if yaml_file.exists() and yaml_file.is_file():
                try:
                    with open(yaml_file, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f) or {}

                    img_train = list((candidate / "images" / "train").glob("*.jpg")) if (candidate / "images" / "train").exists() else []
                    img_val = list((candidate / "images" / "val").glob("*.jpg")) if (candidate / "images" / "val").exists() else []
                    
                    preview_frame = f"/static/{candidate.relative_to(Path('.')).as_posix()}/images/train/{img_train[0].name}" if img_train else None

                    results.append({
                        "id": candidate.name,
                        "name": candidate.name.replace("_", " ").title(),
                        "path": str(candidate),
                        "dataset_yaml": str(yaml_file.resolve()),
                        "total_images": len(img_train) + len(img_val),
                        "train_images": len(img_train),
                        "val_images": len(img_val),
                        "classes": cfg.get("names", {0: "animal"}),
                        "preview_thumbnail": preview_frame,
                    })
                except Exception as e:
                    logger.warning(f"Error parsing dataset at {candidate}: {e}")

    return results


@router.post("/datasets/upload-video")
async def upload_dataset_video_endpoint(
    file: UploadFile = File(...),
    dataset_name: str = Form("custom_dataset"),
    species: str = Form("pig"),
    sample_fps: float = Form(2.0),
    val_split: float = Form(0.2),
    auto_pseudo_label: bool = Form(True),
) -> Dict[str, Any]:
    """Upload raw video file, extract keyframes, auto pseudo-label, and create YOLO dataset."""
    from animallens.training.dataset_builder import VideoDatasetBuilder

    raw_dir = Path("datasets/raw_uploads")
    raw_dir.mkdir(parents=True, exist_ok=True)
    temp_video_path = raw_dir / file.filename

    content = await file.read()
    temp_video_path.write_bytes(content)

    output_dir = Path("datasets") / dataset_name
    builder = VideoDatasetBuilder(output_dir=output_dir, val_split=val_split)
    frames = builder.extract_keyframes(temp_video_path, sample_fps=sample_fps)

    target_classes = ["animal"]
    try:
        adapter = species_registry.get(species)
        target_classes = adapter.config.classes
    except Exception:
        pass

    num_labeled = 0
    if auto_pseudo_label:
        num_labeled = builder.generate_pseudo_labels(classes=target_classes, conf_threshold=0.30)

    dataset_yaml = builder.write_yaml_config(species_name=species, classes=target_classes[:1] if target_classes else ["animal"])

    return {
        "status": "success",
        "dataset_name": dataset_name,
        "species": species,
        "extracted_frames": len(frames),
        "train_frames": len(list(builder.images_train.glob("*.jpg"))),
        "val_frames": len(list(builder.images_val.glob("*.jpg"))),
        "pseudo_labels_generated": num_labeled,
        "dataset_yaml": str(dataset_yaml.resolve()),
    }


@router.get("/datasets/{dataset_name}/frames")
async def get_dataset_frames_endpoint(dataset_name: str) -> Dict[str, Any]:
    """Retrieve keyframes and bounding box annotations for a dataset."""
    output_dir = Path("datasets") / dataset_name
    if not output_dir.exists():
        # Fallback search in models/trained
        output_dir = Path("models/trained") / dataset_name
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found.")

    frames = []
    for split in ["train", "val"]:
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            bboxes = []
            if lbl_path.exists():
                lines = lbl_path.read_text(encoding="utf-8").strip().split("\n")
                for l in lines:
                    parts = l.strip().split()
                    if len(parts) >= 5:
                        try:
                            bboxes.append({
                                "class_id": int(parts[0]),
                                "x_center": float(parts[1]),
                                "y_center": float(parts[2]),
                                "width": float(parts[3]),
                                "height": float(parts[4]),
                            })
                        except ValueError:
                            pass

            frames.append({
                "image_name": img_path.name,
                "split": split,
                "image_url": f"/static/{img_path.relative_to(Path('.')).as_posix()}",
                "bboxes": bboxes,
            })

    return {
        "dataset_name": dataset_name,
        "total_frames": len(frames),
        "frames": frames,
    }


@router.put("/datasets/{dataset_name}/annotations")
async def update_dataset_annotations_endpoint(
    dataset_name: str,
    req: UpdateAnnotationRequest,
) -> Dict[str, Any]:
    """Save updated/verified bounding box annotations for a frame."""
    output_dir = Path("datasets") / dataset_name
    if not output_dir.exists():
        output_dir = Path("models/trained") / dataset_name
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found.")

    lbl_dir = output_dir / "labels" / req.split
    lbl_dir.mkdir(parents=True, exist_ok=True)
    txt_stem = Path(req.image_name).stem
    txt_path = lbl_dir / f"{txt_stem}.txt"

    lines = []
    for b in req.bboxes:
        cls_id = int(b.get("class_id", 0))
        cx = float(b.get("x_center", 0.5))
        cy = float(b.get("y_center", 0.5))
        w = float(b.get("width", 0.2))
        h = float(b.get("height", 0.2))
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return {"status": "saved", "image_name": req.image_name, "box_count": len(lines)}


# ---------------------------------------------------------------------------
# 1-Click Model Training & WebSocket Telemetry Endpoints
# ---------------------------------------------------------------------------

class StartTrainingJobRequest(BaseModel):
    species: str = "pig"
    dataset_yaml: str
    base_model: str = "yolov8s.pt"
    epochs: int = 20
    batch: int = 16
    imgsz: int = 640
    device: str = "cpu"
    experiment_name: Optional[str] = None


@router.post("/train/start")
async def start_training_job_endpoint(req: StartTrainingJobRequest) -> Dict[str, Any]:
    """Spawn an asynchronous model fine-tuning job with real-time WebSocket telemetry."""
    from animallens.training import training_manager

    job = training_manager.create_job(
        species=req.species,
        dataset_yaml=req.dataset_yaml,
        base_model=req.base_model,
        epochs=req.epochs,
        batch=req.batch,
        imgsz=req.imgsz,
        device=req.device,
        experiment_name=req.experiment_name,
    )
    training_manager.start_job_async(job.job_id)
    return job.to_dict()


@router.get("/train/jobs")
async def list_training_jobs_endpoint() -> List[Dict[str, Any]]:
    """List all recent and active model training runs."""
    from animallens.training import training_manager
    return training_manager.list_jobs()


@router.get("/train/status/{job_id}")
async def get_training_job_status_endpoint(job_id: str) -> Dict[str, Any]:
    """Get detailed telemetry, loss history, and checkpoints for a training job."""
    from animallens.training import training_manager
    job = training_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found.")
    return job.to_dict()


@router.websocket("/train/ws/{job_id}")
async def training_telemetry_websocket(websocket: WebSocket, job_id: str) -> None:
    """Real-time WebSocket stream of training epoch loss curves and validation metrics."""
    from animallens.server.websocket import authenticate_websocket

    if not await authenticate_websocket(websocket):
        return

    from animallens.training import training_manager

    job = training_manager.get_job(job_id)
    if not job:
        await websocket.close(code=4004)
        return

    await websocket.accept()
    q = await training_manager.subscribe(job_id)

    # Send initial snapshot
    await websocket.send_json({
        "type": "INITIAL_SNAPSHOT",
        "job_id": job_id,
        "data": job.to_dict(),
    })

    try:
        while True:
            msg = await q.get()
            await websocket.send_json(msg)
            if msg.get("type") == "STATUS_UPDATE" and msg.get("status") in ["COMPLETED", "FAILED"]:
                break
    except WebSocketDisconnect:
        pass
    finally:
        training_manager.unsubscribe(job_id, q)


# Synchronous legacy training endpoint
class TrainRequest(BaseModel):
    video_path: Optional[str] = None
    dataset_yaml: Optional[str] = None
    base_model: str = "yolov8s.pt"
    epochs: int = 20
    batch: int = 16
    device: str = "cpu"


@router.post("/train")
async def trigger_training_endpoint(req: TrainRequest) -> Dict[str, Any]:
    """Trigger synchronous model fine-tuning and return training report."""
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



