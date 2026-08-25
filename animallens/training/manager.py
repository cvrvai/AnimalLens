"""
Background Training Job Manager and WebSocket Telemetry Broadcaster for AnimalLens.
Manages asynchronous model fine-tuning runs, tracks live epoch losses, and emits WebSocket events.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from animallens.training.trainer import ModelTrainer, TrainingReport

logger = logging.getLogger(__name__)


@dataclass
class EpochMetric:
    epoch: int
    total_epochs: int
    box_loss: float = 0.0
    cls_loss: float = 0.0
    dfl_loss: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingJob:
    job_id: str
    species: str
    base_model: str
    dataset_yaml: str
    epochs: int
    batch: int
    imgsz: int
    device: str
    experiment_name: str
    status: str = "QUEUED"  # QUEUED, RUNNING, COMPLETED, FAILED
    current_epoch: int = 0
    progress_pct: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    history: List[EpochMetric] = field(default_factory=list)
    report: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "species": self.species,
            "base_model": self.base_model,
            "dataset_yaml": self.dataset_yaml,
            "epochs": self.epochs,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "device": self.device,
            "experiment_name": self.experiment_name,
            "status": self.status,
            "current_epoch": self.current_epoch,
            "progress_pct": round(self.progress_pct, 1),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
            "latest_metrics": self.history[-1].to_dict() if self.history else None,
            "history": [m.to_dict() for m in self.history],
            "report": self.report,
        }


class TrainingJobManager:
    """Singleton manager for managing and monitoring training jobs."""

    def __init__(self, output_base_dir: str | Path = "models/trained") -> None:
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, TrainingJob] = {}
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}

    def create_job(
        self,
        species: str,
        dataset_yaml: str | Path,
        base_model: str = "yolov8s.pt",
        epochs: int = 20,
        batch: int = 16,
        imgsz: int = 640,
        device: str = "cpu",
        experiment_name: Optional[str] = None,
    ) -> TrainingJob:
        job_id = f"train_{uuid.uuid4().hex[:8]}"
        exp_name = experiment_name or f"{species}_{job_id}"
        
        job = TrainingJob(
            job_id=job_id,
            species=species,
            base_model=base_model,
            dataset_yaml=str(dataset_yaml),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            experiment_name=exp_name,
        )
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[Dict[str, Any]]:
        return [job.to_dict() for job in sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)]

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        if job_id not in self._subscribers:
            self._subscribers[job_id] = []
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers[job_id].append(q)
        return q

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        if job_id in self._subscribers and queue in self._subscribers[job_id]:
            self._subscribers[job_id].remove(queue)

    async def broadcast_metric(self, job_id: str, metric: EpochMetric) -> None:
        if job_id in self._subscribers:
            payload = {
                "type": "EPOCH_PROGRESS",
                "job_id": job_id,
                "data": metric.to_dict(),
            }
            for q in self._subscribers[job_id]:
                await q.put(payload)

    async def broadcast_status(self, job_id: str, status: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if job_id in self._subscribers:
            payload = {
                "type": "STATUS_UPDATE",
                "job_id": job_id,
                "status": status,
                "data": extra or {},
            }
            for q in self._subscribers[job_id]:
                await q.put(payload)

    def start_job_async(self, job_id: str) -> None:
        """Launches the training job in a separate daemon thread or asyncio task."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found.")

        loop = asyncio.get_event_loop() if asyncio.get_event_loop().is_running() else asyncio.new_event_loop()
        asyncio.create_task(self._run_job(job))

    async def _run_job(self, job: TrainingJob) -> None:
        job.status = "RUNNING"
        job.started_at = time.time()
        await self.broadcast_status(job.job_id, "RUNNING")

        def epoch_hook(epoch: int, total: int, loss_dict: Dict[str, float], val_metrics: Dict[str, float]):
            metric = EpochMetric(
                epoch=epoch,
                total_epochs=total,
                box_loss=loss_dict.get("box_loss", 0.0),
                cls_loss=loss_dict.get("cls_loss", 0.0),
                dfl_loss=loss_dict.get("dfl_loss", 0.0),
                map50=val_metrics.get("map50", 0.0),
                map50_95=val_metrics.get("map50_95", 0.0),
                precision=val_metrics.get("precision", 0.0),
                recall=val_metrics.get("recall", 0.0),
            )
            job.history.append(metric)
            job.current_epoch = epoch
            job.progress_pct = (epoch / total) * 100.0
            
            # Non-blocking async queue dispatch
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.broadcast_metric(job.job_id, metric))
            except Exception:
                pass

        trainer = ModelTrainer(
            base_model=job.base_model,
            project_dir=self.output_base_dir,
            experiment_name=job.experiment_name,
        )

        try:
            # Run in threadpool to keep asyncio event loop responsive
            loop = asyncio.get_running_loop()
            report: TrainingReport = await loop.run_in_executor(
                None,
                lambda: trainer.train(
                    dataset_yaml=job.dataset_yaml,
                    epochs=job.epochs,
                    batch=job.batch,
                    imgsz=job.imgsz,
                    device=job.device,
                    epoch_callback=epoch_hook,
                )
            )
            job.status = "COMPLETED"
            job.completed_at = time.time()
            job.progress_pct = 100.0
            job.report = report.to_dict()
            await self.broadcast_status(job.job_id, "COMPLETED", job.report)
            logger.info(f"Training job {job.job_id} COMPLETED successfully.")

        except Exception as e:
            job.status = "FAILED"
            job.completed_at = time.time()
            job.error_message = str(e)
            await self.broadcast_status(job.job_id, "FAILED", {"error": str(e)})
            logger.error(f"Training job {job.job_id} FAILED: {e}")


# Global singleton
training_manager = TrainingJobManager()
