"""
Live RTSP Stream Manager for AnimalLens Server.
Manages persistent background stream workers, frame perception,
real-time WebSocket broadcasting, and MongoDB storage persistence.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional
from animallens.core.schemas import BehaviorEvent
from animallens.sdk import AnimalLens
from animallens.server.websocket import ws_manager
from animallens.sources.stream import StreamSource
from animallens.storage import get_storage

logger = logging.getLogger(__name__)


class StreamWorker:
    """Worker monitoring a single RTSP camera feed in the background."""

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        species: str = "redclaw",
        save_to_db: bool = True,
        target_fps: float = 15.0,
    ) -> None:
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.species = species
        self.save_to_db = save_to_db
        self.target_fps = target_fps

        self.stream_source = StreamSource(
            endpoint=rtsp_url,
            target_fps=target_fps,
            camera_id=camera_id,
        )
        self.lens = AnimalLens(species=species)
        self.storage = get_storage() if save_to_db else None

        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.events_detected_count = 0

    async def _run_loop(self) -> None:
        self._running = True
        logger.info(f"Stream worker started for camera {self.camera_id} ({self.rtsp_url})")

        try:
            async for ts, frame in self.stream_source:
                if not self._running:
                    break

                src_info = self.stream_source.get_source_info()
                events = self.lens.pipeline.process_frame(frame, timestamp=ts, source_info=src_info)

                for event in events:
                    self.events_detected_count += 1

                    # 1. Broadcast via WebSocket
                    await ws_manager.broadcast_event(
                        "behavior.detected",
                        {
                            "camera_id": self.camera_id,
                            "species": event.species.id,
                            "behavior": f"{event.behavior.category}.{event.behavior.label}",
                            "confidence": event.behavior.confidence,
                            "subjects_count": len(event.subjects),
                            "event_id": event.event_id,
                            "timestamp": event.timestamp,
                        },
                    )

                    # 2. Persist to MongoDB
                    if self.storage:
                        try:
                            self.storage.save_event(event)
                        except Exception as e:
                            logger.warning(f"Failed to persist live stream event to MongoDB: {e}")

                await asyncio.sleep(0.001)  # Yield to event loop
        except asyncio.CancelledError:
            logger.info(f"Stream worker cancelled for {self.camera_id}")
        except Exception as e:
            logger.error(f"Stream worker error for {self.camera_id}: {e}")
        finally:
            self._running = False
            self.stream_source.stop()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._run_loop())
        except RuntimeError:
            # Running in non-async environment (e.g. sync test or CLI)
            self._task = None

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self.stream_source.stop()

    def get_status(self) -> Dict[str, Any]:
        metrics = self.stream_source.get_metrics()
        metrics.update({
            "species": self.species,
            "events_detected": self.events_detected_count,
            "save_to_db": self.save_to_db,
        })
        return metrics


class LiveStreamManager:
    """Central registry of active live RTSP streams."""

    def __init__(self) -> None:
        self._workers: Dict[str, StreamWorker] = {}

    def start_stream(
        self,
        camera_id: str,
        rtsp_url: str,
        species: str = "redclaw",
        save_to_db: bool = True,
        target_fps: float = 15.0,
    ) -> Dict[str, Any]:
        if camera_id in self._workers:
            self._workers[camera_id].stop()

        worker = StreamWorker(
            camera_id=camera_id,
            rtsp_url=rtsp_url,
            species=species,
            save_to_db=save_to_db,
            target_fps=target_fps,
        )
        self._workers[camera_id] = worker
        worker.start()
        return {"status": "started", "camera_id": camera_id, "endpoint": rtsp_url}

    def stop_stream(self, camera_id: str) -> bool:
        if camera_id in self._workers:
            self._workers[camera_id].stop()
            del self._workers[camera_id]
            return True
        return False

    def list_streams(self) -> Dict[str, Any]:
        return {
            "active_streams_count": len(self._workers),
            "streams": [w.get_status() for w in self._workers.values()],
        }


live_stream_manager = LiveStreamManager()
