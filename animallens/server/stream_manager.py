"""
Live RTSP Stream Manager for AnimalLens Server.
Manages persistent background stream workers, frame perception,
real-time WebSocket broadcasting, and MongoDB storage persistence.
"""
from __future__ import annotations

import asyncio
from collections import deque
import logging
import time
from typing import Any, Dict, List, Optional

from animallens.core.schemas import BehaviorEvent, SubjectInfo
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

        self.stream_source = StreamSource(
            endpoint=rtsp_url,
            target_fps=target_fps,
            camera_id=camera_id,
        )
        self.target_fps = self.stream_source.target_fps
        self.lens = AnimalLens(species=species)
        self.storage = get_storage() if save_to_db else None

        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.events_detected_count = 0

        # Real-time metrics
        self.processed_frames: int = 0
        self.detected_count: int = 0
        self.density: float = 0.0
        self.activity: float = 0.0
        self.current_fps: float = float(self.target_fps)
        self._frame_timestamps: deque[float] = deque(maxlen=30)
        self._recent_events: deque[BehaviorEvent] = deque(maxlen=100)
        self._start_time: float = time.time()

    async def _run_loop(self) -> None:
        self._running = True
        self._start_time = time.time()
        logger.info(f"Stream worker started for camera {self.camera_id} ({self.rtsp_url})")

        try:
            async for ts, frame in self.stream_source:
                if not self._running:
                    break

                now = time.time()
                self._frame_timestamps.append(now)
                self.processed_frames += 1

                # 1. Instantaneous FPS calculation
                if len(self._frame_timestamps) >= 2:
                    elapsed = self._frame_timestamps[-1] - self._frame_timestamps[0]
                    self.current_fps = round(len(self._frame_timestamps) / max(0.001, elapsed), 2)
                else:
                    self.current_fps = float(self.target_fps)

                src_info = self.stream_source.get_source_info()
                events = self.lens.pipeline.process_frame(frame, timestamp=ts, source_info=src_info)

                # 2. Extract perception tracks / detections
                latest_data = (
                    self.lens.pipeline.buffer._perception_data[-1]
                    if self.lens.pipeline.buffer._perception_data
                    else None
                )

                subjects: List[SubjectInfo] = []
                if latest_data:
                    if latest_data.tracks:
                        subjects = [t.to_subject_info() for t in latest_data.tracks]
                        self.detected_count = len(latest_data.tracks)
                    elif latest_data.detections and latest_data.detections.bboxes:
                        subjects = [
                            SubjectInfo(
                                track_id=i + 1,
                                bbox=b,
                                confidence=latest_data.detections.confidences[i]
                                if i < len(latest_data.detections.confidences)
                                else 0.9,
                            )
                            for i, b in enumerate(latest_data.detections.bboxes)
                        ]
                        self.detected_count = len(latest_data.detections.bboxes)
                    else:
                        self.detected_count = 0
                else:
                    self.detected_count = 0

                # 3. Density calculation via spatial_metrics
                if subjects:
                    try:
                        from animallens.analytics.spatial_metrics import compute_spatial_metrics
                        sm = compute_spatial_metrics(subjects, arena_area=1.0)
                        d = float(sm.crowding_intensity)
                        if d == 0.0 and len(subjects) > 0:
                            # If subjects are dispersed beyond proximity threshold, use cumulative bounding box coverage
                            d = float(sum(s.bbox.area for s in subjects if s.bbox))
                        self.density = round(min(1.0, max(0.0, d)), 4)
                    except Exception as e:
                        logger.warning(f"Failed to compute spatial metrics: {e}")
                else:
                    self.density = 0.0

                # 4. Activity calculation via sampling_protocols
                if events:
                    self.events_detected_count += len(events)
                    self._recent_events.extend(events)

                if self._recent_events:
                    try:
                        from animallens.analytics.sampling_protocols import SamplingProtocols
                        etho = SamplingProtocols.compute_ethogram_time_budget(list(self._recent_events))
                        self.activity = round(float(etho.activity_index), 4)
                    except Exception as e:
                        logger.warning(f"Failed to compute ethogram time budget: {e}")
                elif latest_data and latest_data.tracks:
                    active_tracks = sum(1 for t in latest_data.tracks if getattr(t, "velocity", 0.0) > 0.01)
                    self.activity = round(active_tracks / max(1, len(latest_data.tracks)), 4)
                else:
                    self.activity = 0.0

                # 5. Broadcast behavior events via WebSocket
                for event in events:
                    try:
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
                    except Exception as ws_err:
                        logger.warning(f"Failed to broadcast: {ws_err}")

                    # Persist to MongoDB if configured
                    if self.storage:
                        try:
                            self.storage.save_event(event)
                        except Exception as e:
                            logger.warning(f"Failed to persist live stream event to MongoDB: {e}")

                # 6. Broadcast live stream telemetry via WebSocket
                try:
                    await ws_manager.broadcast_event(
                        "stream.telemetry",
                        {
                            "camera_id": self.camera_id,
                            "endpoint": str(self.stream_source.endpoint),
                            "is_running": self._running,
                            "latency_ms": round(self.stream_source.last_frame_latency_ms, 2),
                            "fps": self.current_fps,
                            "target_fps": self.target_fps,
                            "frames_received": self.stream_source.total_frames_received,
                            "processed_frames": self.processed_frames,
                            "frames_dropped": self.stream_source.total_frames_dropped,
                            "detected_count": self.detected_count,
                            "density": self.density,
                            "activity": self.activity,
                            "species": self.species,
                            "timestamp": ts,
                        },
                    )
                except Exception as ws_err:
                    logger.warning(f"Failed to broadcast: {ws_err}")

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
            "target_fps": self.target_fps,
            "fps": self.current_fps,
            "processed_frames": self.processed_frames,
            "detected_count": self.detected_count,
            "density": self.density,
            "activity": self.activity,
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
