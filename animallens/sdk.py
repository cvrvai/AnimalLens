"""
High-level developer SDK for AnimalLens.
Provides the primary `AnimalLens` class hiding all underlying CV/tracking/temporal complexity.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union
from PIL import Image
from animallens.core.config import settings
from animallens.core.events import EventCollector
from animallens.core.schemas import (
    AnalysisResult,
    BehaviorEvent,
    ReasoningOutput,
    SourceInfo,
    SourceType,
    TimelineEntry,
)
from animallens.perception.base import BaseDetector, BaseTracker
from animallens.perception.pipeline import PerceptionPipeline
from animallens.reasoning.base import BaseReasoningProvider
from animallens.reasoning.factory import get_reasoning_provider
from animallens.sources.base import BaseSource
from animallens.sources.image import ImageSource
from animallens.sources.stream import StreamSource
from animallens.sources.video import VideoSource
from animallens.species.base import SpeciesAdapter
from animallens.species.registry import species_registry


class AnimalLens:
    """
    Primary developer entry point for AnimalLens.

    Example:
        >>> from animallens import AnimalLens
        >>> lens = AnimalLens(species="redclaw", reasoning="ollama:gemma3")
        >>> result = lens.analyze_video("tank.mp4")
        >>> for event in result.behaviors:
        ...     print(event.behavior.category, event.behavior.label)
    """

    def __init__(
        self,
        species: Union[str, SpeciesAdapter] = "redclaw",
        reasoning: Optional[Union[str, BaseReasoningProvider]] = None,
        model_name: Optional[str] = None,
        detector: Optional[BaseDetector] = None,
        tracker: Optional[BaseTracker] = None,
        ollama_base_url: Optional[str] = None,
        buffer_duration_seconds: float = 15.0,
        storage: Optional[Any] = None,
    ) -> None:
        # Resolve species adapter
        if isinstance(species, SpeciesAdapter):
            self.species_adapter = species
        else:
            self.species_adapter = species_registry.get(species)

        self.species_name = self.species_adapter.config.name
        self.model_name = model_name or self.species_adapter.config.default_model

        # Resolve storage
        self.storage = storage

        # Resolve reasoning provider
        self.reasoning: BaseReasoningProvider = get_reasoning_provider(
            reasoning=reasoning,
            base_url=ollama_base_url,
        )

        # Resolve detector
        resolved_detector = detector
        if isinstance(detector, str):
            if detector.lower() in ("yolo", "yolov8", "yolov8n", "yolov8s"):
                from animallens.perception.models.yolov8_detector import YOLOv8Detector
                resolved_detector = YOLOv8Detector()

        # Resolve tracker
        resolved_tracker = tracker
        if isinstance(tracker, str):
            if tracker.lower() in ("botsort", "bot_sort", "bytetrack", "kalman"):
                from animallens.perception.models.botsort_tracker import BoTSORTTracker
                resolved_tracker = BoTSORTTracker()

        # Build Layer A perception pipeline
        self.pipeline = PerceptionPipeline(
            species_adapter=self.species_adapter,
            detector=resolved_detector,
            tracker=resolved_tracker,
            buffer_duration_seconds=buffer_duration_seconds,
        )

        self.events_collector = EventCollector()
        self._last_result: Optional[AnalysisResult] = None

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS string."""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"

    def analyze(
        self,
        source: Union[str, Path, Image.Image, BaseSource],
        **kwargs: Any,
    ) -> AnalysisResult:
        """
        Polymorphic analysis method. Automatically detects image vs video vs stream.
        """
        if isinstance(source, (Image.Image, bytes)):
            return self.analyze_image(source, **kwargs)

        if isinstance(source, BaseSource):
            if source.source_type == SourceType.IMAGE:
                return self.analyze_image(source, **kwargs)
            return self.analyze_video(source, **kwargs)

        src_str = str(source)
        # Check if URL/RTSP stream
        if src_str.startswith("rtsp://") or src_str.startswith("rtsps://"):
            raise ValueError("For live RTSP streams, please use the `stream()` generator method.")

        # Check file extension
        ext = Path(src_str).suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"):
            return self.analyze_image(source, **kwargs)
        elif ext in (".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv"):
            return self.analyze_video(source, **kwargs)
        else:
            # Default to image if static file, else video
            return self.analyze_image(source, **kwargs)

    def analyze_image(
        self,
        image_input: Union[str, Path, Image.Image, bytes, ImageSource],
        include_reasoning: bool = True,
        **kwargs: Any,
    ) -> AnalysisResult:
        """
        Analyze a single image for animal behavior / posture.
        """
        self.pipeline.reset()
        source = image_input if isinstance(image_input, ImageSource) else ImageSource(image_input)
        src_info = source.get_source_info()

        detected_events: List[BehaviorEvent] = []
        raw_frames = []

        for ts, frame in source:
            raw_frames.append(frame)
            events = self.pipeline.process_frame(frame, timestamp=ts, source_info=src_info)
            detected_events.extend(events)

        # Attach reasoning if provider is enabled
        reasoning_out: Optional[ReasoningOutput] = None
        if include_reasoning and self.reasoning.is_enabled and detected_events:
            # Run async call synchronously for sync SDK method
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        reasoning_out = pool.submit(
                            asyncio.run,
                            self.reasoning.explain_event(detected_events[0], frames=raw_frames)
                        ).result()
                else:
                    reasoning_out = loop.run_until_complete(
                        self.reasoning.explain_event(detected_events[0], frames=raw_frames)
                    )
            except Exception:
                reasoning_out = asyncio.run(
                    self.reasoning.explain_event(detected_events[0], frames=raw_frames)
                )

            for e in detected_events:
                e.reasoning = reasoning_out

        timeline = [
            TimelineEntry(
                timestamp_str=self._format_timestamp(e.temporal.start),
                start_seconds=e.temporal.start,
                end_seconds=e.temporal.end,
                behavior=f"{e.behavior.category}.{e.behavior.label}",
                confidence=e.behavior.confidence,
                event_id=e.event_id,
            )
            for e in detected_events
        ]

        result = AnalysisResult(
            source_uri=source.uri,
            species=self.species_name,
            total_frames_analyzed=1,
            duration_seconds=0.0,
            behaviors=detected_events,
            timeline=timeline,
            summary=reasoning_out.summary if reasoning_out else f"Analyzed image for {self.species_name}.",
            reasoning=reasoning_out,
        )
        self._last_result = result
        return result

    def analyze_video(
        self,
        video_input: Union[str, Path, VideoSource],
        sample_fps: Optional[float] = 5.0,
        max_duration_seconds: Optional[float] = None,
        include_reasoning: bool = True,
        **kwargs: Any,
    ) -> AnalysisResult:
        """
        Analyze a recorded video file, generating a behavior timeline and structured events.
        """
        self.pipeline.reset()
        source = video_input if isinstance(video_input, VideoSource) else VideoSource(
            file_path=video_input,
            sample_fps=sample_fps,
            max_duration_seconds=max_duration_seconds,
        )
        src_info = source.get_source_info()

        detected_events: List[BehaviorEvent] = []
        frames_count = 0
        last_ts = 0.0

        for ts, frame in source:
            frames_count += 1
            last_ts = ts
            events = self.pipeline.process_frame(frame, timestamp=ts, source_info=src_info)
            for e in events:
                # Avoid duplicate events occurring in same temporal window
                if not detected_events or abs(detected_events[-1].temporal.start - e.temporal.start) > 1.0:
                    detected_events.append(e)

        # Aggregate reasoning summary if enabled
        reasoning_out: Optional[ReasoningOutput] = None
        if include_reasoning and self.reasoning.is_enabled and detected_events:
            try:
                reasoning_out = asyncio.run(
                    self.reasoning.summarize_events(detected_events)
                )
            except Exception:
                pass

        timeline = [
            TimelineEntry(
                timestamp_str=self._format_timestamp(e.temporal.start),
                start_seconds=e.temporal.start,
                end_seconds=e.temporal.end,
                behavior=f"{e.behavior.category}.{e.behavior.label}",
                confidence=e.behavior.confidence,
                event_id=e.event_id,
            )
            for e in detected_events
        ]

        result = AnalysisResult(
            source_uri=source.uri,
            species=self.species_name,
            total_frames_analyzed=frames_count,
            duration_seconds=last_ts,
            behaviors=detected_events,
            timeline=timeline,
            summary=reasoning_out.summary if reasoning_out else f"Analyzed {frames_count} frames across {last_ts:.1f}s for {self.species_name}.",
            reasoning=reasoning_out,
        )
        self._last_result = result

        # Auto-persist to MongoDB storage if configured
        if self.storage is not None and detected_events:
            try:
                self.storage.save_events(detected_events)
                session_meta = kwargs.get("session_metadata", {})
                session_meta.update({
                    "species": self.species_name,
                    "source_uri": source.uri,
                    "duration_seconds": last_ts,
                    "events_count": len(detected_events),
                })
                self.storage.save_session(session_meta)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"MongoDB storage save error: {e}")

        return result

    def stream(
        self,
        stream_input: Union[str, int, StreamSource],
        target_fps: float = 15.0,
        camera_id: Optional[str] = None,
    ) -> Iterator[BehaviorEvent]:
        """
        Stream behavior events in real-time from a live RTSP camera or webcam.

        Example:
            >>> for event in lens.stream("rtsp://camera.local/live"):
            ...     print(f"[{event.behavior.category}] {event.behavior.label}")
        """
        source = stream_input if isinstance(stream_input, StreamSource) else StreamSource(
            endpoint=stream_input,
            target_fps=target_fps,
            camera_id=camera_id,
        )
        src_info = source.get_source_info()

        for ts, frame in source:
            events = self.pipeline.process_frame(frame, timestamp=ts, source_info=src_info)
            for e in events:
                self.events_collector.add(e)
                yield e

    async def stream_async(
        self,
        stream_input: Union[str, int, StreamSource],
        target_fps: float = 15.0,
        camera_id: Optional[str] = None,
    ) -> AsyncIterator[BehaviorEvent]:
        """
        Asynchronously yield behavior events from a real-time stream.
        """
        source = stream_input if isinstance(stream_input, StreamSource) else StreamSource(
            endpoint=stream_input,
            target_fps=target_fps,
            camera_id=camera_id,
        )
        src_info = source.get_source_info()

        async for ts, frame in source:
            events = self.pipeline.process_frame(frame, timestamp=ts, source_info=src_info)
            for e in events:
                self.events_collector.add(e)
                yield e

    def ask(self, question: str) -> str:
        """
        Ask a natural language question regarding the most recently analyzed behavior results.
        """
        if not self._last_result or not self._last_result.behaviors:
            return "No recent behavior analysis session available to ask about."

        if not self.reasoning.is_enabled:
            return "Reasoning provider is disabled. Initialize AnimalLens with `reasoning='ollama:<model>'` to enable Q&A."

        return asyncio.run(
            self.reasoning.ask_question(question, self._last_result.behaviors)
        )
