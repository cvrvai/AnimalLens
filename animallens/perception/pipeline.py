"""
Perception Pipeline orchestrating Layer A vision models.
Detector -> Tracker -> Rolling Buffer -> Temporal Classifier -> BehaviorEvent
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional
from animallens.core.config import settings
from animallens.core.schemas import (
    BehaviorEvent,
    BehaviorInfo,
    ModelInfo,
    SourceInfo,
    SourceType,
    SubjectInfo,
    TemporalInfo,
)
from animallens.perception.base import (
    BaseBehaviorClassifier,
    BaseDetector,
    BasePoseEstimator,
    BaseTracker,
    FramePerceptionData,
)
from animallens.perception.buffer import RollingVideoBuffer
from animallens.perception.development.mock_detector import MockDetector
from animallens.perception.development.mock_tracker import MockTracker
from animallens.perception.temporal.classifier import TemporalBehaviorClassifier
from animallens.species.base import SpeciesAdapter


class PerceptionPipeline:
    """
    Modular vision pipeline executing Layer A intelligence.
    100% LLM-independent.
    """

    def __init__(
        self,
        species_adapter: SpeciesAdapter,
        detector: Optional[BaseDetector] = None,
        tracker: Optional[BaseTracker] = None,
        pose_estimator: Optional[BasePoseEstimator] = None,
        classifier: Optional[BaseBehaviorClassifier] = None,
        buffer_duration_seconds: float = 15.0,
        fps: float = 30.0,
    ) -> None:
        self.species_adapter = species_adapter
        self.detector = detector or MockDetector(species_name=species_adapter.config.name)
        self.tracker = tracker or MockTracker()
        self.pose_estimator = pose_estimator
        self.classifier = classifier or TemporalBehaviorClassifier()

        self.buffer = RollingVideoBuffer(
            capacity_seconds=buffer_duration_seconds,
            fps=fps,
        )
        self.fps = fps
        self._frame_count = 0
        self._last_event_time: float = 0.0

    def process_frame(
        self,
        frame: Any,
        timestamp: float,
        source_info: Optional[SourceInfo] = None,
    ) -> List[BehaviorEvent]:
        """
        Process a single image or video frame.
        Returns newly detected BehaviorEvents if a behavior pattern completes.
        """
        self._frame_count += 1
        src = source_info or SourceInfo(type=SourceType.IMAGE)

        # 1. Detection
        detections = self.detector.detect(
            frame,
            confidence_threshold=self.species_adapter.config.detection_threshold,
        )

        # 2. Tracking
        tracks = self.tracker.update(detections, timestamp=timestamp, frame=frame)

        # 3. Pose Estimation (optional)
        if self.pose_estimator:
            tracks = self.pose_estimator.estimate_pose(frame, tracks)

        # 4. Frame data assembly
        frame_data = FramePerceptionData(
            frame_index=self._frame_count,
            timestamp=timestamp,
            detections=detections,
            tracks=tracks,
        )

        # 5. Push to rolling buffer
        self.buffer.push(timestamp=timestamp, frame=frame, perception_data=frame_data)

        # 6. Temporal Behavior Classification
        # For single images (buffer len == 1) or every periodic temporal stride
        recent_window = self.buffer.get_window(duration_seconds=3.0)
        raw_events = self.classifier.classify(recent_window, self.species_adapter)

        events: List[BehaviorEvent] = []
        for raw in raw_events:
            behavior_info = self.species_adapter.classify_behavior(
                label=raw.get("label", "unknown"),
                confidence=raw.get("confidence", 0.5),
            )

            start_t = raw.get("start", timestamp)
            end_t = raw.get("end", timestamp)
            dur = max(0.0, end_t - start_t)

            event = BehaviorEvent(
                species=self.species_adapter.get_species_info(),
                source=src,
                subjects=raw.get("subjects", [t.to_subject_info() for t in tracks]),
                behavior=behavior_info,
                temporal=TemporalInfo(
                    start=start_t,
                    end=end_t,
                    duration=dur,
                    frame_start=max(1, self._frame_count - len(recent_window)),
                    frame_end=self._frame_count,
                ),
                model=ModelInfo(
                    species_model=self.species_adapter.config.default_model,
                    version="1.0.0",
                    detector=self.detector.model_name,
                    classifier=self.classifier.model_name,
                ),
            )
            events.append(event)

        return events

    def reset(self) -> None:
        """Reset pipeline state for a new video or stream."""
        self.tracker.reset()
        self.buffer.clear()
        self._frame_count = 0
        self._last_event_time = 0.0
