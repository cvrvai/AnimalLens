"""
Base interfaces and data structures for perception models (Layer A).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from animallens.core.schemas import BoundingBox, Keypoint, SubjectInfo


class DetectionResult(BaseModel):
    """Output from object detector on a single frame."""
    bboxes: List[BoundingBox] = Field(default_factory=list)
    confidences: List[float] = Field(default_factory=list)
    class_names: List[str] = Field(default_factory=list)
    features: Optional[List[List[float]]] = None

    def __len__(self) -> int:
        return len(self.bboxes)


class TrackState(BaseModel):
    """Continuous temporal tracking state for a single animal subject."""
    track_id: int
    animal_id: Optional[str] = None
    current_bbox: BoundingBox
    history_bboxes: List[Tuple[float, BoundingBox]] = Field(default_factory=list)  # (timestamp, bbox)
    velocity: float = 0.0  # Normalized units per second
    confidence: float = 1.0
    keypoints: List[Keypoint] = Field(default_factory=list)
    age_frames: int = 1
    missed_frames: int = 0
    attributes: Dict[str, Any] = Field(default_factory=dict)

    def to_subject_info(self) -> SubjectInfo:
        return SubjectInfo(
            track_id=self.track_id,
            animal_id=self.animal_id,
            bbox=self.current_bbox,
            confidence=self.confidence,
            keypoints=self.keypoints if self.keypoints else None,
            velocity=round(self.velocity, 3),
            attributes=self.attributes,
        )


class FramePerceptionData(BaseModel):
    """Aggregate perception outputs for a single video frame."""
    frame_index: int
    timestamp: float
    detections: DetectionResult
    tracks: List[TrackState] = Field(default_factory=list)
    movement_features: Dict[str, Any] = Field(default_factory=dict)


class BaseDetector(ABC):
    """Interface for animal detection models (e.g. YOLO, RT-DETR, Mask R-CNN)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of detection model."""
        pass

    @abstractmethod
    def detect(self, frame: Any, confidence_threshold: float = 0.45) -> DetectionResult:
        """Run object detection on an image or video frame."""
        pass


class BaseTracker(ABC):
    """Interface for multi-object tracking (e.g. ByteTrack, BoT-SORT, DeepSORT)."""

    @abstractmethod
    def update(
        self,
        detections: DetectionResult,
        timestamp: float,
        frame: Optional[Any] = None,
    ) -> List[TrackState]:
        """Update active tracks given new frame detections."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal tracking state."""
        pass


class BasePoseEstimator(ABC):
    """Interface for animal pose / landmark estimation."""

    @abstractmethod
    def estimate_pose(
        self,
        frame: Any,
        tracks: List[TrackState],
    ) -> List[TrackState]:
        """Attach estimated keypoints to active track states."""
        pass


class BaseBehaviorClassifier(ABC):
    """Interface for temporal behavior classification models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of behavior classifier."""
        pass

    @abstractmethod
    def classify(
        self,
        recent_frames_data: List[FramePerceptionData],
        species_adapter: Any,
    ) -> List[Dict[str, Any]]:
        """
        Classify behaviors occurring within a temporal window.
        Returns list of behavior event candidate dicts.
        """
        pass
