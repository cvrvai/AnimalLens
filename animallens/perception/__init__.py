"""
Perception pipeline and vision models module for AnimalLens (Layer A).
"""
from animallens.perception.base import (
    BaseDetector,
    BaseTracker,
    BasePoseEstimator,
    BaseBehaviorClassifier,
    DetectionResult,
    TrackState,
    FramePerceptionData,
)
from animallens.perception.buffer import RollingVideoBuffer
from animallens.perception.pipeline import PerceptionPipeline

__all__ = [
    "BaseDetector",
    "BaseTracker",
    "BasePoseEstimator",
    "BaseBehaviorClassifier",
    "DetectionResult",
    "TrackState",
    "FramePerceptionData",
    "RollingVideoBuffer",
    "PerceptionPipeline",
]
