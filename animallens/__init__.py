"""
AnimalLens - Open Animal Behavior Intelligence Platform.
Vision AI & Behavior Event Engine with optional LLM reasoning.
"""
from animallens.core.config import settings
from animallens.core.events import EventCollector
from animallens.core.exceptions import (
    AnimalLensError,
    SpeciesNotFoundError,
    ModelNotFoundError,
    SourceError,
    PipelineError,
    ReasoningError,
    RegistryError,
)
from animallens.core.schemas import (
    AnalysisResult,
    BehaviorEvent,
    BehaviorInfo,
    BoundingBox,
    Keypoint,
    ModelInfo,
    ReasoningOutput,
    SourceInfo,
    SourceType,
    SubjectInfo,
    TemporalInfo,
    TimelineEntry,
)
from animallens.models.registry import ModelRegistry, model_registry
from animallens.perception.base import (
    BaseBehaviorClassifier,
    BaseDetector,
    BasePoseEstimator,
    BaseTracker,
    DetectionResult,
    FramePerceptionData,
    TrackState,
)
from animallens.perception.buffer import RollingVideoBuffer
from animallens.perception.pipeline import PerceptionPipeline
from animallens.reasoning.base import BaseReasoningProvider, NoOpReasoningProvider
from animallens.reasoning.factory import get_reasoning_provider
from animallens.reasoning.ollama import OllamaClient, OllamaReasoningProvider
from animallens.sdk import AnimalLens
from animallens.sources.base import BaseSource
from animallens.sources.image import ImageSource
from animallens.sources.stream import StreamSource
from animallens.sources.video import VideoSource
from animallens.species.base import (
    BehaviorCategory,
    BehaviorTaxonomy,
    SpeciesAdapter,
    SpeciesConfig,
)
from animallens.species.redclaw.adapter import RedclawAdapter
from animallens.species.registry import SpeciesRegistry, species_registry

__version__ = "0.1.0"

__all__ = [
    "AnimalLens",
    "settings",
    "BehaviorEvent",
    "AnalysisResult",
    "TimelineEntry",
    "SpeciesInfo",
    "SourceInfo",
    "SourceType",
    "SubjectInfo",
    "BoundingBox",
    "Keypoint",
    "BehaviorInfo",
    "TemporalInfo",
    "ModelInfo",
    "ReasoningOutput",
    "EventCollector",
    "PerceptionPipeline",
    "RollingVideoBuffer",
    "BaseDetector",
    "BaseTracker",
    "BasePoseEstimator",
    "BaseBehaviorClassifier",
    "DetectionResult",
    "TrackState",
    "FramePerceptionData",
    "SpeciesAdapter",
    "SpeciesConfig",
    "BehaviorTaxonomy",
    "BehaviorCategory",
    "SpeciesRegistry",
    "species_registry",
    "RedclawAdapter",
    "BaseReasoningProvider",
    "NoOpReasoningProvider",
    "OllamaReasoningProvider",
    "OllamaClient",
    "get_reasoning_provider",
    "ModelRegistry",
    "model_registry",
    "BaseSource",
    "ImageSource",
    "VideoSource",
    "StreamSource",
    "AnimalLensError",
    "SpeciesNotFoundError",
    "ModelNotFoundError",
    "SourceError",
    "PipelineError",
    "ReasoningError",
    "RegistryError",
]
