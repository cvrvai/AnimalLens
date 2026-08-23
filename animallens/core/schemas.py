"""
Standard strongly-typed Pydantic schemas for AnimalLens.
Provides the universal behavior event format, taxonomy entities, and perception data models.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class SourceType(str, Enum):
    """Supported input source types."""
    IMAGE = "image"
    VIDEO = "video"
    WEBCAM = "webcam"
    RTSP = "rtsp"
    CAMERA = "camera"
    STREAM = "stream"


class BoundingBox(BaseModel):
    """Normalized [x_min, y_min, x_max, y_max] or pixel coordinates."""
    x_min: float = Field(..., description="Left coordinate")
    y_min: float = Field(..., description="Top coordinate")
    x_max: float = Field(..., description="Right coordinate")
    y_max: float = Field(..., description="Bottom coordinate")
    is_normalized: bool = Field(default=True, description="True if coordinates are in [0, 1] range")

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height


class Keypoint(BaseModel):
    """Pose estimation landmark / keypoint."""
    name: str
    x: float
    y: float
    confidence: float = 1.0


class SubjectInfo(BaseModel):
    """Tracked animal or subject participating in an event."""
    track_id: int = Field(..., description="Unique track ID across continuous frames")
    animal_id: Optional[str] = Field(default=None, description="Persistent identifier / re-ID tag")
    bbox: Optional[BoundingBox] = Field(default=None, description="Current or anchor bounding box")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence")
    keypoints: Optional[List[Keypoint]] = Field(default=None, description="Pose keypoints if available")
    velocity: Optional[float] = Field(default=None, description="Estimated movement speed (px/s or m/s)")
    orientation_degrees: Optional[float] = Field(default=None, description="Heading orientation angle")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Custom species-specific attributes")


class SpeciesInfo(BaseModel):
    """Species identification and biological taxonomy."""
    id: str = Field(..., description="Species key, e.g. cherax_quadricarinatus or redclaw")
    name: str = Field(..., description="Common name, e.g. Redclaw Crayfish")
    scientific_name: Optional[str] = Field(default=None, description="Scientific binomial name")
    taxonomy_version: str = Field(default="1.0.0", description="Behavior taxonomy version")


class SourceInfo(BaseModel):
    """Information about the media or stream source."""
    type: SourceType = Field(..., description="Source medium")
    uri: Optional[str] = Field(default=None, description="File path, URL, or stream URI")
    camera_id: Optional[str] = Field(default=None, description="Optional camera hardware identifier")
    fps: Optional[float] = Field(default=None, description="Video/stream frame rate")
    resolution: Optional[tuple[int, int]] = Field(default=None, description="(width, height)")


class BehaviorInfo(BaseModel):
    """Behavior classification details."""
    category: str = Field(
        ...,
        description="Top-level taxonomy category, e.g. locomotion, feeding, aggression, reproduction, abnormal_behavior, unknown"
    )
    label: str = Field(
        ...,
        description="Specific behavior label, e.g. mating, foraging, resting, unknown"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score in [0.0, 1.0]"
    )
    secondary_labels: Optional[List[Dict[str, float]]] = Field(
        default=None,
        description="Top alternatives with confidence scores"
    )
    is_uncertain: bool = Field(
        default=False,
        description="Flagged true if confidence falls below certainty threshold for active learning"
    )

    @field_validator("is_uncertain", mode="before")
    @classmethod
    def check_uncertain(cls, v: Any, info: Any) -> bool:
        if isinstance(v, bool):
            return v
        return False


class TemporalInfo(BaseModel):
    """Temporal extent of the detected behavior event."""
    start: float = Field(..., description="Start timestamp in seconds from video start or stream epoch")
    end: float = Field(..., description="End timestamp in seconds")
    duration: float = Field(..., description="Duration in seconds (end - start)")
    frame_start: Optional[int] = Field(default=None, description="Starting frame number")
    frame_end: Optional[int] = Field(default=None, description="Ending frame number")

    @field_validator("duration", mode="before")
    @classmethod
    def compute_duration(cls, v: Any, info: Any) -> float:
        if v is not None:
            return float(v)
        start = info.data.get("start", 0.0)
        end = info.data.get("end", 0.0)
        return max(0.0, round(float(end) - float(start), 3))


class ModelInfo(BaseModel):
    """Perception model metadata used to generate this event."""
    species_model: str = Field(..., description="Name of the species model, e.g. redclaw-behavior-v1")
    version: str = Field(default="1.0.0", description="Model version")
    detector: Optional[str] = Field(default=None, description="Underlying detector type")
    classifier: Optional[str] = Field(default=None, description="Temporal classifier type")


class ReasoningOutput(BaseModel):
    """Output from an optional reasoning engine (Layer B - Ollama / LLM)."""
    provider: str = Field(..., description="Provider identifier, e.g. ollama:gemma3:12b")
    model: str = Field(..., description="Model name")
    summary: str = Field(..., description="Natural language summary of behavior")
    explanation: Optional[str] = Field(default=None, description="Biological explanation of observed action")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
    raw_response: Optional[str] = Field(default=None, description="Raw LLM text response")


class BehaviorEvent(BaseModel):
    """
    Standard strongly typed Animal Behavior Event.
    Universal format for AnimalLens Layer A intelligence, exportable to ERPs, active learning, and Layer B LLMs.
    """
    schema_version: str = Field(default="1.0", description="Schema specification version")
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:10]}", description="Unique event identifier")
    timestamp: float = Field(default_factory=lambda: time.time(), description="Epoch timestamp of detection")

    species: SpeciesInfo
    source: SourceInfo
    subjects: List[SubjectInfo] = Field(default_factory=list)
    behavior: BehaviorInfo
    temporal: TemporalInfo
    model: ModelInfo

    reasoning: Optional[ReasoningOutput] = Field(
        default=None,
        description="Optional Layer B reasoning output (Ollama, etc.)"
    )

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary custom metadata")

    def to_summary_dict(self) -> Dict[str, Any]:
        """Compact representation for quick logging and streaming."""
        return {
            "event_id": self.event_id,
            "species": self.species.name,
            "behavior": f"{self.behavior.category}.{self.behavior.label}",
            "confidence": round(self.behavior.confidence, 3),
            "subjects_count": len(self.subjects),
            "temporal": f"{self.temporal.start:.1f}s - {self.temporal.end:.1f}s ({self.temporal.duration:.1f}s)",
            "reasoning": self.reasoning.summary if self.reasoning else None,
        }


class TimelineEntry(BaseModel):
    """Entry in a video timeline."""
    timestamp_str: str = Field(..., description="Formatted timestamp, e.g. 00:05:22")
    start_seconds: float
    end_seconds: float
    behavior: str
    confidence: float
    event_id: str


class AnalysisResult(BaseModel):
    """Container returned by high-level AnimalLens analyze APIs."""
    source_uri: Optional[str] = None
    species: str
    total_frames_analyzed: int = 0
    duration_seconds: float = 0.0
    behaviors: List[BehaviorEvent] = Field(default_factory=list)
    timeline: List[TimelineEntry] = Field(default_factory=list)
    summary: Optional[str] = None
    reasoning: Optional[ReasoningOutput] = None
    created_at: float = Field(default_factory=lambda: time.time())

    @property
    def events_count(self) -> int:
        return len(self.behaviors)

    def format_timeline_text(self) -> str:
        """Returns human-readable formatted timeline list (e.g. 00:14:01 Aggression)."""
        lines = []
        for entry in self.timeline:
            lines.append(f"{entry.timestamp_str} {entry.behavior.capitalize()} (conf: {entry.confidence:.2f})")
        return "\n".join(lines) if lines else "No events detected."

    def get_transition_matrix(self) -> Any:
        """Compute Markov behavioral state transition probability matrix."""
        from animallens.analytics.transition_matrix import compute_transition_matrix
        return compute_transition_matrix(self.behaviors)

    def get_spatial_metrics(self, arena_area: float = 1.0) -> Any:
        """Compute spatial dispersion, Nearest Neighbor Distance (NND), and crowding intensity."""
        from animallens.analytics.spatial_metrics import compute_spatial_metrics
        all_subjects = []
        for e in self.behaviors:
            all_subjects.extend(e.subjects)
        return compute_spatial_metrics(all_subjects, arena_area=arena_area)

    def get_ethogram_summary(self) -> Any:
        """Compute quantitative time budget and behavioral frequency distribution."""
        from animallens.analytics.sampling_protocols import SamplingProtocols
        return SamplingProtocols.compute_ethogram_time_budget(self.behaviors, total_duration=self.duration_seconds)
