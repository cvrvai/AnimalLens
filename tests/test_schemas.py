"""
Unit tests for AnimalLens Pydantic schemas and serialization.
"""
import json
import pytest
from animallens.core.schemas import (
    AnalysisResult,
    BehaviorEvent,
    BehaviorInfo,
    BoundingBox,
    ModelInfo,
    ReasoningOutput,
    SourceInfo,
    SourceType,
    SpeciesInfo,
    SubjectInfo,
    TemporalInfo,
)


def test_bounding_box_calculations():
    box = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.8)
    assert round(box.width, 2) == 0.40
    assert round(box.height, 2) == 0.60
    assert box.center == (0.3, 0.5)
    assert round(box.area, 4) == 0.24


def test_behavior_event_standard_schema():
    event = BehaviorEvent(
        event_id="evt_01928",
        species=SpeciesInfo(id="cherax_quadricarinatus", name="Redclaw Crayfish"),
        source=SourceInfo(type=SourceType.CAMERA, camera_id="CAM-001"),
        subjects=[
            SubjectInfo(track_id=17, animal_id="F-003"),
            SubjectInfo(track_id=23, animal_id="M-002"),
        ],
        behavior=BehaviorInfo(
            category="reproduction",
            label="mating",
            confidence=0.93,
        ),
        temporal=TemporalInfo(
            start=42.1,
            end=74.4,
            duration=32.3,
        ),
        model=ModelInfo(
            species_model="redclaw-behavior-v1",
            version="1.0.0",
        ),
    )

    data = event.model_dump(mode="json")
    assert data["schema_version"] == "1.0"
    assert data["event_id"] == "evt_01928"
    assert data["species"]["id"] == "cherax_quadricarinatus"
    assert data["behavior"]["label"] == "mating"
    assert data["behavior"]["confidence"] == 0.93
    assert len(data["subjects"]) == 2
    assert data["temporal"]["duration"] == 32.3


def test_unknown_behavior_and_uncertainty():
    event = BehaviorEvent(
        species=SpeciesInfo(id="cherax_quadricarinatus", name="Redclaw Crayfish"),
        source=SourceInfo(type=SourceType.IMAGE),
        behavior=BehaviorInfo(
            category="unknown",
            label="unknown",
            confidence=0.38,
            is_uncertain=True,
        ),
        temporal=TemporalInfo(start=0.0, end=1.0, duration=1.0),
        model=ModelInfo(species_model="redclaw-behavior-v1"),
    )
    assert event.behavior.label == "unknown"
    assert event.behavior.is_uncertain is True


def test_analysis_result_timeline_formatting():
    res = AnalysisResult(
        species="Redclaw Crayfish",
        total_frames_analyzed=100,
        duration_seconds=10.0,
    )
    assert "No events detected" in res.format_timeline_text()
