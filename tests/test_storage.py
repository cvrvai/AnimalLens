"""
Unit tests for MongoDB storage client, persistence, querying, and aggregation pipelines.
Uses mongomock for fast, isolated in-memory MongoDB testing.
"""
from datetime import datetime, timezone
import mongomock
import pytest
from animallens.core.schemas import (
    BehaviorEvent,
    BehaviorInfo,
    BoundingBox,
    ModelInfo,
    SourceInfo,
    SourceType,
    SpeciesInfo,
    SubjectInfo,
    TemporalInfo,
)
from animallens.storage.config import MongoConfig
from animallens.storage.mongodb import MongoDBStorage


@pytest.fixture
def mock_storage():
    """Create a MongoDBStorage instance backed by an in-memory mongomock client."""
    mock_client = mongomock.MongoClient()
    config = MongoConfig(db_name="test_animallens")
    storage = MongoDBStorage(config=config, client=mock_client)
    storage._client = mock_client
    return storage


def _sample_event(event_id: str, label: str, category: str, start: float, end: float) -> BehaviorEvent:
    return BehaviorEvent(
        event_id=event_id,
        timestamp=1724400000.0 + start,
        species=SpeciesInfo(id="cherax_quadricarinatus", name="Redclaw Crayfish"),
        source=SourceInfo(type=SourceType.VIDEO, session_id="sess_test_01", camera_id="CAM-01"),
        subjects=[SubjectInfo(track_id=1, bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2))],
        behavior=BehaviorInfo(category=category, label=label, confidence=0.92),
        temporal=TemporalInfo(start=start, end=end, duration=end - start),
        model=ModelInfo(species_model="redclaw-behavior-v1"),
    )


def test_mongodb_save_and_get_events(mock_storage):
    event1 = _sample_event("evt_01", "normal_movement", "locomotion", 0.0, 5.0)
    event2 = _sample_event("evt_02", "foraging", "feeding", 5.0, 10.0)

    # Save events
    mock_storage.save_event(event1)
    mock_storage.save_event(event2)

    # Query events
    results = mock_storage.get_events(species_id="cherax_quadricarinatus")
    assert len(results) == 2
    assert results[0]["event_id"] == "evt_01"
    assert results[1]["event_id"] == "evt_02"


def test_mongodb_save_session(mock_storage):
    sess_id = mock_storage.save_session({
        "session_id": "sess_20260823_01",
        "tank_id": "TANK-01",
        "cohort_id": "COHORT-A",
        "duration_seconds": 3600.0,
    })
    assert sess_id == "sess_20260823_01"
    doc = mock_storage.sessions.find_one({"session_id": "sess_20260823_01"})
    assert doc is not None
    assert doc["tank_id"] == "TANK-01"


def test_mongodb_uncertainty_queue(mock_storage):
    low_conf_event = _sample_event("evt_unc_01", "unknown", "social_interaction", 12.0, 18.0)
    low_conf_event.behavior.confidence = 0.40
    low_conf_event.behavior.is_uncertain = True

    unc_id = mock_storage.save_uncertainty(
        low_conf_event,
        notes="Uncertain grappling encounter",
        keyframe_uri="s3://frames/unc_01.jpg",
    )
    assert unc_id is not None

    # Retrieve unverified queue
    queue = mock_storage.get_uncertainty_queue(verified=False)
    assert len(queue) == 1
    assert queue[0]["notes"] == "Uncertain grappling encounter"

    # Human expert verification
    success = mock_storage.verify_uncertainty(
        unc_id=unc_id,
        verified_label="aggression.threat_display",
        verified_by="expert_ethologist",
    )
    assert success is True

    # Check that it's no longer in unverified queue
    queue_after = mock_storage.get_uncertainty_queue(verified=False)
    assert len(queue_after) == 0


def test_mongodb_transition_matrix_aggregation(mock_storage):
    events = [
        _sample_event("e1", "resting", "resting", 0.0, 5.0),
        _sample_event("e2", "normal_movement", "locomotion", 5.0, 10.0),
        _sample_event("e3", "foraging", "feeding", 10.0, 15.0),
        _sample_event("e4", "normal_movement", "locomotion", 15.0, 20.0),
    ]
    mock_storage.save_events(events)

    res = mock_storage.get_transition_matrix(session_id="sess_test_01")
    assert res["total_transitions"] == 3
    assert "resting" in res["states"]
    assert "foraging" in res["states"]


def test_mongodb_circadian_budget(mock_storage):
    events = [
        _sample_event("c1", "resting", "resting", 0.0, 30.0),
        _sample_event("c2", "foraging", "feeding", 30.0, 45.0),
    ]
    mock_storage.save_events(events)

    circadian = mock_storage.get_circadian_budget(species_id="cherax_quadricarinatus")
    assert len(circadian) >= 2
    categories = [c["category"] for c in circadian]
    assert "resting" in categories
    assert "feeding" in categories
