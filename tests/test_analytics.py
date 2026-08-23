"""
Unit tests for Operations Research & Quantitative Ethology analytics module.
"""
import pytest
from animallens.analytics.sampling_protocols import SamplingProtocols
from animallens.analytics.spatial_metrics import compute_spatial_metrics
from animallens.analytics.transition_matrix import compute_transition_matrix
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


def _make_event(category: str, label: str, start: float, end: float, track_ids: list[int] = [1]) -> BehaviorEvent:
    subjects = [
        SubjectInfo(
            track_id=tid,
            bbox=BoundingBox(x_min=0.1 * tid, y_min=0.1 * tid, x_max=0.1 * tid + 0.1, y_max=0.1 * tid + 0.1),
        )
        for tid in track_ids
    ]
    return BehaviorEvent(
        species=SpeciesInfo(id="cherax_quadricarinatus", name="Redclaw Crayfish"),
        source=SourceInfo(type=SourceType.VIDEO),
        subjects=subjects,
        behavior=BehaviorInfo(category=category, label=label, confidence=0.90),
        temporal=TemporalInfo(start=start, end=end, duration=end - start),
        model=ModelInfo(species_model="redclaw-behavior-v1"),
    )


def test_transition_matrix():
    events = [
        _make_event("resting", "resting", 0.0, 5.0),
        _make_event("locomotion", "normal_movement", 5.0, 10.0),
        _make_event("feeding", "foraging", 10.0, 15.0),
        _make_event("locomotion", "normal_movement", 15.0, 20.0),
        _make_event("resting", "resting", 20.0, 25.0),
    ]

    res = compute_transition_matrix(events)
    assert len(res.states) >= 3
    assert res.total_transitions == 4
    dict_matrix = res.to_dict_matrix()
    assert "resting.resting" in dict_matrix
    assert "locomotion.normal_movement" in dict_matrix


def test_spatial_metrics():
    subjects = [
        SubjectInfo(track_id=1, bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2)),
        SubjectInfo(track_id=2, bbox=BoundingBox(x_min=0.8, y_min=0.8, x_max=0.9, y_max=0.9)),
    ]

    metrics = compute_spatial_metrics(subjects, arena_area=1.0)
    assert metrics.active_subjects_count == 2
    assert metrics.mean_inter_individual_distance > 0.0
    assert metrics.clark_evans_dispersion_index > 0.0


def test_sampling_protocols():
    events = [
        _make_event("resting", "resting", 0.0, 10.0, track_ids=[1]),
        _make_event("locomotion", "normal_movement", 10.0, 30.0, track_ids=[1, 2]),
        _make_event("feeding", "foraging", 30.0, 40.0, track_ids=[2]),
    ]

    # Time budget
    summary = SamplingProtocols.compute_ethogram_time_budget(events, total_duration=40.0)
    assert summary.total_events == 3
    assert summary.duration_seconds == 40.0
    assert "resting" in summary.time_budget_percentage
    assert summary.activity_index > 0.0

    # Focal sampling for track_id 2
    focal = SamplingProtocols.extract_focal_sampling(events, track_id=2)
    assert len(focal) == 2

    # Scan sampling
    scans = SamplingProtocols.extract_scan_sampling(events, time_interval_seconds=15.0)
    assert len(scans) >= 2
