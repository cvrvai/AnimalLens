"""
Unit tests for Kinematics Engine, Differential Features, and Temporal Action Classifier.
"""
import pytest
from animallens.analytics.kinematics import KinematicsEngine
from animallens.core.schemas import BoundingBox
from animallens.perception.base import DetectionResult, FramePerceptionData, TrackState
from animallens.perception.pipeline import PerceptionPipeline
from animallens.perception.temporal.classifier import TemporalBehaviorClassifier
from animallens.species.registry import species_registry


def _make_track(track_id: int, x1: float, y1: float, x2: float, y2: float) -> TrackState:
    return TrackState(
        track_id=track_id,
        current_bbox=BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2),
        velocity=0.0,
        confidence=1.0,
    )


def test_kinematics_engine_single_animal_velocity():
    """Verify single-animal velocity and heading computation across sequential frames."""
    trk_t1 = _make_track(1, 0.1, 0.1, 0.2, 0.2)  # Center (0.15, 0.15)
    trk_t2 = _make_track(1, 0.2, 0.2, 0.3, 0.3)  # Center (0.25, 0.25) -> moved (0.1, 0.1)

    dt = 1.0  # 1 second delta
    prev_map = {1: trk_t1}
    report = KinematicsEngine.compute_frame_kinematics([trk_t2], previous_tracks=prev_map, dt=dt)

    t1_kin = report.tracks[1]
    assert pytest.approx(t1_kin.velocity[0], 0.001) == 0.1
    assert pytest.approx(t1_kin.velocity[1], 0.001) == 0.1
    assert pytest.approx(t1_kin.speed, 0.001) == pytest.approx(0.1414, 0.001)
    assert 40.0 <= t1_kin.heading_degrees <= 50.0  # ~45 degrees heading


def test_kinematics_pairwise_iid_and_approach_rate():
    """Verify Inter-Individual Distance (IID) and approach rate between 2 animals."""
    t1_prev = _make_track(1, 0.1, 0.5, 0.2, 0.6)  # center (0.15, 0.55)
    t2_prev = _make_track(2, 0.8, 0.5, 0.9, 0.6)  # center (0.85, 0.55) -> IID = 0.70

    # Moving toward each other
    t1_curr = _make_track(1, 0.2, 0.5, 0.3, 0.6)  # center (0.25, 0.55)
    t2_curr = _make_track(2, 0.7, 0.5, 0.8, 0.6)  # center (0.75, 0.55) -> IID = 0.50

    prev_map = {1: t1_prev, 2: t2_prev}
    report = KinematicsEngine.compute_frame_kinematics([t1_curr, t2_curr], previous_tracks=prev_map, dt=1.0)

    assert len(report.pairwise) == 1
    pair = report.pairwise[0]
    assert pytest.approx(pair.distance, 0.01) == 0.50
    # Negative approach rate indicates closing in
    assert pair.approach_rate < 0.0


def test_temporal_classifier_resting():
    """Verify resting classification for stationary animals."""
    species = species_registry.get("redclaw")
    classifier = TemporalBehaviorClassifier()

    trk = _make_track(1, 0.4, 0.4, 0.5, 0.5)
    frame_data = [
        FramePerceptionData(frame_index=1, timestamp=0.0, detections=DetectionResult(), tracks=[trk]),
        FramePerceptionData(frame_index=2, timestamp=1.0, detections=DetectionResult(), tracks=[trk]),
    ]

    events = classifier.classify(frame_data, species)
    assert len(events) == 1
    assert events[0]["label"] == "resting"


def test_temporal_classifier_fighting_approach():
    """Verify fighting classification when animals rapidly approach and grapple."""
    species = species_registry.get("redclaw")
    classifier = TemporalBehaviorClassifier()

    # Rapid closing encounter
    t1_prev = _make_track(1, 0.1, 0.5, 0.2, 0.6)
    t2_prev = _make_track(2, 0.6, 0.5, 0.7, 0.6)

    t1_curr = _make_track(1, 0.35, 0.5, 0.45, 0.6)  # almost touching
    t2_curr = _make_track(2, 0.46, 0.5, 0.56, 0.6)

    frame_data = [
        FramePerceptionData(frame_index=1, timestamp=0.0, detections=DetectionResult(), tracks=[t1_prev, t2_prev]),
        FramePerceptionData(frame_index=2, timestamp=0.5, detections=DetectionResult(), tracks=[t1_curr, t2_curr]),
    ]

    events = classifier.classify(frame_data, species)
    assert len(events) == 1
    assert events[0]["label"] == "fighting"
    assert events[0]["category"] == "aggression"
