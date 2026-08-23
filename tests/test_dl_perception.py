"""
Unit tests for YOLOv8 Object Detector, BoT-SORT Kalman Tracker, and DL Pipeline.
"""
import numpy as np
import pytest
from animallens.core.schemas import BoundingBox
from animallens.perception.base import DetectionResult
from animallens.perception.models.botsort_tracker import BoTSORTTracker, compute_iou_matrix
from animallens.perception.models.kalman_filter import KalmanBoxTracker
from animallens.perception.models.yolov8_detector import YOLOv8Detector
from animallens.sdk import AnimalLens


def test_kalman_box_tracker():
    """Verify Kalman filter constant velocity motion estimation and kinematics."""
    init_box = [0.1, 0.1, 0.3, 0.3]
    tracker = KalmanBoxTracker(init_box)

    # Initial state verification
    state = tracker.get_state()
    assert len(state) == 4
    assert pytest.approx(state[0], 0.05) == 0.1
    assert pytest.approx(state[1], 0.05) == 0.1

    # Predict next state
    pred = tracker.predict()
    assert len(pred) == 4

    # Update with shifted measurement (moving right and down)
    measurement = [0.15, 0.15, 0.35, 0.35]
    tracker.update(measurement)

    # Speed / velocity should be positive
    speed = tracker.get_velocity()
    assert speed > 0.0


def test_compute_iou_matrix():
    """Verify pairwise bounding box IoU matrix computation."""
    boxes1 = np.array([
        [0.0, 0.0, 0.5, 0.5],
        [0.5, 0.5, 1.0, 1.0],
    ])
    boxes2 = np.array([
        [0.0, 0.0, 0.5, 0.5],  # Exact match with boxes1[0]
        [0.0, 0.0, 1.0, 1.0],  # Overlap with both
    ])

    iou_mat = compute_iou_matrix(boxes1, boxes2)
    assert iou_mat.shape == (2, 2)
    assert pytest.approx(iou_mat[0, 0], 0.01) == 1.0  # Perfect overlap
    assert iou_mat[0, 1] == 0.25  # 0.25 / 1.0
    assert iou_mat[1, 0] == 0.0  # Disjoint


def test_botsort_tracker_persistence():
    """Verify that BoT-SORT maintains persistent track IDs across frames."""
    tracker = BoTSORTTracker(max_age=5, min_hits=1)

    # Frame 1: 2 animals detected
    frame1_dets = DetectionResult(
        bboxes=[
            BoundingBox(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.3),
            BoundingBox(x_min=0.6, y_min=0.6, x_max=0.8, y_max=0.8),
        ],
        confidences=[0.9, 0.85],
        class_names=["redclaw", "redclaw"],
    )
    tracks_f1 = tracker.update(frame1_dets)
    assert len(tracks_f1) == 2
    id1, id2 = tracks_f1[0].track_id, tracks_f1[1].track_id
    assert id1 != id2

    # Frame 2: Slightly moved animals
    frame2_dets = DetectionResult(
        bboxes=[
            BoundingBox(x_min=0.12, y_min=0.11, x_max=0.32, y_max=0.31),
            BoundingBox(x_min=0.61, y_min=0.62, x_max=0.81, y_max=0.82),
        ],
        confidences=[0.92, 0.88],
        class_names=["redclaw", "redclaw"],
    )
    tracks_f2 = tracker.update(frame2_dets)
    assert len(tracks_f2) == 2
    f2_ids = {t.track_id for t in tracks_f2}
    assert id1 in f2_ids
    assert id2 in f2_ids


def test_yolov8_detector_fallback():
    """Verify YOLOv8 detector initialization and fallback inference."""
    detector = YOLOv8Detector(model_path="non_existent_weights.pt")
    assert detector._backend == "fallback"

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame)
    assert len(detections.bboxes) >= 2
    assert detections.bboxes[0].is_normalized is True
    assert 0.0 <= detections.confidences[0] <= 1.0


def test_sdk_with_yolov8_and_botsort():
    """Verify AnimalLens SDK initializes and runs with string detector and tracker names."""
    lens = AnimalLens(
        species="redclaw",
        detector="yolov8",
        tracker="botsort",
    )
    assert isinstance(lens.pipeline.detector, YOLOv8Detector)
    assert isinstance(lens.pipeline.tracker, BoTSORTTracker)

    # Analyze synthetic frame
    dummy_frame = np.zeros((300, 300, 3), dtype=np.uint8)
    result = lens.analyze_image(dummy_frame)
    assert result.species == "Redclaw Crayfish"
    assert result.total_frames_analyzed == 1
