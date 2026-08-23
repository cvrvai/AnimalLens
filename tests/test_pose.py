"""
Unit Tests for YOLOv8PoseEstimator and PoseKinematicsEngine (Phase 9).
"""
import numpy as np
import pytest
from animallens.core.schemas import BoundingBox
from animallens.perception.models.yolov8_pose import (
    CANINE_KEYPOINT_NAMES,
    CANINE_SKELETAL_BONES,
    AnimalPose,
    Keypoint,
    YOLOv8PoseEstimator,
)
from animallens.analytics.pose_kinematics import BiomechanicalMetrics, PoseKinematicsEngine


def test_keypoint_and_pose_structures():
    kp = Keypoint(name="snout", x=0.5, y=0.5, confidence=0.95)
    d = kp.to_dict()
    assert d["name"] == "snout"
    assert d["x"] == 0.5
    assert d["confidence"] == 0.95

    pose = AnimalPose(
        track_id=1,
        display_id="DOG-01",
        keypoints={"snout": kp},
        overall_confidence=0.95,
    )
    assert pose.get_coords("snout") == (0.5, 0.5)
    assert pose.get_coords("nonexistent") is None
    assert "snout" in pose.to_dict()["keypoints"]


def test_pose_estimator_fallback_rig():
    estimator = YOLOv8PoseEstimator(conf_threshold=0.3)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = [
        BoundingBox(x_min=0.1, y_min=0.1, x_max=0.4, y_max=0.6),
        BoundingBox(x_min=0.5, y_min=0.2, x_max=0.8, y_max=0.7),
    ]
    poses = estimator.estimate_pose(frame, bboxes=boxes, track_ids=[1, 2])
    assert len(poses) == 2
    assert poses[0].display_id == "DOG-01"
    assert poses[1].display_id == "DOG-02"

    for name in CANINE_KEYPOINT_NAMES:
        assert name in poses[0].keypoints
        kp = poses[0].keypoints[name]
        assert 0.0 <= kp.x <= 1.0
        assert 0.0 <= kp.y <= 1.0


def test_pose_kinematics_angles():
    engine = PoseKinematicsEngine()

    # 90-degree right angle at (0,0)
    p1 = (1.0, 0.0)
    p2 = (0.0, 0.0)
    p3 = (0.0, 1.0)
    angle = engine.calculate_angle_3pt(p1, p2, p3)
    assert pytest.approx(angle, 0.01) == 90.0

    # 180-degree straight line
    p4 = (-1.0, 0.0)
    angle_straight = engine.calculate_angle_3pt(p1, p2, p4)
    assert pytest.approx(angle_straight, 0.01) == 180.0


def test_pose_biomechanical_analysis():
    estimator = YOLOv8PoseEstimator()
    engine = PoseKinematicsEngine()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    box = BoundingBox(x_min=0.2, y_min=0.2, x_max=0.6, y_max=0.8)
    poses = estimator.estimate_pose(frame, bboxes=[box])
    assert len(poses) == 1

    metrics = engine.analyze_pose(poses[0])
    assert isinstance(metrics, BiomechanicalMetrics)
    assert metrics.display_id == "DOG-01"
    assert 0.0 <= metrics.spine_flexion_angle_deg <= 180.0
    assert 0.0 <= metrics.gait_asymmetry_score <= 1.0
    assert "Gait" in metrics.veterinary_gait_classification

    d = metrics.to_dict()
    assert "spine_flexion_angle_deg" in d
    assert "veterinary_gait_classification" in d
