"""
Pose Kinematics & Biomechanical Joint Angle Analysis Engine for AnimalLens.
Calculates 3-point joint angles, spine curvature kappa, play-bow geometry, and lameness asymmetry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from animallens.perception.models.yolov8_pose import AnimalPose


@dataclass
class BiomechanicalMetrics:
    """Biomechanical and Gait Metrics extracted from AnimalPose."""
    track_id: int
    display_id: str
    spine_flexion_angle_deg: float  # Angle at mid-spine between withers and tail base
    left_elbow_angle_deg: float
    right_elbow_angle_deg: float
    left_stifle_angle_deg: float
    right_stifle_angle_deg: float
    head_pitch_angle_deg: float     # Head elevation relative to withers
    is_play_bow: bool
    is_hunched_posture: bool
    gait_asymmetry_score: float     # 0.0 (perfectly symmetric) to 1.0 (severe lameness)
    veterinary_gait_classification: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "display_id": self.display_id,
            "spine_flexion_angle_deg": round(self.spine_flexion_angle_deg, 2),
            "left_elbow_angle_deg": round(self.left_elbow_angle_deg, 2),
            "right_elbow_angle_deg": round(self.right_elbow_angle_deg, 2),
            "left_stifle_angle_deg": round(self.left_stifle_angle_deg, 2),
            "right_stifle_angle_deg": round(self.right_stifle_angle_deg, 2),
            "head_pitch_angle_deg": round(self.head_pitch_angle_deg, 2),
            "is_play_bow": self.is_play_bow,
            "is_hunched_posture": self.is_hunched_posture,
            "gait_asymmetry_score": round(self.gait_asymmetry_score, 3),
            "veterinary_gait_classification": self.veterinary_gait_classification,
        }


class PoseKinematicsEngine:
    """
    Computes mathematical joint angles, biomechanical posture, and lameness asymmetry.
    """

    @staticmethod
    def calculate_angle_3pt(
        p1: Tuple[float, float],
        p2: Tuple[float, float],  # Vertex
        p3: Tuple[float, float],
    ) -> float:
        """
        Calculates internal angle at vertex p2 between rays (p2->p1) and (p2->p3) in degrees.
        """
        u = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        if norm_u < 1e-6 or norm_v < 1e-6:
            return 180.0

        cosine = np.dot(u, v) / (norm_u * norm_v)
        cosine = np.clip(cosine, -1.0, 1.0)
        angle_rad = np.arccos(cosine)
        return float(np.degrees(angle_rad))

    def analyze_pose(self, pose: AnimalPose) -> BiomechanicalMetrics:
        """
        Analyzes an AnimalPose instance and returns full BiomechanicalMetrics.
        """
        # 1. Spine Flexion Angle (Withers -> Mid-Spine -> Tail Base)
        withers = pose.get_coords("withers")
        mid_spine = pose.get_coords("mid_spine")
        tail_base = pose.get_coords("tail_base")

        if withers and mid_spine and tail_base:
            spine_angle = self.calculate_angle_3pt(withers, mid_spine, tail_base)
        else:
            spine_angle = 175.0

        # 2. Left & Right Elbow Flexion (Shoulder -> Elbow -> Front Paw)
        l_shoulder = pose.get_coords("left_shoulder")
        l_elbow = pose.get_coords("left_elbow")
        l_paw = pose.get_coords("left_front_paw")
        l_elbow_angle = self.calculate_angle_3pt(l_shoulder, l_elbow, l_paw) if (l_shoulder and l_elbow and l_paw) else 135.0

        r_shoulder = pose.get_coords("right_shoulder")
        r_elbow = pose.get_coords("right_elbow")
        r_paw = pose.get_coords("right_front_paw")
        r_elbow_angle = self.calculate_angle_3pt(r_shoulder, r_elbow, r_paw) if (r_shoulder and r_elbow and r_paw) else 135.0

        # 3. Left & Right Stifle/Knee Flexion (Hip -> Stifle -> Back Paw)
        l_hip = pose.get_coords("left_hip")
        l_stifle = pose.get_coords("left_stifle")
        l_back_paw = pose.get_coords("left_back_paw")
        l_stifle_angle = self.calculate_angle_3pt(l_hip, l_stifle, l_back_paw) if (l_hip and l_stifle and l_back_paw) else 140.0

        r_hip = pose.get_coords("right_hip")
        r_stifle = pose.get_coords("right_stifle")
        r_back_paw = pose.get_coords("right_back_paw")
        r_stifle_angle = self.calculate_angle_3pt(r_hip, r_stifle, r_back_paw) if (r_hip and r_stifle and r_back_paw) else 140.0

        # 4. Head Pitch Angle (Snout -> Neck -> Withers)
        snout = pose.get_coords("snout")
        neck = pose.get_coords("neck")
        head_pitch = self.calculate_angle_3pt(snout, neck, withers) if (snout and neck and withers) else 150.0

        # 5. Play Bow Geometry: Forequarters lowered (elbows flexed < 90 deg) while Hindquarters elevated
        is_play_bow = False
        if l_elbow and l_hip:
            # In image coords, higher y means lower in space
            forelimbs_collapsed = (l_elbow_angle < 95.0 or r_elbow_angle < 95.0)
            hindquarters_high = l_hip[1] < l_elbow[1]  # Hip is vertically higher than elbow
            is_play_bow = forelimbs_collapsed and hindquarters_high

        # 6. Hunched Posture (Kyphosis indicator)
        is_hunched = spine_angle < 150.0

        # 7. Left-Right Limb Asymmetry (Early Gait Lameness Indicator)
        elbow_diff = abs(l_elbow_angle - r_elbow_angle) / 180.0
        stifle_diff = abs(l_stifle_angle - r_stifle_angle) / 180.0
        asymmetry_score = min(1.0, (elbow_diff + stifle_diff) / 2.0)

        if asymmetry_score < 0.08:
            gait_class = "Symmetric Normal Gait"
        elif asymmetry_score < 0.20:
            gait_class = "Mild Asymmetry / Guarding"
        else:
            gait_class = "Significant Stride Asymmetry (Potential Lameness)"

        return BiomechanicalMetrics(
            track_id=pose.track_id,
            display_id=pose.display_id,
            spine_flexion_angle_deg=spine_angle,
            left_elbow_angle_deg=l_elbow_angle,
            right_elbow_angle_deg=r_elbow_angle,
            left_stifle_angle_deg=l_stifle_angle,
            right_stifle_angle_deg=r_stifle_angle,
            head_pitch_angle_deg=head_pitch,
            is_play_bow=is_play_bow,
            is_hunched_posture=is_hunched,
            gait_asymmetry_score=asymmetry_score,
            veterinary_gait_classification=gait_class,
        )
