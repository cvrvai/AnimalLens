"""
High-Performance Tracking & Kinematics Engine for AnimalLens.
Provides BoT-SORT Multi-Object Tracking with persistent animal IDs (e.g. DOG-01, DOG-02),
sub-pixel Kalman velocity vectors, heading angles, acceleration, and Inter-Individual Distance (IID) matrices.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field
from animallens.core.schemas import BoundingBox, SubjectInfo
from animallens.perception.base import DetectionResult, TrackState
from animallens.perception.models.botsort_tracker import BoTSORTTracker


class SubjectTelemetry(BaseModel):
    """Real-time kinematic telemetry for an individual tracked animal."""
    track_id: int
    display_id: str  # e.g. "DOG-01", "REDCLAW-01"
    species: str
    confidence: float
    bbox: BoundingBox
    center_pct: Tuple[float, float]  # (cx%, cy%) normalized 0-100%
    velocity_mps: float  # Scalar speed in meters/second
    velocity_vector: Tuple[float, float]  # (vx, vy) in m/s
    heading_degrees: float  # Heading orientation [0, 360) degrees
    acceleration_mps2: float  # Magnitude in m/s^2
    is_active: bool = True


class FrameTrackingTelemetry(BaseModel):
    """Complete multi-animal tracking telemetry for a single video frame."""
    timestamp: float
    frame_index: int
    subjects: List[SubjectTelemetry] = Field(default_factory=list)
    iid_matrix: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Inter-Individual Distance (IID) pairwise matrix in meters/normalized units"
    )
    group_centroid_pct: Tuple[float, float] = (50.0, 50.0)
    group_mean_speed: float = 0.0
    group_polarization: float = 0.0  # [0.0 - 1.0] Alignment coefficient


class AnimalTracker:
    """
    Production tracker integrating YOLOv8 detections, Kalman state estimation,
    and mathematical kinematics for web dashboards and microservices.
    """

    def __init__(
        self,
        species_prefix: str = "DOG",
        pixel_to_meter_ratio: float = 2.5,  # Scale factor: 1.0 normalized screen width approx 2.5 meters
        max_age: int = 15,
        min_hits: int = 2,
        iou_threshold: float = 0.25,
    ) -> None:
        self.species_prefix = species_prefix.upper()
        self.pixel_to_meter_ratio = pixel_to_meter_ratio
        self._tracker = BoTSORTTracker(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        self._prev_tracks: Dict[int, Dict[str, Any]] = {}
        self._frame_count = 0

    def reset(self) -> None:
        self._tracker.reset()
        self._prev_tracks.clear()
        self._frame_count = 0

    def update_frame(
        self,
        detections: DetectionResult,
        timestamp: float,
        dt: float = 0.0333,
        species_name: str = "Canis lupus familiaris",
    ) -> FrameTrackingTelemetry:
        """
        Process frame detections, update Kalman tracks, and compute kinematics.
        """
        self._frame_count += 1
        dt = max(1e-4, dt)

        # 1. Update BoT-SORT Kalman tracker
        raw_tracks = self._tracker.update(detections, timestamp=timestamp)

        subjects: List[SubjectTelemetry] = []
        positions_pct: List[Tuple[float, float]] = []
        unit_vels: List[Tuple[float, float]] = []

        # 2. Compute individual kinematic metrics
        for trk in raw_tracks:
            cx, cy = trk.current_bbox.center
            cx_pct = round(cx * 100.0, 2)
            cy_pct = round(cy * 100.0, 2)
            positions_pct.append((cx_pct, cy_pct))

            display_id = f"{self.species_prefix}-{trk.track_id:02d}"

            # Calculate differential velocity in meters/second
            if trk.track_id in self._prev_tracks:
                prev_data = self._prev_tracks[trk.track_id]
                prev_cx = prev_data["cx"]
                prev_cy = prev_data["cy"]
                prev_vx = prev_data["vx"]
                prev_vy = prev_data["vy"]

                # Displacement in normalized coordinates scaled by physical reference ratio
                vx = ((cx - prev_cx) * self.pixel_to_meter_ratio) / dt
                vy = ((cy - prev_cy) * self.pixel_to_meter_ratio) / dt
                speed = math.sqrt(vx ** 2 + vy ** 2)

                # Acceleration
                ax = (vx - prev_vx) / dt
                ay = (vy - prev_vy) / dt
                accel_mag = math.sqrt(ax ** 2 + ay ** 2)
            else:
                vx, vy, speed = 0.0, 0.0, 0.0
                ax, ay, accel_mag = 0.0, 0.0, 0.0

            # Heading angle in degrees
            heading = (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0 if speed > 0.05 else 0.0

            if speed > 0.05:
                unit_vels.append((vx / speed, vy / speed))

            # Store track state for next delta
            self._prev_tracks[trk.track_id] = {
                "cx": cx,
                "cy": cy,
                "vx": vx,
                "vy": vy,
                "timestamp": timestamp,
            }

            subjects.append(
                SubjectTelemetry(
                    track_id=trk.track_id,
                    display_id=display_id,
                    species=species_name,
                    confidence=trk.confidence,
                    bbox=trk.current_bbox,
                    center_pct=(cx_pct, cy_pct),
                    velocity_mps=round(speed, 2),
                    velocity_vector=(round(vx, 2), round(vy, 2)),
                    heading_degrees=round(heading, 1),
                    acceleration_mps2=round(accel_mag, 2),
                    is_active=True,
                )
            )

        # 3. Compute Inter-Individual Distance (IID) Matrix
        iid_matrix: Dict[str, Dict[str, float]] = {}
        for i, s1 in enumerate(subjects):
            iid_matrix[s1.display_id] = {}
            for j, s2 in enumerate(subjects):
                if i == j:
                    iid_matrix[s1.display_id][s2.display_id] = 0.0
                else:
                    dx = (s1.bbox.center[0] - s2.bbox.center[0]) * self.pixel_to_meter_ratio
                    dy = (s1.bbox.center[1] - s2.bbox.center[1]) * self.pixel_to_meter_ratio
                    dist = math.sqrt(dx ** 2 + dy ** 2)
                    iid_matrix[s1.display_id][s2.display_id] = round(dist, 3)

        # 4. Group Centroid & Polarization
        if positions_pct:
            mean_cx = sum(p[0] for p in positions_pct) / len(positions_pct)
            mean_cy = sum(p[1] for p in positions_pct) / len(positions_pct)
            group_centroid = (round(mean_cx, 2), round(mean_cy, 2))
        else:
            group_centroid = (50.0, 50.0)

        group_mean_speed = (
            sum(s.velocity_mps for s in subjects) / len(subjects) if subjects else 0.0
        )

        if unit_vels:
            sum_uv_x = sum(u[0] for u in unit_vels)
            sum_uv_y = sum(u[1] for u in unit_vels)
            polarization = math.sqrt(sum_uv_x ** 2 + sum_uv_y ** 2) / len(unit_vels)
        else:
            polarization = 0.0

        return FrameTrackingTelemetry(
            timestamp=round(timestamp, 3),
            frame_index=self._frame_count,
            subjects=subjects,
            iid_matrix=iid_matrix,
            group_centroid_pct=group_centroid,
            group_mean_speed=round(group_mean_speed, 2),
            group_polarization=round(polarization, 3),
        )
