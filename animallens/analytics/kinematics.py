"""
Kinematic Feature Extraction Engine for Multi-Animal Behavior Analysis.
Calculates velocities, accelerations, heading angles, pairwise Inter-Individual Distance (IID),
approach rates, and group polarization metrics.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field
from animallens.core.schemas import BoundingBox, SubjectInfo
from animallens.perception.base import TrackState


class TrackKinematics(BaseModel):
    track_id: int
    position: Tuple[float, float]  # (x_center, y_center)
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy)
    speed: float = 0.0
    acceleration: Tuple[float, float] = (0.0, 0.0)  # (ax, ay)
    acceleration_magnitude: float = 0.0
    heading_degrees: float = 0.0  # Angle in [0, 360) degrees


class PairwiseKinematics(BaseModel):
    track_id_1: int
    track_id_2: int
    distance: float  # Inter-Individual Distance (IID)
    relative_speed: float  # ||v1 - v2||
    approach_rate: float  # d(IID)/dt (negative = closing in, positive = separating)
    is_in_contact: bool = False  # True if bounding boxes overlap


class FrameKinematicsReport(BaseModel):
    timestamp: float
    dt: float
    tracks: Dict[int, TrackKinematics]
    pairwise: List[PairwiseKinematics]
    group_centroid: Tuple[float, float] = (0.5, 0.5)
    mean_speed: float = 0.0
    polarization_index: float = 0.0  # Alignment of movement vectors [0, 1]


class KinematicsEngine:
    """
    Computes rigorous differential kinematic features across sequential tracking frames.
    """

    @staticmethod
    def compute_frame_kinematics(
        current_tracks: List[TrackState],
        previous_tracks: Optional[Dict[int, TrackState]] = None,
        dt: float = 0.0333,  # Default 30 FPS (~33.3ms)
    ) -> FrameKinematicsReport:
        dt = max(1e-4, dt)
        prev_map = previous_tracks or {}

        track_kinematics: Dict[int, TrackKinematics] = {}
        positions = []
        unit_velocities = []

        for trk in current_tracks:
            cx, cy = trk.current_bbox.center
            positions.append((cx, cy))

            # Compute velocity and acceleration from history
            if trk.track_id in prev_map:
                prev_trk = prev_map[trk.track_id]
                prev_cx, prev_cy = prev_trk.current_bbox.center

                vx = (cx - prev_cx) / dt
                vy = (cy - prev_cy) / dt
                speed = math.sqrt(vx ** 2 + vy ** 2)

                prev_vx = getattr(prev_trk, "_vx", 0.0)
                prev_vy = getattr(prev_trk, "_vy", 0.0)
                ax = (vx - prev_vx) / dt
                ay = (vy - prev_vy) / dt
                accel_mag = math.sqrt(ax ** 2 + ay ** 2)
            else:
                vx, vy, speed = 0.0, 0.0, 0.0
                ax, ay, accel_mag = 0.0, 0.0, 0.0

            # Store hidden velocity on track for next delta computation
            trk._vx = vx  # type: ignore[attr-defined]
            trk._vy = vy  # type: ignore[attr-defined]

            # Heading angle
            heading = (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0 if speed > 1e-4 else 0.0

            if speed > 1e-4:
                unit_velocities.append((vx / speed, vy / speed))

            track_kinematics[trk.track_id] = TrackKinematics(
                track_id=trk.track_id,
                position=(round(cx, 4), round(cy, 4)),
                velocity=(round(vx, 4), round(vy, 4)),
                speed=round(speed, 4),
                acceleration=(round(ax, 4), round(ay, 4)),
                acceleration_magnitude=round(accel_mag, 4),
                heading_degrees=round(heading, 1),
            )

        # Compute Pairwise Kinematics (Inter-Individual Distance & Approach Rates)
        pairwise: List[PairwiseKinematics] = []
        track_list = list(track_kinematics.values())
        for i in range(len(track_list)):
            for j in range(i + 1, len(track_list)):
                t1 = track_list[i]
                t2 = track_list[j]

                dx = t1.position[0] - t2.position[0]
                dy = t1.position[1] - t2.position[1]
                dist = math.sqrt(dx ** 2 + dy ** 2)

                rel_vx = t1.velocity[0] - t2.velocity[0]
                rel_vy = t1.velocity[1] - t2.velocity[1]
                rel_speed = math.sqrt(rel_vx ** 2 + rel_vy ** 2)

                # Approach rate: derivative of distance
                approach_rate = (dx * rel_vx + dy * rel_vy) / max(1e-4, dist)

                # Contact check
                b1 = current_tracks[i].current_bbox
                b2 = current_tracks[j].current_bbox
                in_contact = not (
                    b1.x_max < b2.x_min or b1.x_min > b2.x_max or
                    b1.y_max < b2.y_min or b1.y_min > b2.y_max
                )

                pairwise.append(PairwiseKinematics(
                    track_id_1=t1.track_id,
                    track_id_2=t2.track_id,
                    distance=round(dist, 4),
                    relative_speed=round(rel_speed, 4),
                    approach_rate=round(approach_rate, 4),
                    is_in_contact=in_contact,
                ))

        # Group Centroid and Polarization Index
        if positions:
            mean_cx = sum(p[0] for p in positions) / len(positions)
            mean_cy = sum(p[1] for p in positions) / len(positions)
            centroid = (round(mean_cx, 4), round(mean_cy, 4))
        else:
            centroid = (0.5, 0.5)

        mean_speed = sum(t.speed for t in track_list) / len(track_list) if track_list else 0.0

        # Polarization = || sum(v_unit) || / N (1.0 = all moving in same direction)
        if unit_velocities:
            sum_uv_x = sum(u[0] for u in unit_velocities)
            sum_uv_y = sum(u[1] for u in unit_velocities)
            polarization = math.sqrt(sum_uv_x ** 2 + sum_uv_y ** 2) / len(unit_velocities)
        else:
            polarization = 0.0

        return FrameKinematicsReport(
            timestamp=0.0,
            dt=dt,
            tracks=track_kinematics,
            pairwise=pairwise,
            group_centroid=centroid,
            mean_speed=round(mean_speed, 4),
            polarization_index=round(polarization, 4),
        )
