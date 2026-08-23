"""
Temporal Behavior Action Classifier for Ethological Action Segmentation.
Uses multi-frame kinematic trajectory analysis and temporal smoothing
to classify complex interactive behaviors (mating, fighting, threat display, foraging, resting).
"""
from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional
from animallens.analytics.kinematics import KinematicsEngine
from animallens.perception.base import BaseBehaviorClassifier, FramePerceptionData


class TemporalBehaviorClassifier(BaseBehaviorClassifier):
    """
    Classifies animal behaviors across a rolling temporal window.
    Applies ethological trajectory rules, approach rates, and temporal hysteresis smoothing.
    """

    def __init__(
        self,
        min_duration_seconds: float = 1.0,
        mating_proximity_threshold: float = 0.20,
        aggression_approach_rate_threshold: float = -0.08,
        resting_speed_threshold: float = 0.015,
        escape_accel_threshold: float = 0.50,
    ) -> None:
        self.min_duration_seconds = min_duration_seconds
        self.mating_proximity_threshold = mating_proximity_threshold
        self.aggression_approach_rate_threshold = aggression_approach_rate_threshold
        self.resting_speed_threshold = resting_speed_threshold
        self.escape_accel_threshold = escape_accel_threshold
        self._history_labels: collections.deque[str] = collections.deque(maxlen=5)

    @property
    def model_name(self) -> str:
        return "temporal-kinematic-v1"

    def classify(
        self,
        recent_frames_data: List[FramePerceptionData],
        species_adapter: Any,
    ) -> List[Dict[str, Any]]:
        if not recent_frames_data:
            return []

        first_frame = recent_frames_data[0]
        last_frame = recent_frames_data[-1]
        t_start = first_frame.timestamp
        t_end = last_frame.timestamp
        duration = max(0.0, t_end - t_start)

        # Extract kinematics from the latest frames
        curr_tracks = last_frame.tracks
        prev_tracks = {t.track_id: t for t in recent_frames_data[-2].tracks} if len(recent_frames_data) >= 2 else None

        dt = (t_end - recent_frames_data[-2].timestamp) if len(recent_frames_data) >= 2 else 0.0333
        kinematics = KinematicsEngine.compute_frame_kinematics(curr_tracks, prev_tracks, dt=dt)

        candidate_label = "resting"
        candidate_category = "resting"
        confidence = 0.85
        subjects_info = [t.to_subject_info() for t in curr_tracks]

        max_accel = max([t.acceleration_magnitude for t in kinematics.tracks.values()], default=0.0)

        # 1. Multi-Animal Social & Agonistic Interactions (Prioritized when interacting)
        if len(kinematics.pairwise) > 0:
            closest_pair = min(kinematics.pairwise, key=lambda p: p.distance)

            # A. Fighting / Grappling (Rapid closing approach and close contact)
            if closest_pair.approach_rate < self.aggression_approach_rate_threshold and closest_pair.distance < 0.30:
                candidate_label = "fighting"
                candidate_category = "aggression"
                confidence = 0.94

            # B. Mating / Courtship (Close contact, low relative speed, sustained duration)
            elif closest_pair.distance < self.mating_proximity_threshold and closest_pair.relative_speed < 0.06:
                if duration >= 1.5:
                    candidate_label = "mating"
                    candidate_category = "reproduction"
                    confidence = 0.92
                else:
                    candidate_label = "courtship"
                    candidate_category = "reproduction"
                    confidence = 0.86

            # C. Threat Display (Facing opponent with low translation speed)
            elif closest_pair.distance < 0.35 and kinematics.mean_speed < 0.03:
                candidate_label = "threat_display"
                candidate_category = "aggression"
                confidence = 0.88

            # D. Tail-Flip Escape / Avoidance Retreat (Rapid retreat away from partner)
            elif closest_pair.approach_rate > 0.15 and max_accel > self.escape_accel_threshold:
                candidate_label = "tail_flip_escape"
                candidate_category = "locomotion"
                confidence = 0.95

            elif closest_pair.approach_rate > 0.08 and closest_pair.distance < 0.40:
                candidate_label = "avoidance_retreat"
                candidate_category = "social_interaction"
                confidence = 0.82

        # 2. Individual Solitary Behaviors
        if candidate_label == "resting":
            if max_accel > self.escape_accel_threshold:
                candidate_label = "tail_flip_escape"
                candidate_category = "locomotion"
                confidence = 0.92
            elif kinematics.mean_speed >= self.resting_speed_threshold:
                if kinematics.mean_speed > 0.10:
                    candidate_label = "rapid_locomotion"
                    candidate_category = "locomotion"
                    confidence = 0.89
                else:
                    candidate_label = "foraging"
                    candidate_category = "feeding"
                    confidence = 0.87

        # 3. Species-Specific Label Mapping (e.g. Canines vs Crustaceans)
        is_canine = hasattr(species_adapter, "config") and "canis" in species_adapter.config.id.lower()

        if is_canine:
            if candidate_label in ("resting", "burrowing"):
                candidate_label = "standing"
                candidate_category = "posture"
            elif candidate_label in ("foraging", "rapid_locomotion"):
                candidate_label = "walking" if kinematics.mean_speed < 0.12 else "running_gallop"
                candidate_category = "locomotion"
            elif candidate_label in ("tail_flip_escape", "avoidance_retreat"):
                candidate_label = "defensive_retreat"
                candidate_category = "aggression"
            elif candidate_label == "fighting":
                candidate_label = "aggressive_lunge"
                candidate_category = "aggression"
            elif candidate_label in ("mating", "courtship"):
                candidate_label = "play_bow" if duration < 3.0 else "following"
                candidate_category = "social_behavior"

        # 4. Temporal Smoothing (Hysteresis majority voting to eliminate 1-frame noise)
        self._history_labels.append(candidate_label)
        smoothed_label = collections.Counter(self._history_labels).most_common(1)[0][0]

        # Look up taxonomy category from species adapter
        resolved_category = candidate_category
        if hasattr(species_adapter, "taxonomy") and species_adapter.taxonomy:
            for cat_name, cat_obj in species_adapter.taxonomy.categories.items():
                if smoothed_label in cat_obj.labels:
                    resolved_category = cat_name
                    break

        return [
            {
                "label": smoothed_label,
                "category": resolved_category,
                "confidence": confidence,
                "start": round(t_start, 3),
                "end": round(t_end, 3),
                "duration": round(duration, 3),
                "subjects": subjects_info,
                "kinematics": kinematics.model_dump(),
            }
        ]
