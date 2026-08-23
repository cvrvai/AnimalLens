"""
Rule-based temporal behavior classifier for Redclaw Crayfish (Cherax quadricarinatus).
Translates kinematic trajectories, inter-individual distance, and duration into ethological events.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List
from animallens.perception.base import BaseBehaviorClassifier, FramePerceptionData, TrackState
from animallens.species.base import SpeciesAdapter


class RuleBasedRedclawClassifier(BaseBehaviorClassifier):
    """
    Temporal behavior classifier using ethological kinematic rules.
    Designed for development & baseline benchmarking.
    """

    def __init__(self, proximity_threshold: float = 0.22, contact_threshold: float = 0.14) -> None:
        self.proximity_threshold = proximity_threshold
        self.contact_threshold = contact_threshold

    @property
    def model_name(self) -> str:
        return "redclaw-kinematic-rules-v1"

    def _calculate_distance(self, t1: TrackState, t2: TrackState) -> float:
        c1 = t1.current_bbox.center
        c2 = t2.current_bbox.center
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    def classify(
        self,
        recent_frames_data: List[FramePerceptionData],
        species_adapter: SpeciesAdapter,
    ) -> List[Dict[str, Any]]:
        if not recent_frames_data:
            return []

        latest = recent_frames_data[-1]
        active_tracks = latest.tracks
        duration = recent_frames_data[-1].timestamp - recent_frames_data[0].timestamp
        events: List[Dict[str, Any]] = []

        if len(active_tracks) >= 2:
            # Pairwise interaction analysis
            t1, t2 = active_tracks[0], active_tracks[1]
            dist = self._calculate_distance(t1, t2)
            avg_vel = (t1.velocity + t2.velocity) / 2.0

            # 1. Mating: close contact for continuous time with low/medium velocity
            if dist < self.contact_threshold and duration >= 2.0 and avg_vel < 0.25:
                events.append({
                    "category": "reproduction",
                    "label": "mating",
                    "confidence": 0.92,
                    "subjects": [t1.to_subject_info(), t2.to_subject_info()],
                    "start": recent_frames_data[0].timestamp,
                    "end": latest.timestamp,
                    "duration": round(duration, 2),
                })
            # 2. Aggression: high velocity close approach
            elif dist < self.proximity_threshold and avg_vel > 0.40:
                events.append({
                    "category": "aggression",
                    "label": "aggression",
                    "confidence": 0.86,
                    "subjects": [t1.to_subject_info(), t2.to_subject_info()],
                    "start": recent_frames_data[0].timestamp,
                    "end": latest.timestamp,
                    "duration": round(duration, 2),
                })
            # 3. Social interaction: moderate proximity
            elif dist < self.proximity_threshold:
                events.append({
                    "category": "social_interaction",
                    "label": "social_interaction",
                    "confidence": 0.78,
                    "subjects": [t1.to_subject_info(), t2.to_subject_info()],
                    "start": recent_frames_data[0].timestamp,
                    "end": latest.timestamp,
                    "duration": round(duration, 2),
                })

        # Solitary behaviors
        for track in active_tracks:
            # If solitary or far apart
            if len(active_tracks) == 1 or (len(active_tracks) > 1 and dist >= self.proximity_threshold):
                if track.velocity < 0.03:
                    # Resting or Sheltering
                    events.append({
                        "category": "resting",
                        "label": "resting",
                        "confidence": 0.89,
                        "subjects": [track.to_subject_info()],
                        "start": recent_frames_data[0].timestamp,
                        "end": latest.timestamp,
                        "duration": round(duration, 2),
                    })
                elif 0.03 <= track.velocity <= 0.25:
                    events.append({
                        "category": "locomotion",
                        "label": "normal_movement",
                        "confidence": 0.91,
                        "subjects": [track.to_subject_info()],
                        "start": recent_frames_data[0].timestamp,
                        "end": latest.timestamp,
                        "duration": round(duration, 2),
                    })
                elif track.velocity > 0.25:
                    # Feeding / foraging or rapid movement
                    events.append({
                        "category": "feeding",
                        "label": "foraging",
                        "confidence": 0.81,
                        "subjects": [track.to_subject_info()],
                        "start": recent_frames_data[0].timestamp,
                        "end": latest.timestamp,
                        "duration": round(duration, 2),
                    })

        # If nothing detected or ambiguous, emit unknown
        if not events and active_tracks:
            events.append({
                "category": "unknown",
                "label": "unknown",
                "confidence": 0.38,
                "subjects": [t.to_subject_info() for t in active_tracks],
                "start": recent_frames_data[0].timestamp,
                "end": latest.timestamp,
                "duration": round(duration, 2),
            })

        return events
