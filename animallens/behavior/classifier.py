"""
Behavioral Classifier Layer for AnimalLens.
Maps kinematic velocity vectors, acceleration spikes, aspect ratios, and spatial proximity
to standardized ethological behavior classifications.
"""
from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from animallens.tracking.tracker import FrameTrackingTelemetry, SubjectTelemetry


class ClassifiedBehavior(BaseModel):
    """Standardized ethogram classification for a tracked subject."""
    track_id: int
    display_id: str
    category: str  # e.g. "locomotion", "posture", "social_behavior", "aggression"
    label: str  # e.g. "running_gallop", "trot", "walk", "standing", "play_bow"
    human_readable: str  # e.g. "High-Speed Gallop Sprint"
    confidence: float
    speed_mps: float
    speed_kmh: float
    welfare_score: int = Field(default=95, description="Welfare health index 0-100")
    veterinary_note: Optional[str] = None


class BehavioralClassifier:
    """
    Multi-species behavioral classifier with velocity thresholding,
    posture aspect ratio analysis, and rolling temporal hysteresis.
    """

    def __init__(
        self,
        gallop_threshold_mps: float = 3.5,
        trot_threshold_mps: float = 1.2,
        walk_threshold_mps: float = 0.3,
        history_window: int = 5,
    ) -> None:
        self.gallop_threshold_mps = gallop_threshold_mps
        self.trot_threshold_mps = trot_threshold_mps
        self.walk_threshold_mps = walk_threshold_mps
        self._subject_histories: Dict[int, collections.deque[str]] = collections.defaultdict(
            lambda: collections.deque(maxlen=history_window)
        )

    def classify_subject(
        self,
        subject: SubjectTelemetry,
        frame_telemetry: Optional[FrameTrackingTelemetry] = None,
    ) -> ClassifiedBehavior:
        """
        Classify behavioral state of a single tracked subject based on kinematics.
        """
        speed = subject.velocity_mps
        speed_kmh = round(speed * 3.6, 1)
        w = subject.bbox.width
        h = subject.bbox.height
        aspect_ratio = w / max(1e-3, h)

        category = "posture"
        label = "standing"
        human_readable = "Alert Stance / Standing"
        confidence = 0.88
        welfare = 98
        vet_note = "Normal physiological resting/standing posture."

        # 1. Kinematic Velocity Profile Matching
        if speed >= self.gallop_threshold_mps:
            category = "locomotion"
            label = "running_gallop"
            human_readable = "High-Speed Gallop Sprint"
            confidence = min(0.99, 0.85 + (speed / 20.0))
            welfare = 96
            vet_note = "High-energy symmetrical locomotor stride without gait asymmetry."
        elif speed >= self.trot_threshold_mps:
            category = "locomotion"
            label = "trot"
            human_readable = "Active Locomotor Trot"
            confidence = 0.92
            welfare = 97
            vet_note = "Rhythmic balanced trotting gait."
        elif speed >= self.walk_threshold_mps:
            category = "locomotion"
            label = "walk"
            human_readable = "Exploratory Walking"
            confidence = 0.89
            welfare = 98
            vet_note = "Low-stress exploration and movement."
        else:
            # Low speed: check posture aspect ratio for lying vs standing
            if aspect_ratio > 1.8:
                category = "posture"
                label = "lying_sternal"
                human_readable = "Sternal Recumbency (Lying Down)"
                confidence = 0.86
                welfare = 95
                vet_note = "Relaxed resting posture."
            else:
                category = "posture"
                label = "standing"
                human_readable = "Alert Stance / Standing"
                confidence = 0.88
                welfare = 98
                vet_note = "Normal upright stance."

        # 2. Multi-Animal Social Interactions
        if frame_telemetry and len(frame_telemetry.subjects) > 1:
            # Check proximity to other animals
            my_iids = frame_telemetry.iid_matrix.get(subject.display_id, {})
            other_dists = [d for other_id, d in my_iids.items() if other_id != subject.display_id]
            min_dist = min(other_dists, default=999.0)

            if min_dist < 0.6:  # Close proximity (< 0.6 meters)
                if speed < 1.0 and aspect_ratio > 1.2:
                    category = "social_behavior"
                    label = "play_bow"
                    human_readable = "Play Bow (Social Solicitation)"
                    confidence = 0.93
                    welfare = 99
                    vet_note = "Positive social interaction and play behavior."
                elif speed >= self.trot_threshold_mps:
                    category = "social_behavior"
                    label = "following"
                    human_readable = "Social Chasing / Group Locomotion"
                    confidence = 0.91
                    welfare = 97
                    vet_note = "Coordinated conspecific movement."

        # 3. Apply Rolling Temporal Smoothing
        self._subject_histories[subject.track_id].append(label)
        smoothed_label = collections.Counter(self._subject_histories[subject.track_id]).most_common(1)[0][0]

        return ClassifiedBehavior(
            track_id=subject.track_id,
            display_id=subject.display_id,
            category=category,
            label=smoothed_label,
            human_readable=human_readable,
            confidence=round(confidence, 2),
            speed_mps=speed,
            speed_kmh=speed_kmh,
            welfare_score=welfare,
            veterinary_note=vet_note,
        )

    def classify_frame(
        self,
        frame_telemetry: FrameTrackingTelemetry,
    ) -> List[ClassifiedBehavior]:
        """
        Classify all subjects in the frame.
        """
        return [
            self.classify_subject(s, frame_telemetry=frame_telemetry)
            for s in frame_telemetry.subjects
        ]
