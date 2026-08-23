"""
Mock / Development Object Detector for AnimalLens.
Provides deterministic/simulated animal bounding boxes during development when real weights are not loaded.
"""
from __future__ import annotations

import math
import random
from typing import Any, List, Optional
from animallens.core.schemas import BoundingBox
from animallens.perception.base import BaseDetector, DetectionResult


class MockDetector(BaseDetector):
    """
    Development-only detector.
    Simulates detection of animal subjects with realistic spatial coordinates.
    """

    def __init__(self, num_subjects: int = 2, species_name: str = "redclaw") -> None:
        self.num_subjects = num_subjects
        self._species_name = species_name
        self._step = 0

    @property
    def model_name(self) -> str:
        return "mock-yolo-detector-dev"

    def detect(self, frame: Any, confidence_threshold: float = 0.45) -> DetectionResult:
        """Generate realistic synthetic bounding boxes."""
        self._step += 1
        bboxes: List[BoundingBox] = []
        confidences: List[float] = []
        class_names: List[str] = []

        # Simulate movement trajectories
        for i in range(self.num_subjects):
            # Deterministic pseudo-orbit
            phase = (self._step * 0.05) + (i * math.pi)
            center_x = 0.5 + 0.25 * math.cos(phase) * (0.8 if i == 1 else 1.0)
            center_y = 0.5 + 0.20 * math.sin(phase * 1.3)

            w = 0.15
            h = 0.12

            x_min = max(0.05, min(0.95 - w, center_x - w / 2))
            y_min = max(0.05, min(0.95 - h, center_y - h / 2))
            x_max = x_min + w
            y_max = y_min + h

            conf = 0.88 + 0.08 * math.sin(phase)
            if conf >= confidence_threshold:
                bboxes.append(BoundingBox(x_min=round(x_min, 4), y_min=round(y_min, 4), x_max=round(x_max, 4), y_max=round(y_max, 4)))
                confidences.append(round(conf, 3))
                class_names.append(self._species_name)

        return DetectionResult(
            bboxes=bboxes,
            confidences=confidences,
            class_names=class_names,
        )
