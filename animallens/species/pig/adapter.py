"""
Swine / Domestic Pig (Sus scrofa domesticus) species adapter.
Provides clinical recumbency posture classification, commercial barn huddling index, and agonistic welfare analytics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from animallens.species.base import (
    BehaviorCategory,
    BehaviorTaxonomy,
    SpeciesAdapter,
    SpeciesConfig,
)


class PigAdapter(SpeciesAdapter):
    """Domain-specific adapter for Sus scrofa domesticus (Swine / Domestic Pig)."""

    def __init__(self, directory: Path | None = None) -> None:
        if directory is None:
            directory = Path(__file__).parent
        super().__init__(directory=directory)

    def default_config(self) -> SpeciesConfig:
        return SpeciesConfig(
            id="sus_scrofa_domesticus",
            name="Domestic Pig",
            scientific_name="Sus scrofa domesticus",
            taxonomy_version="1.0.0",
            default_model="pig-posture-v1.pt",
            detection_threshold=0.35,
            min_track_length=3,
            uncertainty_threshold=0.50,
            classes=["pig", "swine", "hog", "sow", "piglet", "sus_scrofa_domesticus"],
        )

    def default_taxonomy(self) -> BehaviorTaxonomy:
        return BehaviorTaxonomy(
            version="1.0.0",
            categories={
                "posture": BehaviorCategory(
                    name="posture",
                    labels=[
                        "standing",
                        "sternal_recumbency",
                        "lateral_recumbency",
                        "sitting_dog_sitting",
                        "huddling_cold_stress",
                        "panting_heat_stress",
                    ],
                ),
                "locomotion": BehaviorCategory(
                    name="locomotion",
                    labels=["walking", "running_gallop", "trotting", "rooting_exploration"],
                ),
                "feeding": BehaviorCategory(
                    name="feeding",
                    labels=["eating_trough", "drinking_nipple", "sham_chewing_stress"],
                ),
                "social_behavior": BehaviorCategory(
                    name="social_behavior",
                    labels=["nosing_conspecific", "mounting_estrus"],
                ),
                "aggression": BehaviorCategory(
                    name="aggression",
                    labels=["head_knocking_aggression", "parallel_pressing", "tail_biting_cannibalism"],
                ),
                "unknown": BehaviorCategory(
                    name="unknown",
                    labels=["unknown"],
                ),
            },
        )

    def compute_huddling_index(self, tracks: List[Any]) -> float:
        """
        Computes the commercial barn Huddling Clumping Index [0.0, 1.0].
        Scores whether pigs are clustering tightly together due to cold ambient barn temperature (<18°C).
        """
        if len(tracks) < 2:
            return 0.0

        centers = []
        for t in tracks:
            bbox = getattr(t, "current_bbox", getattr(t, "bbox", None))
            if bbox:
                cx = (bbox.x_min + bbox.x_max) / 2.0
                cy = (bbox.y_min + bbox.y_max) / 2.0
                centers.append((cx, cy))

        if len(centers) < 2:
            return 0.0

        # Calculate average pairwise distance between all pigs in pen
        distances = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                distances.append(dist)

        avg_dist = float(np.mean(distances))
        # Normalized inverse distance: small average distance (<0.15 screen width) indicates severe huddling
        huddling_score = max(0.0, min(1.0, 1.0 - (avg_dist / 0.35)))
        return round(huddling_score, 3)

    def classify_recumbency(self, bbox: Any, velocity: float = 0.0) -> str:
        """
        Differentiates lateral lying (flat on side) vs sternal lying (chest down) based on aspect ratio.
        """
        if velocity > 0.3:
            return "walking"

        width = getattr(bbox, "width", 0.0)
        height = getattr(bbox, "height", 1.0)
        aspect_ratio = width / max(height, 0.001)

        if aspect_ratio >= 1.8:
            return "lateral_recumbency"
        elif aspect_ratio >= 1.2:
            return "sternal_recumbency"
        elif aspect_ratio >= 0.8:
            return "standing"
        else:
            return "sitting_dog_sitting"

    def extract_custom_features(self, tracks: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract swine ethological features, thermal huddling index, and recumbency distribution."""
        huddling_idx = self.compute_huddling_index(tracks)

        postures = []
        for t in tracks:
            bbox = getattr(t, "current_bbox", getattr(t, "bbox", None))
            vel = getattr(t, "velocity", 0.0)
            if bbox:
                postures.append(self.classify_recumbency(bbox, velocity=vel))

        return {
            "species": "sus_scrofa_domesticus",
            "active_pig_count": len(tracks),
            "huddling_cold_stress_index": huddling_idx,
            "thermal_comfort_status": "Cold Thermal Stress (Huddling)" if huddling_idx > 0.65 else "Normal Thermal Range",
            "posture_distribution": {p: postures.count(p) for p in set(postures)} if postures else {},
        }
