"""
Domestic Dog (Canis lupus familiaris) species adapter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from animallens.species.base import BehaviorCategory, BehaviorTaxonomy, SpeciesAdapter, SpeciesConfig


class DogAdapter(SpeciesAdapter):
    """Domain-specific adapter for Canis lupus familiaris."""

    def __init__(self, directory: Path | None = None) -> None:
        if directory is None:
            directory = Path(__file__).parent
        super().__init__(directory=directory)

    def default_config(self) -> SpeciesConfig:
        return SpeciesConfig(
            id="canis_lupus_familiaris",
            name="Domestic Dog",
            scientific_name="Canis lupus familiaris",
            taxonomy_version="1.0.0",
            default_model="yolov8n.pt",
            detection_threshold=0.40,
            min_track_length=3,
            uncertainty_threshold=0.50,
        )

    def default_taxonomy(self) -> BehaviorTaxonomy:
        return BehaviorTaxonomy(
            version="1.0.0",
            categories={
                "posture": BehaviorCategory(
                    name="posture",
                    labels=["standing", "sitting", "lying_sternal", "lying_lateral", "sleeping"],
                ),
                "locomotion": BehaviorCategory(
                    name="locomotion",
                    labels=["walking", "trotting", "running_gallop", "jumping", "tail_wagging"],
                ),
                "social_behavior": BehaviorCategory(
                    name="social_behavior",
                    labels=["play_bow", "sniffing_conspecific", "following", "mounting", "greeting"],
                ),
                "aggression": BehaviorCategory(
                    name="aggression",
                    labels=["growling_stance", "aggressive_lunge", "biting_grapple", "defensive_retreat"],
                ),
                "maintenance": BehaviorCategory(
                    name="maintenance",
                    labels=["eating", "drinking", "grooming_scratching", "panting"],
                ),
                "unknown": BehaviorCategory(
                    name="unknown",
                    labels=["unknown"],
                ),
            },
        )

    def extract_custom_features(self, tracks: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract canine kinematic features (e.g. speed, play bow posture, social spacing)."""
        return {
            "species": "canis_lupus_familiaris",
            "track_count": len(tracks),
        }
