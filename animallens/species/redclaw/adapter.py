"""
Redclaw Crayfish (Cherax quadricarinatus) species adapter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from animallens.species.base import BehaviorCategory, BehaviorTaxonomy, SpeciesAdapter, SpeciesConfig


class RedclawAdapter(SpeciesAdapter):
    """Domain-specific adapter for Cherax quadricarinatus."""

    def __init__(self, directory: Path | None = None) -> None:
        if directory is None:
            directory = Path(__file__).parent
        super().__init__(directory=directory)

    def default_config(self) -> SpeciesConfig:
        return SpeciesConfig(
            id="cherax_quadricarinatus",
            name="Redclaw Crayfish",
            scientific_name="Cherax quadricarinatus",
            taxonomy_version="1.0.0",
            default_model="redclaw-behavior-v1",
            detection_threshold=0.45,
            min_track_length=3,
            uncertainty_threshold=0.45,
        )

    def default_taxonomy(self) -> BehaviorTaxonomy:
        return BehaviorTaxonomy(
            version="1.0.0",
            categories={
                "locomotion": BehaviorCategory(
                    name="locomotion",
                    labels=["normal_movement", "crawling", "swimming", "tail_flip_escape"],
                ),
                "resting": BehaviorCategory(
                    name="resting",
                    labels=["resting", "sheltering", "burrowing"],
                ),
                "feeding": BehaviorCategory(
                    name="feeding",
                    labels=["feeding", "foraging", "food_competition"],
                ),
                "social_interaction": BehaviorCategory(
                    name="social_interaction",
                    labels=["social_interaction", "approach", "following", "avoidance", "antennal_contact"],
                ),
                "aggression": BehaviorCategory(
                    name="aggression",
                    labels=["aggression", "threat_display", "claw_sparring", "chasing"],
                ),
                "reproduction": BehaviorCategory(
                    name="reproduction",
                    labels=["mating", "courtship", "copulatory_position"],
                ),
                "maternal_behavior": BehaviorCategory(
                    name="maternal_behavior",
                    labels=["egg_carrying", "egg_fanning", "juvenile_release"],
                ),
                "abnormal_behavior": BehaviorCategory(
                    name="abnormal_behavior",
                    labels=["abnormal_inactivity", "disorientation", "molting_failure"],
                ),
                "unknown": BehaviorCategory(
                    name="unknown",
                    labels=["unknown"],
                ),
            },
        )

    def extract_custom_features(self, tracks: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Redclaw-specific interaction metrics (e.g. inter-individual distance, claw orientation)."""
        return {
            "species": "cherax_quadricarinatus",
            "track_count": len(tracks),
        }
