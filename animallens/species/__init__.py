"""
Species adapters and taxonomy module for AnimalLens.
"""
from animallens.species.base import (
    BehaviorCategory,
    BehaviorTaxonomy,
    SpeciesAdapter,
    SpeciesConfig,
)
from animallens.species.registry import SpeciesRegistry, species_registry
from animallens.species.redclaw.adapter import RedclawAdapter

__all__ = [
    "BehaviorCategory",
    "BehaviorTaxonomy",
    "SpeciesAdapter",
    "SpeciesConfig",
    "SpeciesRegistry",
    "species_registry",
    "RedclawAdapter",
]
