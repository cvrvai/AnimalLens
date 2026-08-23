"""
Species adapters and taxonomy module for AnimalLens.
"""
from animallens.species.base import (
    BehaviorCategory,
    BehaviorTaxonomy,
    SpeciesAdapter,
    SpeciesConfig,
)
from animallens.species.dog.adapter import DogAdapter
from animallens.species.redclaw.adapter import RedclawAdapter
from animallens.species.registry import SpeciesRegistry, species_registry

__all__ = [
    "BehaviorCategory",
    "BehaviorTaxonomy",
    "SpeciesAdapter",
    "SpeciesConfig",
    "SpeciesRegistry",
    "species_registry",
    "RedclawAdapter",
    "DogAdapter",
]
