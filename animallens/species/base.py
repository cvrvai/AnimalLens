"""
Base abstractions for species configuration, taxonomy, and adapters.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field
from animallens.core.schemas import BehaviorInfo, SpeciesInfo


class BehaviorCategory(BaseModel):
    """A category in the behavior hierarchy containing specific sub-labels."""
    name: str
    description: str = ""
    labels: List[str] = Field(default_factory=list)


class BehaviorTaxonomy(BaseModel):
    """Hierarchical taxonomy of animal behaviors."""
    version: str = "1.0.0"
    categories: Dict[str, BehaviorCategory] = Field(default_factory=dict)

    def get_all_labels(self) -> List[str]:
        """Flattened list of all defined labels."""
        labels = []
        for cat in self.categories.values():
            labels.extend(cat.labels)
        return list(set(labels))

    def get_category_for_label(self, label: str) -> Optional[str]:
        """Find the parent category for a given behavior label."""
        for cat_name, cat in self.categories.items():
            if label in cat.labels or label == cat_name:
                return cat_name
        return "unknown"


class SpeciesConfig(BaseModel):
    """Configuration metadata for a species."""
    id: str
    name: str
    scientific_name: str
    taxonomy_version: str = "1.0.0"
    default_model: str = "redclaw-behavior-v1"
    detection_threshold: float = 0.50
    min_track_length: int = 5
    uncertainty_threshold: float = 0.45
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SpeciesAdapter(ABC):
    """
    Abstract adapter for a specific animal species.
    Encapsulates biological taxonomy, physical thresholds, and domain-specific rules.
    """

    def __init__(self, directory: Optional[Path] = None) -> None:
        self.directory = directory
        self._config: Optional[SpeciesConfig] = None
        self._taxonomy: Optional[BehaviorTaxonomy] = None
        if directory:
            self._load_from_dir(directory)

    def _load_from_dir(self, dir_path: Path) -> None:
        """Load config.yaml and taxonomy.yaml from species directory if present."""
        config_file = dir_path / "config.yaml"
        taxonomy_file = dir_path / "taxonomy.yaml"

        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f)
                self._config = SpeciesConfig(**raw_cfg)

        if taxonomy_file.exists():
            with open(taxonomy_file, "r", encoding="utf-8") as f:
                raw_tax = yaml.safe_load(f)
                categories = {}
                for cat_name, details in raw_tax.get("categories", {}).items():
                    categories[cat_name] = BehaviorCategory(
                        name=cat_name,
                        description=details.get("description", ""),
                        labels=details.get("labels", []),
                    )
                self._taxonomy = BehaviorTaxonomy(
                    version=raw_tax.get("version", "1.0.0"),
                    categories=categories,
                )

    @property
    def config(self) -> SpeciesConfig:
        if not self._config:
            self._config = self.default_config()
        return self._config

    @property
    def taxonomy(self) -> BehaviorTaxonomy:
        if not self._taxonomy:
            self._taxonomy = self.default_taxonomy()
        return self._taxonomy

    @abstractmethod
    def default_config(self) -> SpeciesConfig:
        """Provide fallback configuration if config.yaml is absent."""
        pass

    @abstractmethod
    def default_taxonomy(self) -> BehaviorTaxonomy:
        """Provide fallback taxonomy if taxonomy.yaml is absent."""
        pass

    def get_species_info(self) -> SpeciesInfo:
        """Generate standard SpeciesInfo schema."""
        cfg = self.config
        return SpeciesInfo(
            id=cfg.id,
            name=cfg.name,
            scientific_name=cfg.scientific_name,
            taxonomy_version=cfg.taxonomy_version,
        )

    def classify_behavior(
        self,
        label: str,
        confidence: float,
        secondary_labels: Optional[List[Dict[str, float]]] = None,
    ) -> BehaviorInfo:
        """Construct validated BehaviorInfo respecting confidence and unknown thresholds."""
        is_uncertain = confidence < self.config.uncertainty_threshold or label == "unknown"
        category = self.taxonomy.get_category_for_label(label) or "unknown"
        if is_uncertain and label != "unknown":
            # If uncertain, mark uncertainty flag
            pass

        return BehaviorInfo(
            category=category,
            label=label,
            confidence=round(confidence, 4),
            secondary_labels=secondary_labels or [],
            is_uncertain=is_uncertain,
        )

    def extract_custom_features(self, tracks: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Optional hook for species-specific kinematic or morphological feature extraction."""
        return {}
