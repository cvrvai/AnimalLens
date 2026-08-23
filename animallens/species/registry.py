"""
Species registry for discovering and instantiating species adapters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Type
from animallens.core.exceptions import SpeciesNotFoundError
from animallens.species.base import SpeciesAdapter
from animallens.species.redclaw.adapter import RedclawAdapter


class SpeciesRegistry:
    """Registry for managing and discovering species adapters."""

    def __init__(self) -> None:
        self._adapters: Dict[str, SpeciesAdapter] = {}
        self._classes: Dict[str, Type[SpeciesAdapter]] = {}
        self._aliases: Dict[str, str] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register built-in species adapters."""
        redclaw = RedclawAdapter()
        self.register_adapter("cherax_quadricarinatus", redclaw)
        self.register_alias("redclaw", "cherax_quadricarinatus")
        self.register_alias("redclaw_crayfish", "cherax_quadricarinatus")
        self.register_alias("crayfish", "cherax_quadricarinatus")

    def register_adapter(self, species_id: str, adapter: SpeciesAdapter) -> None:
        """Register an instantiated SpeciesAdapter."""
        canonical_id = species_id.strip().lower().replace(" ", "_").replace("-", "_")
        self._adapters[canonical_id] = adapter

    def register_alias(self, alias: str, canonical_id: str) -> None:
        """Register an alias for a species id."""
        clean_alias = alias.strip().lower().replace(" ", "_").replace("-", "_")
        clean_target = canonical_id.strip().lower().replace(" ", "_").replace("-", "_")
        self._aliases[clean_alias] = clean_target

    def get(self, identifier: str) -> SpeciesAdapter:
        """Retrieve a species adapter by id or alias."""
        clean_id = identifier.strip().lower().replace(" ", "_").replace("-", "_")
        target_id = self._aliases.get(clean_id, clean_id)

        if target_id in self._adapters:
            return self._adapters[target_id]

        # Check if directory exists in species folder for dynamic discovery
        species_dir = Path(__file__).parent / target_id
        if species_dir.exists() and (species_dir / "config.yaml").exists():
            adapter = RedclawAdapter(directory=species_dir)
            self._adapters[target_id] = adapter
            return adapter

        available = list(self._adapters.keys()) + list(self._aliases.keys())
        raise SpeciesNotFoundError(
            f"Species '{identifier}' not found. Available species: {sorted(set(available))}"
        )

    def list_species(self) -> List[Dict[str, str]]:
        """List all registered canonical species."""
        result = []
        for sp_id, adapter in self._adapters.items():
            cfg = adapter.config
            result.append({
                "id": cfg.id,
                "name": cfg.name,
                "scientific_name": cfg.scientific_name,
                "default_model": cfg.default_model,
                "taxonomy_version": cfg.taxonomy_version,
            })
        return result


# Global singleton instance
species_registry = SpeciesRegistry()
