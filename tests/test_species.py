"""
Unit tests for species adapters, registry, and taxonomies.
"""
import pytest
from animallens.core.exceptions import SpeciesNotFoundError
from animallens.species.base import BehaviorTaxonomy
from animallens.species.redclaw.adapter import RedclawAdapter
from animallens.species.registry import SpeciesRegistry


def test_redclaw_adapter_taxonomy():
    adapter = RedclawAdapter()
    assert adapter.config.id == "cherax_quadricarinatus"
    assert adapter.config.name == "Redclaw Crayfish"

    tax = adapter.taxonomy
    assert "locomotion" in tax.categories
    assert "reproduction" in tax.categories
    assert "aggression" in tax.categories

    # Test category lookup
    assert tax.get_category_for_label("mating") == "reproduction"
    assert tax.get_category_for_label("normal_movement") == "locomotion"
    assert tax.get_category_for_label("foraging") == "feeding"
    assert tax.get_category_for_label("unknown_action") == "unknown"


def test_species_registry_lookups():
    reg = SpeciesRegistry()
    adapter1 = reg.get("redclaw")
    adapter2 = reg.get("cherax_quadricarinatus")
    adapter3 = reg.get("crayfish")

    assert adapter1.config.id == adapter2.config.id == adapter3.config.id

    with pytest.raises(SpeciesNotFoundError):
        reg.get("non_existent_species_xyz")


def test_species_classify_behavior():
    adapter = RedclawAdapter()
    beh_info = adapter.classify_behavior(label="mating", confidence=0.92)
    assert beh_info.category == "reproduction"
    assert beh_info.label == "mating"
    assert beh_info.confidence == 0.92
    assert beh_info.is_uncertain is False

    # Low confidence behavior should be flagged uncertain
    beh_uncertain = adapter.classify_behavior(label="mating", confidence=0.30)
    assert beh_uncertain.is_uncertain is True
