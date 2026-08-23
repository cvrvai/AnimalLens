"""
Unit Tests for Domestic Pig / Swine (Sus scrofa domesticus) Ethology Adapter (Phase 14).
"""
import pytest
from animallens.core.schemas import BoundingBox
from animallens.perception.base import TrackState
from animallens.sdk import AnimalLens
from animallens.species.pig.adapter import PigAdapter
from animallens.species.registry import species_registry


def test_pig_adapter_registration():
    adapter = species_registry.get("pig")
    assert isinstance(adapter, PigAdapter)
    assert adapter.config.id == "sus_scrofa_domesticus"
    assert adapter.config.name == "Domestic Pig"

    # Verify aliases
    assert species_registry.get("swine").config.id == "sus_scrofa_domesticus"
    assert species_registry.get("hog").config.id == "sus_scrofa_domesticus"
    assert species_registry.get("sow").config.id == "sus_scrofa_domesticus"


def test_pig_taxonomy_categories():
    adapter = PigAdapter()
    tax = adapter.taxonomy

    assert "posture" in tax.categories
    assert "locomotion" in tax.categories
    assert "feeding" in tax.categories
    assert "aggression" in tax.categories

    posture_labels = tax.categories["posture"].labels
    assert "sternal_recumbency" in posture_labels
    assert "lateral_recumbency" in posture_labels
    assert "huddling_cold_stress" in posture_labels

    aggression_labels = tax.categories["aggression"].labels
    assert "tail_biting_cannibalism" in aggression_labels
    assert "head_knocking_aggression" in aggression_labels


def test_pig_recumbency_classification():
    adapter = PigAdapter()

    # Lateral recumbency (lying flat on side, wide aspect ratio)
    bbox_lateral = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.35)  # w=0.4, h=0.15 -> ratio=2.66
    assert adapter.classify_recumbency(bbox_lateral, velocity=0.0) == "lateral_recumbency"

    # Sternal recumbency (chest down, moderate aspect ratio)
    bbox_sternal = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.4, y_max=0.4)  # w=0.3, h=0.2 -> ratio=1.5
    assert adapter.classify_recumbency(bbox_sternal, velocity=0.0) == "sternal_recumbency"

    # Standing (upright, squarish aspect ratio)
    bbox_standing = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.3)  # w=0.2, h=0.2 -> ratio=1.0, y_max=0.3
    assert adapter.classify_recumbency(bbox_standing, velocity=0.0) == "standing"

    # Rooting / Nesting (head lowered to pen floor grating, y_max > 0.50)
    bbox_nesting = BoundingBox(x_min=0.25, y_min=0.16, x_max=0.60, y_max=0.57)
    assert adapter.classify_recumbency(bbox_nesting, velocity=0.0) == "rooting_nesting"


def test_pig_huddling_cold_stress_index():
    adapter = PigAdapter()

    # Case A: Pigs tightly clustered together in pen (cold thermal stress)
    tight_tracks = [
        TrackState(track_id=1, current_bbox=BoundingBox(x_min=0.40, y_min=0.40, x_max=0.50, y_max=0.50)),
        TrackState(track_id=2, current_bbox=BoundingBox(x_min=0.42, y_min=0.42, x_max=0.52, y_max=0.52)),
        TrackState(track_id=3, current_bbox=BoundingBox(x_min=0.44, y_min=0.41, x_max=0.54, y_max=0.51)),
    ]
    huddle_high = adapter.compute_huddling_index(tight_tracks)
    assert huddle_high > 0.70

    # Case B: Pigs evenly spread out across pen (normal thermal comfort)
    spread_tracks = [
        TrackState(track_id=1, current_bbox=BoundingBox(x_min=0.05, y_min=0.05, x_max=0.15, y_max=0.15)),
        TrackState(track_id=2, current_bbox=BoundingBox(x_min=0.80, y_min=0.80, x_max=0.90, y_max=0.90)),
    ]
    huddle_low = adapter.compute_huddling_index(spread_tracks)
    assert huddle_low < 0.20


def test_pig_animallens_sdk_execution():
    import numpy as np

    lens = AnimalLens(species="pig")
    assert lens.species_name == "Domestic Pig"

    dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    res = lens.analyze_image(dummy_frame)
    assert res.species == "Domestic Pig"
    assert len(res.behaviors) > 0
