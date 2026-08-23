"""
Unit tests for Hugging Face Hub Distribution, SHA256 Integrity Verification, and Model Registry.
"""
from pathlib import Path
import pytest
from animallens.models.hub import OFFICIAL_HUB_CATALOGUE, HuggingFaceModelHub
from animallens.models.registry import ModelRegistry


def test_hub_catalogue_listing(tmp_path):
    """Verify listing official Hugging Face models."""
    hub = HuggingFaceModelHub(cache_dir=tmp_path)
    models = hub.list_official_models()

    assert len(models) >= 3
    names = [m.name for m in models]
    assert "redclaw-behavior-v1" in names
    assert "redclaw-yolov8n-v1" in names
    assert "pig-posture-v1" in names


def test_hub_pull_and_integrity_verification(tmp_path):
    """Verify downloading, caching, and SHA256 integrity verification."""
    hub = HuggingFaceModelHub(cache_dir=tmp_path)

    progress_messages = []
    installed_dir = hub.pull_model(
        "redclaw-yolov8n-v1",
        progress_callback=lambda msg, frac: progress_messages.append(msg),
    )

    assert installed_dir.exists()
    assert (installed_dir / "redclaw-yolov8n-v1.pt").exists()
    assert (installed_dir / "manifest.json").exists()
    assert len(progress_messages) >= 2

    # Check that model shows as installed
    models = hub.list_official_models()
    redclaw_model = next(m for m in models if m.name == "redclaw-yolov8n-v1")
    assert redclaw_model.is_installed is True


def test_model_registry_integration(tmp_path):
    """Verify ModelRegistry pulls from Hugging Face Hub."""
    reg = ModelRegistry(models_dir=tmp_path / "models")
    reg.hub = HuggingFaceModelHub(cache_dir=tmp_path)

    avail = reg.list_available()
    assert len(avail) >= 3

    pulled_path = reg.pull("redclaw-behavior-v1")
    assert pulled_path.exists()


def test_canine_models_in_catalogue(tmp_path):
    """Verify canine pose, detector, reid, and ethogram models are in catalogue."""
    hub = HuggingFaceModelHub(cache_dir=tmp_path)
    models = hub.list_official_models()
    names = [m.name for m in models]

    assert "canine-pose-v1" in names
    assert "canine-detector-v1" in names
    assert "canine-reid-v1" in names
    assert "canine-ethogram-stgcn-v1" in names


def test_model_card_generation(tmp_path):
    """Verify automated Hugging Face README.md model card generation."""
    from animallens.models.model_card import ModelCardGenerator

    art = OFFICIAL_HUB_CATALOGUE["canine-pose-v1"]
    card_text = ModelCardGenerator.generate(art)

    assert "pipeline_tag: keypoint-detection" in card_text
    assert "canine-pose-v1" in card_text
    assert "canis_lupus_familiaris" in card_text
    assert "Altmann" in card_text
    assert "BoT-SORT" in card_text

    card_path = ModelCardGenerator.write_to_file(tmp_path, art)
    assert card_path.exists()
    assert card_path.name == "README.md"
    assert "keypoint-detection" in card_path.read_text(encoding="utf-8")
