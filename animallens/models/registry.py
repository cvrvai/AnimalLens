"""
Model Registry for AnimalLens model management and Hub integration.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from animallens.models.hub import OFFICIAL_HUB_CATALOGUE, HuggingFaceModelHub, HubModelArtifact


class ModelRegistry:
    """Manages locally installed and official Hugging Face models."""

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.hub = HuggingFaceModelHub(cache_dir=self.models_dir)

    def list_available(self) -> List[Dict[str, Any]]:
        """List all models in the official catalogue with installation status."""
        models = self.hub.list_official_models()
        return [m.to_dict() for m in models]

    def list_installed(self) -> List[Dict[str, Any]]:
        """List only models currently installed locally in models directory."""
        all_models = self.hub.list_official_models()
        installed = []
        for m in all_models:
            model_dir = self.hub.cache_dir / m.name
            target_pt = model_dir / (m.filename or f"{m.name}.pt")
            if target_pt.exists():
                d = m.to_dict()
                d["is_installed"] = True
                d["installed_path"] = str(target_pt)
                installed.append(d)

        # Also search for custom trained models in models/trained
        trained_dir = self.models_dir / "trained"
        if trained_dir.exists():
            for pt in trained_dir.glob("**/weights/best.pt"):
                exp_name = pt.parent.parent.name
                installed.append({
                    "name": exp_name,
                    "species": "custom",
                    "task": "object-detection",
                    "version": "1.0.0",
                    "is_installed": True,
                    "installed_path": str(pt),
                    "description": f"Custom fine-tuned checkpoint: {exp_name}",
                })

        return installed

    def pull(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Path:
        """Download or link model package into local registry."""
        installed_dir = self.hub.pull_model(model_name, progress_callback=progress_callback)
        art = OFFICIAL_HUB_CATALOGUE.get(model_name)
        target_file = installed_dir / (art.filename if art and art.filename else f"{model_name}.pt")
        return target_file

    def remove(self, model_name: str) -> bool:
        """Remove a cached model package."""
        target_dir = self.hub.cache_dir / model_name
        if target_dir.exists():
            shutil.rmtree(target_dir)
            return True
        return False

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Resolve path to weights for a model name."""
        # 1. Check direct path
        p = Path(model_name)
        if p.exists():
            return p

        # 2. Check hub cache
        art = OFFICIAL_HUB_CATALOGUE.get(model_name)
        filename = art.filename if art and art.filename else f"{model_name}.pt"
        hub_path = self.hub.cache_dir / model_name / filename
        if hub_path.exists():
            return hub_path

        # 3. Check trained
        trained_path = self.models_dir / "trained" / model_name / "weights" / "best.pt"
        if trained_path.exists():
            return trained_path

        return None


# Global singleton registry instance
model_registry = ModelRegistry()
