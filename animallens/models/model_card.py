"""
Automated Model Card Generator for Hugging Face Hub distribution.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from animallens.models.hub import HubModelArtifact


class ModelCardGenerator:
    """Generates standardized Hugging Face model cards (README.md) with metadata YAML header."""

    @staticmethod
    def generate(artifact: HubModelArtifact) -> str:
        """Generate markdown card text conforming to HF model card standards."""
        slug_scientific = artifact.scientific_name.lower().replace(" ", "_")
        card = f"""---
language:
- en
license: apache-2.0
tags:
- animallens
- computer-vision
- ethology
- {slug_scientific}
pipeline_tag: {artifact.pipeline_tag}
---

# {artifact.name}

## Model Overview
- **Species**: {artifact.species} (*{artifact.scientific_name}*)
- **Task**: {artifact.task}
- **Version**: {artifact.version}
- **Pipeline Tag**: {artifact.pipeline_tag}
- **Framework**: AnimalLens Open Intelligence Platform (PyTorch + BoT-SORT Kalman kinematics)

## Ethological Methodology
This model architecture incorporates focal animal and continuous sampling methodologies formalized by Altmann (1974), ensuring robust behavioral and spatial state tracking without temporal leakage.

## Description
{artifact.description}

## Usage in AnimalLens
```python
from animallens import AnimalLens

lens = AnimalLens(species="{artifact.name.split('-')[0]}", model_name="{artifact.name}")
result = lens.analyze("sample_media.mp4")
print(result.format_timeline_text())
```
"""
        return card

    @classmethod
    def write_to_file(cls, directory: Path, artifact: HubModelArtifact) -> Path:
        """Generate and write README.md model card into directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        dest = directory / "README.md"
        content = cls.generate(artifact)
        dest.write_text(content, encoding="utf-8")
        return dest
