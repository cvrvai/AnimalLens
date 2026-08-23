"""
Configuration management for AnimalLens platform.
"""
from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Global configuration settings for AnimalLens."""

    app_name: str = "AnimalLens"
    app_version: str = "0.1.0"
    debug: bool = Field(default_factory=lambda: os.getenv("ANIMALLENS_DEBUG", "false").lower() in ("true", "1"))

    # Storage paths
    cache_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("ANIMALLENS_CACHE_DIR", str(Path.home() / ".cache" / "animallens"))
        )
    )
    models_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("ANIMALLENS_MODELS_DIR", str(Path.home() / ".cache" / "animallens" / "models"))
        )
    )
    datasets_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("ANIMALLENS_DATASETS_DIR", str(Path.home() / ".cache" / "animallens" / "datasets"))
        )
    )

    # Server settings
    server_host: str = Field(default_factory=lambda: os.getenv("ANIMALLENS_HOST", "0.0.0.0"))
    server_port: int = Field(default_factory=lambda: int(os.getenv("ANIMALLENS_PORT", "8088")))

    # Ollama settings
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_timeout_seconds: float = Field(default=30.0)

    # Video buffer settings
    default_buffer_duration_seconds: float = 15.0
    default_fps: float = 30.0

    # Uncertainty / Active Learning threshold
    uncertainty_threshold: float = Field(default=0.45)

    def ensure_directories(self) -> None:
        """Create cache, model and dataset directories if they do not exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
