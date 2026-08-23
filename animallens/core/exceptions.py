"""
Standard exceptions hierarchy for AnimalLens.
"""
from __future__ import annotations


class AnimalLensError(Exception):
    """Base exception for all AnimalLens errors."""
    pass


class SpeciesNotFoundError(AnimalLensError):
    """Raised when a requested species adapter or taxonomy is not found."""
    pass


class ModelNotFoundError(AnimalLensError):
    """Raised when a specified model weights package or manifest is not found."""
    pass


class SourceError(AnimalLensError):
    """Raised when an input source (image, video, RTSP, webcam) fails to load or stream."""
    pass


class PipelineError(AnimalLensError):
    """Raised when an error occurs during perception pipeline processing."""
    pass


class ReasoningError(AnimalLensError):
    """Raised when an error occurs during LLM reasoning inference."""
    pass


class RegistryError(AnimalLensError):
    """Raised when model registration, pulling, or discovery fails."""
    pass
