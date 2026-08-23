"""
Input sources module for AnimalLens.
"""
from animallens.sources.base import BaseSource
from animallens.sources.image import ImageSource
from animallens.sources.video import VideoSource
from animallens.sources.stream import StreamSource

__all__ = [
    "BaseSource",
    "ImageSource",
    "VideoSource",
    "StreamSource",
]
