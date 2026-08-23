"""
Base Source interface for ingesting images, video files, webcams, and RTSP streams.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterator, Optional, Tuple
from animallens.core.schemas import SourceInfo, SourceType


class BaseSource(ABC):
    """
    Abstract source for media frames.
    Yields (timestamp_seconds, frame) tuples.
    """

    def __init__(self, uri: str, source_type: SourceType, fps: float = 30.0) -> None:
        self.uri = uri
        self.source_type = source_type
        self.fps = fps
        self.width: int = 1280
        self.height: int = 720
        self.camera_id: Optional[str] = None

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[float, Any]]:
        """Synchronously yield (timestamp_seconds, frame) tuples."""
        pass

    async def __aiter__(self) -> AsyncIterator[Tuple[float, Any]]:
        """Asynchronously yield (timestamp_seconds, frame) tuples."""
        for item in self:
            yield item

    def get_source_info(self) -> SourceInfo:
        """Construct standard SourceInfo metadata."""
        return SourceInfo(
            type=self.source_type,
            uri=self.uri,
            camera_id=self.camera_id,
            fps=self.fps,
            resolution=(self.width, self.height),
        )

    def close(self) -> None:
        """Release underlying system resources (cameras, files)."""
        pass
