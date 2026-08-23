"""
Rolling video buffer for real-time temporal behavior classification.
Maintains a rolling window of recent video frames and perception metadata.
"""
from __future__ import annotations

from collections import deque
from typing import Any, List, Optional, Tuple
from animallens.perception.base import FramePerceptionData


class RollingVideoBuffer:
    """
    In-memory rolling buffer for real-time video processing.
    Retains the last N seconds of raw frames and perception states for temporal analysis.
    """

    def __init__(self, capacity_seconds: float = 15.0, fps: float = 30.0) -> None:
        self.capacity_seconds = capacity_seconds
        self.fps = fps
        self.max_frames = max(10, int(capacity_seconds * fps))
        self._frames: deque[Tuple[float, Any]] = deque(maxlen=self.max_frames)
        self._perception_data: deque[FramePerceptionData] = deque(maxlen=self.max_frames)

    def push(
        self,
        timestamp: float,
        frame: Optional[Any],
        perception_data: FramePerceptionData,
    ) -> None:
        """Push a new frame and its perception data into the rolling window."""
        self._frames.append((timestamp, frame))
        self._perception_data.append(perception_data)

    def get_window(self, duration_seconds: Optional[float] = None) -> List[FramePerceptionData]:
        """
        Get recent perception data covering the specified duration up to the latest frame.
        """
        if not self._perception_data:
            return []

        if duration_seconds is None or duration_seconds >= self.capacity_seconds:
            return list(self._perception_data)

        latest_time = self._perception_data[-1].timestamp
        start_cutoff = latest_time - duration_seconds

        window = [d for d in self._perception_data if d.timestamp >= start_cutoff]
        return window

    def get_recent_frames(self, count: int = 5) -> List[Any]:
        """Retrieve recent raw image frames for reasoning or visualization."""
        if not self._frames:
            return []
        items = list(self._frames)[-count:]
        return [f[1] for f in items if f[1] is not None]

    def clear(self) -> None:
        """Clear the rolling buffer."""
        self._frames.clear()
        self._perception_data.clear()

    @property
    def current_duration(self) -> float:
        """Current temporal span in seconds stored in buffer."""
        if len(self._perception_data) < 2:
            return 0.0
        return self._perception_data[-1].timestamp - self._perception_data[0].timestamp

    def __len__(self) -> int:
        return len(self._perception_data)
