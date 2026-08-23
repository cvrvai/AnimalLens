"""
Video Source for recorded video files (MP4, AVI, MKV, MOV).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple
from PIL import Image
from animallens.core.exceptions import SourceError
from animallens.core.schemas import SourceType
from animallens.sources.base import BaseSource

logger = logging.getLogger(__name__)


class VideoSource(BaseSource):
    """Source adapter for recorded video files."""

    def __init__(
        self,
        file_path: str | Path,
        sample_fps: Optional[float] = None,
        max_duration_seconds: Optional[float] = None,
    ) -> None:
        path = Path(file_path)
        super().__init__(uri=str(path), source_type=SourceType.VIDEO)
        self.file_path = path
        self.sample_fps = sample_fps
        self.max_duration_seconds = max_duration_seconds
        self._cv2 = None

        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            logger.debug("OpenCV not installed; video reading will run in simulation/fallback mode.")

    def __iter__(self) -> Iterator[Tuple[float, Any]]:
        # If real video file exists and cv2 is installed, use cv2
        if self._cv2 is not None and self.file_path.exists():
            cap = self._cv2.VideoCapture(str(self.file_path))
            if not cap.isOpened():
                raise SourceError(f"Cannot open video file: {self.file_path}")

            native_fps = cap.get(self._cv2.CAP_PROP_FPS) or 30.0
            self.fps = native_fps
            self.width = int(cap.get(self._cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            self.height = int(cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT) or 720)

            frame_idx = 0
            stride = max(1, int(round(native_fps / self.sample_fps))) if self.sample_fps else 1

            try:
                while True:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break

                    if frame_idx % stride == 0:
                        timestamp = frame_idx / native_fps
                        if self.max_duration_seconds and timestamp > self.max_duration_seconds:
                            break

                        # Convert BGR to RGB PIL Image
                        frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)
                        yield (timestamp, pil_img)

                    frame_idx += 1
            finally:
                cap.release()
        else:
            # Synthetic / fallback frames iterator for development or when cv2 is not present
            logger.info(f"Running video analysis in development stream mode for '{self.file_path}'")
            total_duration = self.max_duration_seconds or 10.0
            step_fps = self.sample_fps or 5.0
            total_steps = int(total_duration * step_fps)

            # Create standard blank frame
            dummy_frame = Image.new("RGB", (640, 480), color=(20, 30, 40))
            for i in range(total_steps):
                ts = i / step_fps
                yield (ts, dummy_frame)
