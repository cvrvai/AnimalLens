"""
Live Camera & RTSP Stream Source.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Iterator, Optional, Tuple, Union
from PIL import Image
from animallens.core.exceptions import SourceError
from animallens.core.schemas import SourceType
from animallens.sources.base import BaseSource

logger = logging.getLogger(__name__)


class StreamSource(BaseSource):
    """Source adapter for live webcams and RTSP network streams."""

    def __init__(
        self,
        endpoint: Union[str, int],
        target_fps: float = 15.0,
        camera_id: Optional[str] = None,
    ) -> None:
        source_type = SourceType.WEBCAM if isinstance(endpoint, int) or (isinstance(endpoint, str) and endpoint.isdigit()) else SourceType.RTSP
        super().__init__(uri=str(endpoint), source_type=source_type, fps=target_fps)
        self.endpoint = int(endpoint) if isinstance(endpoint, str) and endpoint.isdigit() else endpoint
        self.camera_id = camera_id or f"CAM-{str(endpoint)}"
        self._cv2 = None
        self._running = False

        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            logger.debug("OpenCV not installed; stream will operate in synthetic mode.")

    def __iter__(self) -> Iterator[Tuple[float, Any]]:
        self._running = True
        start_time = time.time()

        if self._cv2 is not None:
            cap = self._cv2.VideoCapture(self.endpoint)
            if not cap.isOpened():
                raise SourceError(f"Could not connect to camera/stream: {self.endpoint}")

            self.width = int(cap.get(self._cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            self.height = int(cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT) or 720)

            try:
                while self._running:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        time.sleep(0.01)
                        continue

                    current_ts = time.time() - start_time
                    frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    yield (current_ts, pil_img)
            finally:
                cap.release()
        else:
            # Synthetic camera stream for testing / non-cv2 environments
            dummy_frame = Image.new("RGB", (640, 480), color=(10, 20, 30))
            frame_idx = 0
            while self._running:
                ts = time.time() - start_time
                yield (ts, dummy_frame)
                frame_idx += 1
                time.sleep(1.0 / self.fps)

    def close(self) -> None:
        self._running = False
