"""
Image Source for single or batch images.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple, Union
from PIL import Image
import httpx
from animallens.core.exceptions import SourceError
from animallens.core.schemas import SourceType
from animallens.sources.base import BaseSource


class ImageSource(BaseSource):
    """Source adapter for single static images (file path, URL, Pillow image, or bytes)."""

    def __init__(self, target: Union[str, Path, Image.Image, bytes]) -> None:
        uri_str = str(target) if isinstance(target, (str, Path)) else "memory://image"
        super().__init__(uri=uri_str, source_type=SourceType.IMAGE, fps=1.0)
        self.target = target
        self._image: Optional[Image.Image] = None

    def _load_image(self) -> Image.Image:
        if self._image is not None:
            return self._image

        try:
            if isinstance(self.target, Image.Image):
                self._image = self.target
            elif hasattr(self.target, "shape") and hasattr(self.target, "dtype"):  # numpy array
                self._image = Image.fromarray(self.target)
            elif isinstance(self.target, bytes):
                self._image = Image.open(io.BytesIO(self.target)).convert("RGB")
            elif isinstance(self.target, (str, Path)):
                path_str = str(self.target)
                if path_str.startswith(("http://", "https://")):
                    res = httpx.get(path_str, timeout=15.0)
                    res.raise_for_status()
                    self._image = Image.open(io.BytesIO(res.content)).convert("RGB")
                else:
                    path = Path(path_str)
                    if not path.exists():
                        raise SourceError(f"Image file not found: {path_str}")
                    self._image = Image.open(path).convert("RGB")
            else:
                raise SourceError(f"Unsupported image input type: {type(self.target)}")

            self.width, self.height = self._image.size
            return self._image
        except Exception as e:
            if isinstance(e, SourceError):
                raise
            raise SourceError(f"Failed to load image source '{self.uri}': {e}") from e

    def __iter__(self) -> Iterator[Tuple[float, Any]]:
        img = self._load_image()
        yield (0.0, img)
