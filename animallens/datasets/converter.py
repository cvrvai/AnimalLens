"""
Dataset Format Converter for Computer Vision & Behavior Datasets.
Supports conversion between Label Studio, CVAT, COCO, and YOLOv8 format.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
from pydantic import BaseModel


class BBoxConverter:
    """Utilities for converting between normalized and pixel coordinates across formats."""

    @staticmethod
    def xyxy_to_yolo(
        x_min: float, y_min: float, x_max: float, y_max: float
    ) -> Tuple[float, float, float, float]:
        """
        Convert normalized [x_min, y_min, x_max, y_max] to YOLO [x_center, y_center, width, height].
        """
        w = max(0.0, min(1.0, x_max - x_min))
        h = max(0.0, min(1.0, y_max - y_min))
        x_c = min(1.0, max(0.0, x_min + w / 2.0))
        y_c = min(1.0, max(0.0, y_min + h / 2.0))
        return (round(x_c, 6), round(y_c, 6), round(w, 6), round(h, 6))

    @staticmethod
    def yolo_to_xyxy(
        x_c: float, y_c: float, w: float, h: float
    ) -> Tuple[float, float, float, float]:
        """
        Convert YOLO [x_center, y_center, width, height] to normalized [x_min, y_min, x_max, y_max].
        """
        x_min = max(0.0, x_c - w / 2.0)
        y_min = max(0.0, y_c - h / 2.0)
        x_max = min(1.0, x_c + w / 2.0)
        y_max = min(1.0, y_c + h / 2.0)
        return (round(x_min, 6), round(y_min, 6), round(x_max, 6), round(y_max, 6))


class DatasetExporter:
    """Exports structured datasets to YOLOv8 dataset folders and dataset.yaml."""

    @staticmethod
    def generate_yolo_yaml(
        dataset_dir: Union[str, Path],
        class_names: List[str],
        train_path: str = "images/train",
        val_path: str = "images/val",
        test_path: Optional[str] = "images/test",
    ) -> str:
        """
        Generate YOLOv8 dataset.yaml content.
        """
        data = {
            "path": str(Path(dataset_dir).as_posix()),
            "train": train_path,
            "val": val_path,
            "names": {i: name for i, name in enumerate(class_names)},
        }
        if test_path:
            data["test"] = test_path

        return yaml.dump(data, sort_keys=False)

    @staticmethod
    def export_yolo_labels(
        annotations: List[Dict[str, Any]],
        output_txt_file: Union[str, Path],
        class_mapping: Dict[str, int],
    ) -> None:
        """
        Write a YOLO format .txt label file.
        Each line: <class_id> <x_center> <y_center> <width> <height>
        """
        lines = []
        for ann in annotations:
            cls_name = ann.get("class_name", "")
            cls_id = class_mapping.get(cls_name, 0)
            bbox = ann.get("bbox", [0.0, 0.0, 1.0, 1.0])
            x_c, y_c, w, h = BBoxConverter.xyxy_to_yolo(bbox[0], bbox[1], bbox[2], bbox[3])
            lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        Path(output_txt_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_txt_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n" if lines else "")
