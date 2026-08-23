"""
Automated Video Dataset Builder & Keyframe Ingestion Pipeline for AnimalLens.
Slices raw video streams into training/validation splits with YOLOv8 dataset structuring.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

logger = logging.getLogger(__name__)


class VideoDatasetBuilder:
    """
    Builds annotated training/validation datasets directly from video clips.
    """

    def __init__(
        self,
        output_dir: str | Path = "datasets/custom_training",
        val_split: float = 0.2,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.val_split = val_split
        self.images_train = self.output_dir / "images" / "train"
        self.images_val = self.output_dir / "images" / "val"
        self.labels_train = self.output_dir / "labels" / "train"
        self.labels_val = self.output_dir / "labels" / "val"

        self._init_dirs()

    def _init_dirs(self) -> None:
        """Create standard YOLO dataset directory hierarchy."""
        for d in [self.images_train, self.images_val, self.labels_train, self.labels_val]:
            d.mkdir(parents=True, exist_ok=True)

    def extract_keyframes(
        self,
        video_path: str | Path,
        sample_fps: float = 2.0,
        max_frames: Optional[int] = None,
    ) -> List[Path]:
        """
        Extracts uniform keyframes from video and partitions into train/val splits.
        """
        import cv2

        vid = Path(video_path)
        if not vid.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(fps / sample_fps))

        extracted_paths: List[Path] = []
        frame_idx = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Determine split
                is_val = (saved_count % int(1.0 / self.val_split) == 0) if self.val_split > 0 else False
                target_dir = self.images_val if is_val else self.images_train

                frame_name = f"{vid.stem}_f{frame_idx:06d}.jpg"
                out_path = target_dir / frame_name
                cv2.imwrite(str(out_path), frame)
                extracted_paths.append(out_path)
                saved_count += 1

                if max_frames and saved_count >= max_frames:
                    break

            frame_idx += 1

        cap.release()
        logger.info(f"Extracted {saved_count} frames from {vid.name} into {self.output_dir}")
        return extracted_paths

    def generate_pseudo_labels(
        self,
        classes: List[str] = ["dog"],
        conf_threshold: float = 0.35,
    ) -> int:
        """
        Automatically generates initial pseudo-labels using the base detector to accelerate training.
        """
        from animallens.perception.models.yolov8_detector import YOLOv8Detector
        import cv2

        detector = YOLOv8Detector(classes=classes, conf_threshold=conf_threshold)
        total_labeled = 0

        for split in ["train", "val"]:
            img_dir = self.output_dir / "images" / split
            lbl_dir = self.output_dir / "labels" / split

            for img_path in img_dir.glob("*.jpg"):
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue

                res = detector.detect(frame, confidence_threshold=conf_threshold)
                txt_path = lbl_dir / f"{img_path.stem}.txt"

                lines = []
                for box in res.bboxes:
                    # YOLO format: <class_id> <x_center> <y_center> <width> <height>
                    cls_id = 0
                    lines.append(
                        f"{cls_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"
                    )

                txt_path.write_text("\n".join(lines), encoding="utf-8")
                total_labeled += 1

        logger.info(f"Generated pseudo-labels for {total_labeled} images.")
        return total_labeled

    def write_yaml_config(
        self,
        species_name: str = "canine",
        classes: List[str] = ["dog"],
    ) -> Path:
        """
        Writes the dataset.yaml configuration required for Ultralytics YOLO training.
        """
        data = {
            "path": str(self.output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(classes)},
        }

        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info(f"Wrote dataset config to: {yaml_path}")
        return yaml_path
