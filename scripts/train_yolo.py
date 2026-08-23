"""
YOLOv8 Training Pipeline for Cherax quadricarinatus (Redclaw Crayfish).
Trains YOLOv8 on the anti-leakage partitioned dataset and exports production weights.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys


def train_yolo(
    data_yaml: str = "data/processed_yolo/dataset.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 25,
    imgsz: int = 640,
    batch_size: int = 8,
) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics not yet installed. Please run `pip install ultralytics` first.")
        sys.exit(1)

    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.exists():
        print(f"Error: Dataset configuration not found at {yaml_path}")
        sys.exit(1)

    print("=" * 60)
    print("🦞 AnimalLens YOLOv8 Model Training Pipeline")
    print("=" * 60)
    print(f"Dataset Config: {yaml_path}")
    print(f"Base Weights:   {base_model}")
    print(f"Target Epochs:  {epochs}")
    print(f"Image Size:     {imgsz}x{imgsz}")
    print(f"Batch Size:     {batch_size}")
    print("-" * 60)

    # Initialize model
    model = YOLO(base_model)

    # Start training
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=0,  # Single-process worker for Windows stability
        project="runs/detect",
        name="redclaw_yolov8n",
        exist_ok=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)

    # Output paths
    best_weights = Path("runs/detect/redclaw_yolov8n/weights/best.pt")
    if best_weights.exists():
        target_dir = Path("data/trained_models")
        target_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        saved_path = target_dir / "redclaw-behavior-v1.pt"
        shutil.copy(best_weights, saved_path)
        print(f"Best Weights Exported To: {saved_path.resolve()}")
    else:
        print(f"Model saved in: runs/detect/redclaw_yolov8n/")


if __name__ == "__main__":
    train_yolo()
