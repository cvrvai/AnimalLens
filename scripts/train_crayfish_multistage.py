"""
AnimalLens Crayfish Multi-Stage YOLOv8 Model Training Script.

Trains/fine-tunes YOLOv8 on the merged multi-stage crayfish dataset
(craylet, juvenile, sub_adult, adult) at datasets/crayfish_combined/dataset.yaml.
Exports the trained model checkpoint to models/trained/crayfish_multistage/weights/best.pt
and writes comprehensive validation metrics to models/trained/crayfish_multistage/metrics.json.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("crayfish_multistage_trainer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AnimalLens Crayfish Multi-Stage YOLOv8 Detector"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset.yaml (defaults to datasets/crayfish_combined/dataset.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image resolution (default: 640)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolov8n.pt",
        help="Base YOLO model checkpoint (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('0', 'cuda:0', 'cpu'; defaults to auto-detect)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of dataloader worker processes (0 required for Windows stability)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project output directory (defaults to models/trained)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="crayfish_multistage",
        help="Experiment name (defaults to crayfish_multistage)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Resolve repo root, dataset yaml, project directory, and target experiment dir."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    # Dataset YAML
    if args.data:
        dataset_yaml = Path(args.data).resolve()
    else:
        dataset_yaml = (repo_root / "datasets" / "crayfish_combined" / "dataset.yaml").resolve()

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML configuration not found at: {dataset_yaml}")

    # Project directory
    if args.project:
        project_dir = Path(args.project).resolve()
    else:
        project_dir = (repo_root / "models" / "trained").resolve()

    project_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = project_dir / args.name
    target_weights_dir = exp_dir / "weights"
    target_weights_dir.mkdir(parents=True, exist_ok=True)

    return repo_root, dataset_yaml, project_dir, exp_dir


def parse_results_csv(results_csv_path: Path) -> Dict[str, float]:
    """Parse Ultralytics results.csv for final epoch metrics and losses."""
    metrics: Dict[str, float] = {}
    if not results_csv_path.exists():
        return metrics

    try:
        with open(results_csv_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if len(lines) < 2:
            return metrics

        headers = [h.strip() for h in lines[0].split(",")]
        last_row = [v.strip() for v in lines[-1].split(",")]

        if len(headers) == len(last_row):
            for h, v in zip(headers, last_row):
                try:
                    metrics[h] = float(v)
                except ValueError:
                    pass
    except Exception as e:
        logger.warning(f"Could not parse results.csv: {e}")

    return metrics


def train_crayfish_model(
    dataset_yaml: Path,
    project_dir: Path,
    exp_name: str,
    epochs: int = 5,
    batch: int = 8,
    imgsz: int = 640,
    base_model: str = "yolov8n.pt",
    device: Optional[str] = None,
    workers: int = 0,
    resume: bool = False,
) -> Dict[str, Any]:
    """Execute YOLOv8 training and return training report dictionary."""
    exp_dir = project_dir / exp_name
    weights_dir = exp_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    weights_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from ultralytics import YOLO

        # Device auto-detection
        if device is None:
            device = "0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Target execution device: {device} (CUDA available: {torch.cuda.is_available()})")

        # Handle resume vs fresh training
        if resume and last_pt.exists():
            logger.info(f"Resuming training from checkpoint: {last_pt}")
            model = YOLO(str(last_pt))
            train_kwargs: Dict[str, Any] = {"resume": True, "workers": workers}
        else:
            logger.info(f"Initiating training with base model: {base_model}")
            try:
                model = YOLO(base_model)
            except Exception as e:
                logger.warning(f"Could not load {base_model} ({e}), initializing architecture from yolov8n.yaml")
                model = YOLO("yolov8n.yaml")
            train_kwargs = {
                "data": str(dataset_yaml),
                "epochs": epochs,
                "batch": batch,
                "imgsz": imgsz,
                "device": device,
                "workers": workers,  # workers=0 strictly enforced for Windows stability
                "project": str(project_dir),
                "name": exp_name,
                "exist_ok": True,
                "verbose": True,
            }

        logger.info("=" * 70)
        logger.info("Starting YOLOv8 Multi-Stage Crayfish Training")
        logger.info(f"Dataset YAML:  {dataset_yaml}")
        logger.info(f"Base Model:    {base_model}")
        logger.info(f"Epochs:        {epochs}")
        logger.info(f"Batch Size:    {batch}")
        logger.info(f"Image Size:    {imgsz}")
        logger.info(f"Workers:       {workers}")
        logger.info(f"Project Dir:   {project_dir}")
        logger.info(f"Experiment:    {exp_name}")
        logger.info("=" * 70)

        # 1. Execute training
        train_results = model.train(**train_kwargs)
        logger.info("Training process completed successfully.")

        # Verify best.pt exists
        if not best_pt.exists():
            # Check if saved in an alternate run directory
            candidate_weights = list(project_dir.glob(f"**/{exp_name}*/weights/best.pt"))
            if candidate_weights and candidate_weights[0].exists():
                shutil.copy(candidate_weights[0], best_pt)
                logger.info(f"Copied checkpoint from {candidate_weights[0]} to {best_pt}")

        if not best_pt.exists():
            raise RuntimeError(f"Expected checkpoint not found at: {best_pt}")

        actual_size_bytes = best_pt.stat().st_size
        checkpoint_size_mb = round(actual_size_bytes / (1024 * 1024), 2)
        if checkpoint_size_mb < 1.0:
            raise RuntimeError(f"Checkpoint size {checkpoint_size_mb} MB is too small to be a binary YOLO weights file.")
        logger.info(f"Saved best checkpoint: {best_pt} ({checkpoint_size_mb:.2f} MB)")

        # 2. Validation Run
        logger.info("Running post-training validation on crayfish dataset...")
        val_results = model.val(data=str(dataset_yaml), workers=0, device=device)

        # Extract authentic metrics from val_results
        map50 = float(getattr(getattr(val_results, "box", None), "map50", 0.0))
        map50_95 = float(getattr(getattr(val_results, "box", None), "map", 0.0))
        precision = float(getattr(getattr(val_results, "box", None), "mp", 0.0))
        recall = float(getattr(getattr(val_results, "box", None), "mr", 0.0))

        # Parse results.csv for authentic loss metrics
        csv_metrics = parse_results_csv(exp_dir / "results.csv")
        box_loss = csv_metrics.get("val/box_loss", csv_metrics.get("train/box_loss", 0.0))
        cls_loss = csv_metrics.get("val/cls_loss", csv_metrics.get("train/cls_loss", 0.0))
        dfl_loss = csv_metrics.get("val/dfl_loss", csv_metrics.get("train/dfl_loss", 0.0))
        total_loss = round(box_loss + cls_loss + dfl_loss, 4)

        # 3. Assemble structured metrics
        metrics_summary = {
            "model_name": "crayfish_multistage",
            "species": "cherax_quadricarinatus",
            "dataset": str(dataset_yaml),
            "classes": ["craylet", "juvenile", "sub_adult", "adult"],
            "num_classes": 4,
            "epochs_requested": epochs,
            "epochs_completed": epochs,
            "batch_size": batch,
            "imgsz": imgsz,
            "device": str(device),
            "best_weights_path": str(best_pt),
            "last_weights_path": str(last_pt) if last_pt.exists() else str(best_pt),
            "checkpoint_size_mb": checkpoint_size_mb,
            "metrics": {
                "mAP50": round(map50, 4),
                "mAP50_95": round(map50_95, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "box_loss": round(box_loss, 4),
                "cls_loss": round(cls_loss, 4),
                "dfl_loss": round(dfl_loss, 4),
                "total_loss": total_loss,
            },
            "csv_metrics": csv_metrics,
            "status": "SUCCESS",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        # 4. Write metrics.json
        metrics_path = exp_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)
        logger.info(f"Validation metrics logged to: {metrics_path}")

        return metrics_summary
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def verify_checkpoint(best_pt: Path, min_size_bytes: int = 1_000_000) -> bool:
    """Strictly verify that best.pt is a genuine, loadable binary YOLO/PyTorch checkpoint."""
    logger.info(f"Strictly verifying trained checkpoint: {best_pt}")
    if not best_pt.exists():
        logger.error(f"Checkpoint does not exist: {best_pt}")
        return False

    file_size = best_pt.stat().st_size
    if file_size < min_size_bytes:
        logger.error(f"Checkpoint file size {file_size} bytes is below minimum {min_size_bytes} bytes (not a binary model).")
        return False

    # 1. PyTorch deserialization test
    try:
        import torch
        ckpt = torch.load(str(best_pt), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or ("model" not in ckpt and "state_dict" not in ckpt and "ema" not in ckpt):
            logger.error("Checkpoint file is not a valid YOLO/PyTorch model state dict dictionary.")
            return False
        logger.info(f"PyTorch checkpoint verified. Keys: {list(ckpt.keys())[:5]}, size: {file_size / (1024*1024):.2f} MB")
    except Exception as e:
        logger.error(f"PyTorch failed to deserialize {best_pt}: {e}")
        return False

    # 2. Ultralytics YOLO load and inference test
    try:
        from ultralytics import YOLO
        import numpy as np

        test_model = YOLO(str(best_pt))
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = test_model.predict(dummy_img, verbose=False)
        if not results:
            logger.error("YOLO predict returned empty results.")
            return False
        logger.info("Verification passed: genuine YOLOv8 PyTorch model loaded and inferred successfully.")
        return True
    except Exception as e:
        logger.error(f"Ultralytics failed to load/infer checkpoint: {e}")
        return False


def main() -> None:
    args = parse_args()
    repo_root, dataset_yaml, project_dir, exp_dir = resolve_paths(args)

    report = train_crayfish_model(
        dataset_yaml=dataset_yaml,
        project_dir=project_dir,
        exp_name=args.name,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        base_model=args.base_model,
        device=args.device,
        workers=args.workers,
        resume=args.resume,
    )

    best_pt = Path(report["best_weights_path"])
    is_valid = verify_checkpoint(best_pt)

    print("\n" + "=" * 70)
    print("🦞 AnimalLens Crayfish Multi-Stage Model Training Complete")
    print("=" * 70)
    print(f"Checkpointed Weights: {best_pt}")
    print(f"Metrics JSON:         {exp_dir / 'metrics.json'}")
    print(f"Validation mAP@50:    {report['metrics']['mAP50']:.4f}")
    print(f"Validation mAP@50-95: {report['metrics']['mAP50_95']:.4f}")
    print(f"Checkpoint Verified:  {is_valid}")
    print("=" * 70 + "\n")

    if not is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
