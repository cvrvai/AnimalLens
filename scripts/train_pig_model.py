"""
End-to-end pig behavior model training pipeline.
1. Extract keyframes from pig_farm_pen.mp4 at 2 FPS
2. Generate pseudo-labels with YOLOv8 detector
3. Run transfer learning fine-tuning
"""
import logging
import sys
import os
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger("pig_training")


def main():
    print("=" * 70)
    print("  AnimalLens — Swine / Domestic Pig Model Training Pipeline")
    print("=" * 70)

    video_path = Path("data/raw/videos/pig_farm_pen.mp4")
    if not video_path.exists():
        print(f"ERROR: Video not found at {video_path}")
        sys.exit(1)

    print(f"\n📹 Input video: {video_path} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # --- Step 1: Extract keyframes ---
    print("\n[Step 1/3] Extracting keyframes at 2 FPS...")
    from animallens.training.dataset_builder import VideoDatasetBuilder

    output_dir = Path("models/trained/pig_dataset")
    builder = VideoDatasetBuilder(output_dir=output_dir)
    frames = builder.extract_keyframes(video_path, sample_fps=2.0)
    print(f"   ✅ Extracted {len(frames)} frames")
    print(f"   Train: {list(builder.images_train.glob('*.jpg')).__len__()} frames")
    print(f"   Val:   {list(builder.images_val.glob('*.jpg')).__len__()} frames")

    # --- Step 2: Generate pseudo-labels ---
    print("\n[Step 2/3] Generating pseudo-labels with YOLOv8 pig detector...")
    pig_classes = ["pig", "swine", "hog", "sow", "piglet", "sus_scrofa_domesticus"]
    num_labeled = builder.generate_pseudo_labels(classes=pig_classes, conf_threshold=0.30)
    print(f"   ✅ Generated pseudo-labels for {num_labeled} images")

    # Show sample label
    sample_labels = list(builder.labels_train.glob("*.txt"))
    if sample_labels:
        content = sample_labels[0].read_text(encoding="utf-8").strip()
        print(f"   Sample label ({sample_labels[0].name}):")
        for line in content.split("\n")[:3]:
            print(f"     {line}")

    # --- Step 3: Write YOLO dataset config ---
    print("\n[Step 3/3] Writing YOLO dataset.yaml config...")
    dataset_yaml = builder.write_yaml_config(species_name="pig", classes=["pig"])
    print(f"   ✅ Config written to: {dataset_yaml}")

    # Show config contents
    print(f"\n   dataset.yaml contents:")
    for line in dataset_yaml.read_text(encoding="utf-8").strip().split("\n"):
        print(f"     {line}")

    # --- Step 4: Run model training ---
    print("\n" + "=" * 70)
    print("  Model Training (Transfer Learning Fine-Tuning)")
    print("=" * 70)

    from animallens.training.trainer import ModelTrainer

    trainer = ModelTrainer(
        base_model="yolov8s.pt",
        project_dir=Path("models/trained"),
        experiment_name="pig_behavior_v1",
    )

    print(f"\n🏋️ Starting training with:")
    print(f"   Base Model:    yolov8s.pt")
    print(f"   Dataset:       {dataset_yaml}")
    print(f"   Epochs:        10 (demo)")
    print(f"   Image Size:    640")
    print(f"   Device:        cpu")

    report = trainer.train(
        dataset_yaml=dataset_yaml,
        epochs=10,
        imgsz=640,
        batch=8,
        device="cpu",
        resume=False,
    )

    print(f"\n📊 Training Results:")
    print(f"   Status:            {report.status}")
    print(f"   Epochs Completed:  {report.epochs_completed}")
    print(f"   Validation mAP@50:     {report.map50:.4f}")
    print(f"   Validation mAP@50-95:  {report.map50_95:.4f}")
    print(f"   Best Weights:      {report.best_weights_path}")
    print(f"   Last Checkpoint:   {report.last_weights_path}")
    if report.onnx_weights_path:
        print(f"   ONNX Export:       {report.onnx_weights_path}")

    print("\n" + "=" * 70)
    print("  ✅ Pig behavior model training pipeline COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
