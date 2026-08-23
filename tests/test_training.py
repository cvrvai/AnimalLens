"""
Unit Tests for VideoDatasetBuilder, ModelTrainer, and Training CLI (Phase 10).
"""
import shutil
from pathlib import Path
import pytest
from animallens.training.dataset_builder import VideoDatasetBuilder
from animallens.training.trainer import ModelTrainer, TrainingReport


@pytest.fixture
def temp_training_dir(tmp_path):
    d = tmp_path / "test_train_pipeline"
    d.mkdir()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_video_dataset_builder_structure(temp_training_dir):
    builder = VideoDatasetBuilder(output_dir=temp_training_dir / "dataset", val_split=0.2)
    assert (temp_training_dir / "dataset" / "images" / "train").exists()
    assert (temp_training_dir / "dataset" / "images" / "val").exists()
    assert (temp_training_dir / "dataset" / "labels" / "train").exists()
    assert (temp_training_dir / "dataset" / "labels" / "val").exists()

    yaml_path = builder.write_yaml_config(classes=["dog", "carapace"])
    assert yaml_path.exists()
    assert "dataset.yaml" in yaml_path.name


def test_model_trainer_checkpointing(temp_training_dir):
    import cv2
    import numpy as np

    builder = VideoDatasetBuilder(output_dir=temp_training_dir / "dataset")
    
    # Create dummy images & labels for train and val splits
    dummy_frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for split in ["train", "val"]:
        img_path = temp_training_dir / "dataset" / "images" / split / "dummy_01.jpg"
        lbl_path = temp_training_dir / "dataset" / "labels" / split / "dummy_01.txt"
        cv2.imwrite(str(img_path), dummy_frame)
        lbl_path.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    yaml_path = builder.write_yaml_config(classes=["dog"])

    trainer = ModelTrainer(
        base_model="yolov8n.pt",
        project_dir=temp_training_dir / "models",
        experiment_name="test_canine_exp",
    )

    report = trainer.train(
        dataset_yaml=yaml_path,
        epochs=1,
        imgsz=64,
        batch=1,
        device="cpu",
        export_onnx=False,
    )

    assert isinstance(report, TrainingReport)
    assert report.epochs_completed == 1
    assert report.best_weights_path.exists()
    assert report.last_weights_path.exists()
    assert 0.0 <= report.map50 <= 1.0

    d = report.to_dict()
    assert "best_weights_path" in d
    assert "map50" in d
