"""
Unit tests for Anti-Leakage Dataset Partitioning, Cohen's Kappa, and Format Converters.
"""
from pathlib import Path
import pytest
from animallens.datasets.converter import BBoxConverter, DatasetExporter
from animallens.datasets.kappa import CohenKappaValidator
from animallens.datasets.partitioner import AntiLeakagePartitioner, DatasetSample


def test_anti_leakage_partitioner_disjoint_sessions():
    """Verify that sessions are strictly disjoint across train, val, and test splits."""
    samples = [
        DatasetSample(sample_id="s1_f1", session_id="session_01", tank_id="TANK-A", species_id="redclaw", file_path="img1.jpg"),
        DatasetSample(sample_id="s1_f2", session_id="session_01", tank_id="TANK-A", species_id="redclaw", file_path="img2.jpg"),
        DatasetSample(sample_id="s2_f1", session_id="session_02", tank_id="TANK-A", species_id="redclaw", file_path="img3.jpg"),
        DatasetSample(sample_id="s3_f1", session_id="session_03", tank_id="TANK-B", species_id="redclaw", file_path="img4.jpg"),
        DatasetSample(sample_id="s4_f1", session_id="session_04", tank_id="TANK-B", species_id="redclaw", file_path="img5.jpg"),
        DatasetSample(sample_id="s5_f1", session_id="session_05", tank_id="TANK-C", species_id="redclaw", file_path="img6.jpg"),
    ]

    partitioner = AntiLeakagePartitioner(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42)
    result = partitioner.split_by_session(samples)

    assert result.total_count == 6
    train_sess = set(result.train_sessions)
    val_sess = set(result.val_sessions)
    test_sess = set(result.test_sessions)

    # Strictly 0% leakage guarantee
    assert train_sess.isdisjoint(val_sess)
    assert train_sess.isdisjoint(test_sess)
    assert val_sess.isdisjoint(test_sess)
    assert result.leakage_score == 0.0


def test_cohen_kappa_perfect_agreement():
    """Verify Cohen's Kappa for 100% agreement."""
    a1 = ["mating", "fighting", "foraging", "resting"] * 10
    a2 = ["mating", "fighting", "foraging", "resting"] * 10

    report = CohenKappaValidator.compute_kappa(a1, a2)
    assert report.cohen_kappa == 1.0
    assert report.observed_agreement == 1.0
    assert report.is_valid_for_training is True
    assert "Almost Perfect" in report.interpretation or "Excellent" in report.interpretation


def test_cohen_kappa_partial_agreement():
    """Verify Cohen's Kappa computation with realistic partial inter-rater disagreement."""
    a1 = ["fighting", "fighting", "foraging", "resting", "resting", "mating", "mating", "mating"]
    a2 = ["fighting", "foraging", "foraging", "resting", "resting", "mating", "mating", "resting"]

    report = CohenKappaValidator.compute_kappa(a1, a2, threshold=0.60)
    assert 0.40 <= report.cohen_kappa <= 0.85
    assert report.sample_count == 8
    assert "fighting" in report.categories


def test_bbox_converter_roundtrip():
    """Verify roundtrip conversion between xyxy and YOLO formats."""
    orig_xyxy = (0.1, 0.2, 0.5, 0.8)
    x_c, y_c, w, h = BBoxConverter.xyxy_to_yolo(*orig_xyxy)
    recovered = BBoxConverter.yolo_to_xyxy(x_c, y_c, w, h)

    assert pytest.approx(recovered[0], 0.001) == orig_xyxy[0]
    assert pytest.approx(recovered[1], 0.001) == orig_xyxy[1]
    assert pytest.approx(recovered[2], 0.001) == orig_xyxy[2]
    assert pytest.approx(recovered[3], 0.001) == orig_xyxy[3]


def test_dataset_exporter_yaml(tmp_path):
    """Verify YOLOv8 dataset.yaml creation."""
    classes = ["cherax_quadricarinatus", "carapace", "chela"]
    yaml_content = DatasetExporter.generate_yolo_yaml(
        dataset_dir=tmp_path,
        class_names=classes,
    )
    assert "cherax_quadricarinatus" in yaml_content
    assert "images/train" in yaml_content
