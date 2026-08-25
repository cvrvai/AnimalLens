"""
Tests for Dataset Management and Asynchronous Training REST & WebSocket APIs (Phase 15).
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from animallens.server.app import create_app
from animallens.training import training_manager


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_list_datasets_endpoint(client):
    response = client.get("/v1/datasets")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_start_training_job_and_poll_status(client, tmp_path):
    # Create dummy dataset.yaml
    dummy_yaml = tmp_path / "dataset.yaml"
    dummy_yaml.write_text("names: {0: pig}\n", encoding="utf-8")

    req_payload = {
        "species": "pig",
        "dataset_yaml": str(dummy_yaml.resolve()),
        "base_model": "yolov8s.pt",
        "epochs": 2,
        "batch": 4,
        "device": "cpu",
        "experiment_name": "test_exp_v1",
    }

    # 1. Start job
    start_resp = client.post("/v1/train/start", json=req_payload)
    assert start_resp.status_code == 200
    job_data = start_resp.json()
    assert "job_id" in job_data
    assert job_data["species"] == "pig"
    assert job_data["epochs"] == 2

    job_id = job_data["job_id"]

    # 2. List jobs
    list_resp = client.get("/v1/train/jobs")
    assert list_resp.status_code == 200
    all_jobs = list_resp.json()
    assert any(j["job_id"] == job_id for j in all_jobs)

    # 3. Poll status
    status_resp = client.get(f"/v1/train/status/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["job_id"] == job_id
    assert status_data["status"] in ["QUEUED", "RUNNING", "COMPLETED"]


def test_update_dataset_annotation_endpoint(client, tmp_path):
    # Setup test dataset folder
    dataset_name = "test_annotation_ds"
    ds_dir = Path("datasets") / dataset_name
    (ds_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (ds_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (ds_dir / "dataset.yaml").write_text("names: {0: pig}\n", encoding="utf-8")

    payload = {
        "image_name": "frame_0001.jpg",
        "split": "train",
        "bboxes": [
            {"class_id": 0, "x_center": 0.5, "y_center": 0.5, "width": 0.3, "height": 0.4}
        ]
    }

    resp = client.put(f"/v1/datasets/{dataset_name}/annotations", json=payload)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["status"] == "saved"
    assert res_data["box_count"] == 1

    # Verify saved file
    txt_file = ds_dir / "labels" / "train" / "frame_0001.txt"
    assert txt_file.exists()
    content = txt_file.read_text(encoding="utf-8").strip()
    assert content.startswith("0 0.500000 0.500000 0.300000 0.400000")

    # Clean up test dataset
    import shutil
    shutil.rmtree(ds_dir, ignore_errors=True)
