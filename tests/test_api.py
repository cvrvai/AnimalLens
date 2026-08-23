"""
Unit tests for FastAPI endpoints, video analysis, tracking telemetry, and behavioral classifier.
"""
import io
import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from animallens.behavior.classifier import BehavioralClassifier
from animallens.core.schemas import BoundingBox
from animallens.perception.base import DetectionResult
from animallens.server.app import app
from animallens.tracking.tracker import AnimalTracker

client = TestClient(app)


def test_api_health():
    res = client.get("/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "AnimalLens"
    assert "device" in data
    assert "cuda_available" in data
    assert "installed_models" in data


def test_api_species():
    res = client.get("/v1/species")
    assert res.status_code == 200
    data = res.json()
    assert any(s["id"] == "cherax_quadricarinatus" for s in data)
    assert any(s["id"] == "canis_lupus_familiaris" for s in data)


def test_api_species_details():
    res = client.get("/v1/species/redclaw")
    assert res.status_code == 200
    data = res.json()
    assert "categories" in data["taxonomy"]

    res_dog = client.get("/v1/species/dog")
    assert res_dog.status_code == 200
    data_dog = res_dog.json()
    assert data_dog["config"]["name"] == "Domestic Dog"


def test_api_models():
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert "available" in data
    assert "installed" in data


def test_api_analyze_image():
    # Create test image in memory
    img = Image.new("RGB", (320, 240), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    res = client.post(
        "/v1/analyze/image",
        data={"species": "dog"},
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["species"] == "Domestic Dog"
    assert len(data["behaviors"]) > 0


def test_api_analyze_video():
    # Create a small in-memory test MP4 video (10 frames)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = 120, 160
    buf = io.BytesIO()

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_name = tmp.name

    writer = cv2.VideoWriter(tmp_name, fourcc, 10.0, (w, h))
    for i in range(10):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(frame, (20 + i * 5, 60), 10, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    with open(tmp_name, "rb") as f:
        vid_bytes = f.read()
    os.unlink(tmp_name)

    res = client.post(
        "/v1/analyze/video",
        data={"species": "dog", "sample_fps": 5.0},
        files={"file": ("test_vid.mp4", io.BytesIO(vid_bytes), "video/mp4")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["species"] == "Domestic Dog"
    assert data["total_frames_analyzed"] > 0
    assert len(data["timeline"]) > 0


def test_tracking_and_kinematic_math():
    tracker = AnimalTracker(species_prefix="DOG", pixel_to_meter_ratio=25.0)
    classifier = BehavioralClassifier()

    # Frame 1: Detection at (0.2, 0.5)
    det1 = DetectionResult(
        bboxes=[BoundingBox(x_min=0.15, y_min=0.45, x_max=0.25, y_max=0.55)],
        confidences=[0.95],
        class_names=["dog"],
    )
    telemetry1 = tracker.update_frame(det1, timestamp=0.0, dt=0.1)
    assert len(telemetry1.subjects) == 1
    assert telemetry1.subjects[0].display_id == "DOG-01"
    assert telemetry1.subjects[0].center_pct == (20.0, 50.0)

    # Frame 2: Continuous displacement along X (tracking lock)
    det2 = DetectionResult(
        bboxes=[BoundingBox(x_min=0.18, y_min=0.45, x_max=0.28, y_max=0.55)],
        confidences=[0.97],
        class_names=["dog"],
    )
    telemetry2 = tracker.update_frame(det2, timestamp=0.1, dt=0.1)
    assert len(telemetry2.subjects) == 1
    sub = telemetry2.subjects[0]
    assert sub.track_id == 1  # Continuous identity
    assert sub.velocity_mps > 3.5  # (0.03 * 25) / 0.1 = 7.5 m/s (Fast sprint)
    assert sub.heading_degrees >= 0.0

    # Classify behavior
    classified = classifier.classify_subject(sub, frame_telemetry=telemetry2)
    assert classified.category == "locomotion"
    assert classified.label == "running_gallop"
    assert classified.speed_kmh > 0


def test_api_reid_gallery_and_register():
    # Test register endpoint
    res_reg = client.post(
        "/v1/reid/register",
        json={"name": "Max", "species": "dog", "metadata": {"breed": "Border Collie"}},
    )
    assert res_reg.status_code == 200
    assert res_reg.json()["status"] == "registered"
    assert res_reg.json()["name"] == "Max"

    # Test gallery listing endpoint
    res_gal = client.get("/v1/reid/gallery")
    assert res_gal.status_code == 200
    gal_data = res_gal.json()
    assert gal_data["total_registered"] >= 1
    assert any(p["name"] == "Max" for p in gal_data["profiles"])
