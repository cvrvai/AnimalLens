"""
Unit tests for FastAPI endpoints.
"""
import io
from fastapi.testclient import TestClient
from PIL import Image
from animallens.server.app import app

client = TestClient(app)


def test_api_health():
    res = client.get("/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "AnimalLens"


def test_api_species():
    res = client.get("/v1/species")
    assert res.status_code == 200
    data = res.json()
    assert any(s["id"] == "cherax_quadricarinatus" for s in data)


def test_api_species_details():
    res = client.get("/v1/species/redclaw")
    assert res.status_code == 200
    data = res.json()
    assert "categories" in data["taxonomy"]


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
        data={"species": "redclaw"},
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["species"] == "Redclaw Crayfish"
    assert len(data["behaviors"]) > 0
