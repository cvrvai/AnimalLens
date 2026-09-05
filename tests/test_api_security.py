"""
Automated tests for AnimalLens API Key Security and Cloudflare Tunnel readiness.
Validates:
1. Public endpoints (/v1/health) bypass authentication.
2. Protected endpoints reject missing/invalid keys with HTTP 401.
3. Protected endpoints accept valid X-API-Key and Authorization: Bearer tokens.
4. Open local mode when ANIMALLENS_API_KEY is unset.
"""
import io
import os
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from animallens.server.app import create_app


def create_dummy_jpeg() -> bytes:
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def auth_client():
    """Create test client with ANIMALLENS_API_KEY configured."""
    original_key = os.environ.get("ANIMALLENS_API_KEY")
    os.environ["ANIMALLENS_API_KEY"] = "secret_crayfish_token_xyz987"
    app = create_app()
    client = TestClient(app)
    yield client
    if original_key is not None:
        os.environ["ANIMALLENS_API_KEY"] = original_key
    else:
        os.environ.pop("ANIMALLENS_API_KEY", None)


@pytest.fixture
def open_client():
    """Create test client with NO ANIMALLENS_API_KEY (local development mode)."""
    original_key = os.environ.get("ANIMALLENS_API_KEY")
    os.environ["ANIMALLENS_API_KEY"] = ""
    app = create_app()
    client = TestClient(app)
    yield client
    if original_key is not None:
        os.environ["ANIMALLENS_API_KEY"] = original_key
    else:
        os.environ.pop("ANIMALLENS_API_KEY", None)


def test_public_health_bypasses_auth(auth_client):
    """Health check endpoint must remain public for Cloudflare Tunnel monitoring."""
    resp = auth_client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


def test_protected_endpoint_rejects_missing_key(auth_client):
    """Protected endpoints must return 401 when no key is supplied."""
    jpeg_bytes = create_dummy_jpeg()
    resp = auth_client.post(
        "/v1/analyze/image",
        files={"file": ("tray.jpg", jpeg_bytes, "image/jpeg")},
        data={"species": "redclaw"},
    )
    assert resp.status_code == 401
    assert "Invalid or missing AnimalLens API Key" in resp.json().get("detail", "")


def test_protected_endpoint_rejects_wrong_key(auth_client):
    """Protected endpoints must return 401 when an invalid key is supplied."""
    jpeg_bytes = create_dummy_jpeg()
    resp = auth_client.post(
        "/v1/analyze/image",
        headers={"X-API-Key": "wrong_invalid_key"},
        files={"file": ("tray.jpg", jpeg_bytes, "image/jpeg")},
        data={"species": "redclaw"},
    )
    assert resp.status_code == 401


def test_protected_endpoint_accepts_x_api_key(auth_client):
    """Protected endpoints must succeed when valid X-API-Key is provided."""
    jpeg_bytes = create_dummy_jpeg()
    resp = auth_client.post(
        "/v1/analyze/image",
        headers={"X-API-Key": "secret_crayfish_token_xyz987"},
        files={"file": ("tray.jpg", jpeg_bytes, "image/jpeg")},
        data={"species": "redclaw"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "count" in data
    assert "detections" in data


def test_protected_endpoint_accepts_bearer_token(auth_client):
    """Protected endpoints must succeed when valid Authorization: Bearer token is provided."""
    jpeg_bytes = create_dummy_jpeg()
    resp = auth_client.post(
        "/v1/analyze/image",
        headers={"Authorization": "Bearer secret_crayfish_token_xyz987"},
        files={"file": ("tray.jpg", jpeg_bytes, "image/jpeg")},
        data={"species": "redclaw"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "count" in data
    assert "detections" in data


def test_open_local_mode_when_key_unset(open_client):
    """When ANIMALLENS_API_KEY is empty, endpoints allow unauthenticated local calls."""
    jpeg_bytes = create_dummy_jpeg()
    resp = open_client.post(
        "/v1/analyze/image",
        files={"file": ("tray.jpg", jpeg_bytes, "image/jpeg")},
        data={"species": "redclaw"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "count" in data


def test_websocket_events_rejects_missing_key(auth_client):
    """WebSocket /v1/events must close connection when API key is missing."""
    import pytest
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with auth_client.websocket_connect("/v1/events") as ws:
            pass
    assert excinfo.value.code in (1008, 4401)


def test_websocket_events_accepts_valid_query_key(auth_client):
    """WebSocket /v1/events must connect when valid ?api_key= query parameter is passed."""
    with auth_client.websocket_connect("/v1/events?api_key=secret_crayfish_token_xyz987") as ws:
        ws.send_text("ping")
        resp = ws.receive_json()
        assert resp.get("type") == "pong"


def test_websocket_events_accepts_valid_header_key(auth_client):
    """WebSocket /v1/events must connect when valid x-api-key header is passed."""
    with auth_client.websocket_connect("/v1/events", headers={"x-api-key": "secret_crayfish_token_xyz987"}) as ws:
        ws.send_text("ping")
        resp = ws.receive_json()
        assert resp.get("type") == "pong"

