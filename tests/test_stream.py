"""
Unit tests for Low-Latency RTSP StreamSource, LiveStreamManager, and Streaming REST Endpoints.
"""
import time
from fastapi.testclient import TestClient
import pytest
from animallens.server.app import app
from animallens.server.stream_manager import LiveStreamManager, live_stream_manager
from animallens.sources.stream import StreamSource

client = TestClient(app)


def test_stream_source_metrics_and_queue():
    """Verify StreamSource non-blocking queue, metrics reporting, and shutdown."""
    source = StreamSource(
        endpoint="rtsp://fake.camera/live",
        target_fps=30.0,
        camera_id="CAM-TANK-01",
        buffer_size=2,
    )

    metrics = source.get_metrics()
    assert metrics["camera_id"] == "CAM-TANK-01"
    assert metrics["frames_received"] >= 0

    # Start source and read 2 frames
    frames_read = 0
    for ts, frame in source:
        frames_read += 1
        assert frame is not None
        assert ts >= 0.0
        if frames_read >= 2:
            break

    source.stop()
    assert frames_read == 2


def test_live_stream_manager():
    """Verify LiveStreamManager starting, listing, and stopping camera workers."""
    mgr = LiveStreamManager()

    start_res = mgr.start_stream(
        camera_id="CAM-01",
        rtsp_url="rtsp://192.168.1.100/live",
        species="redclaw",
        save_to_db=False,
    )
    assert start_res["status"] == "started"
    assert start_res["camera_id"] == "CAM-01"

    status_list = mgr.list_streams()
    assert status_list["active_streams_count"] == 1
    assert status_list["streams"][0]["camera_id"] == "CAM-01"

    stopped = mgr.stop_stream("CAM-01")
    assert stopped is True

    status_after = mgr.list_streams()
    assert status_after["active_streams_count"] == 0


def test_stream_rest_endpoints():
    """Verify FastAPI endpoints for stream management."""
    # 1. Start stream
    res = client.post(
        "/v1/stream/start",
        json={
            "camera_id": "CAM-API-01",
            "rtsp_url": "rtsp://192.168.1.150/feed",
            "species": "redclaw",
            "save_to_db": False,
        },
    )
    assert res.status_code == 200
    assert res.json()["status"] == "started"

    # 2. List active streams
    list_res = client.get("/v1/stream/active")
    assert list_res.status_code == 200
    active_cams = [s["camera_id"] for s in list_res.json()["streams"]]
    assert "CAM-API-01" in active_cams

    # 3. Stop stream
    stop_res = client.post("/v1/stream/stop/CAM-API-01")
    assert stop_res.status_code == 200
    assert stop_res.json()["status"] == "stopped"
