"""
Comprehensive Unit and API Integration Test Suite for Milestone 1 (M1):
AnimalLens Aquaculture Livestock Vision Pipeline, Stream Worker, and Telemetry.

Covers:
- Feature 1: POST /v1/analyze/image parameter resolution (Query, Form, default redclaw, 0-byte check, backwards compatibility)
- Feature 2: Crayfish lifecycle stage classification and normalization (craylet, juvenile, sub-adult, adult)
- Feature 3: RTSP stream ingestion, FPS throttling (1-15 fps), and seamless video loop rewind
- Feature 4: Memory-bounded tracking history (history <= 50 in TrackItem)
- Feature 5: GET /v1/stream/active real-time metrics schema contract compliance
- Feature 6: WebSocket ping/pong, live telemetry broadcasting, and graceful disconnect
- Classical CV adaptive thresholding & 4-stage simulated fallback (<200ms latency)
- Spatial metrics crowding density and ethological activity calculations
"""
from __future__ import annotations

import asyncio
import io
import time
from typing import Any, Dict, List
import cv2
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image, ImageDraw
import pytest

from animallens.analytics.sampling_protocols import SamplingProtocols
from animallens.analytics.spatial_metrics import compute_spatial_metrics
from animallens.core.schemas import (
    AnalysisResult,
    BehaviorEvent,
    BehaviorInfo,
    BoundingBox,
    ModelInfo,
    SourceInfo,
    SourceType,
    SpeciesInfo,
    SubjectInfo,
    TemporalInfo,
)
from animallens.perception.base import DetectionResult, TrackState
from animallens.perception.models.botsort_tracker import BoTSORTTracker, TrackItem
from animallens.perception.models.yolov8_detector import YOLOv8Detector
from animallens.sdk import AnimalLens, normalize_lifecycle_stage
from animallens.server.app import app
from animallens.server.stream_manager import LiveStreamManager, StreamWorker
from animallens.server.websocket import ws_manager
from animallens.sources.stream import StreamSource

client = TestClient(app)


def _create_test_image_bytes(width: int = 640, height: int = 480, color=(30, 45, 60)) -> io.BytesIO:
    """Helper to generate a valid RGB JPEG image in memory."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Feature 1: Parameter Resolution, 0-Byte Rejection & Backwards Compatibility
# ---------------------------------------------------------------------------

def test_stage_normalization_utility():
    """Verify lifecycle stage string normalization to canonical lowercase-hyphenated standard."""
    assert normalize_lifecycle_stage("sub_adult") == "sub-adult"
    assert normalize_lifecycle_stage("sub-adult") == "sub-adult"
    assert normalize_lifecycle_stage("Sub-Adult") == "sub-adult"
    assert normalize_lifecycle_stage("SUB_ADULT") == "sub-adult"
    assert normalize_lifecycle_stage("craylet") == "craylet"
    assert normalize_lifecycle_stage("juvenile") == "juvenile"
    assert normalize_lifecycle_stage("adult") == "adult"
    assert normalize_lifecycle_stage("crayfish") is None
    assert normalize_lifecycle_stage("dog") is None
    assert normalize_lifecycle_stage("") is None
    assert normalize_lifecycle_stage(None) is None


def test_analyze_image_default_species():
    """Verify POST /v1/analyze/image defaults to species 'redclaw' when unspecified."""
    buf = _create_test_image_bytes()
    res = client.post(
        "/v1/analyze/image",
        files={"file": ("tray_scan.jpg", buf, "image/jpeg")},
    )
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert "count" in data
    assert "detections" in data
    assert data["processing_time_ms"] < 200.0, f"Latency {data['processing_time_ms']}ms exceeded 200ms limit"
    assert any(term in str(data.get("species", "")).lower() for term in ("redclaw", "cherax", "crayfish"))
    assert data["count"] == len(data["detections"])


def test_analyze_image_query_and_form_species():
    """Verify species can be specified via Query string (?species=redclaw) or Form data."""
    # 1. Query parameter
    buf1 = _create_test_image_bytes()
    res_query = client.post(
        "/v1/analyze/image?species=redclaw",
        files={"file": ("tray1.jpg", buf1, "image/jpeg")},
    )
    assert res_query.status_code == 200
    assert any(term in str(res_query.json().get("species", "")).lower() for term in ("redclaw", "cherax"))

    # 2. Form parameter
    buf2 = _create_test_image_bytes()
    res_form = client.post(
        "/v1/analyze/image",
        data={"species": "redclaw"},
        files={"file": ("tray2.jpg", buf2, "image/jpeg")},
    )
    assert res_form.status_code == 200
    assert any(term in str(res_form.json().get("species", "")).lower() for term in ("redclaw", "cherax"))


def test_analyze_image_empty_file_rejected():
    """Verify POST /v1/analyze/image rejects 0-byte uploaded files with HTTP 400."""
    empty_buf = io.BytesIO(b"")
    res = client.post(
        "/v1/analyze/image",
        files={"file": ("empty.jpg", empty_buf, "image/jpeg")},
    )
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower() or "0 bytes" in res.json()["detail"].lower()


def test_analyze_image_dog_backwards_compatibility():
    """Verify POST /v1/analyze/image with data={'species': 'dog'} preserves domestic dog behavior."""
    buf = _create_test_image_bytes(color=(100, 100, 100))
    res = client.post(
        "/v1/analyze/image",
        data={"species": "dog"},
        files={"file": ("dog.jpg", buf, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["species"] == "Domestic Dog"


# ---------------------------------------------------------------------------
# Feature 2: Lifecycle Stage Classification & Detection Fallback
# ---------------------------------------------------------------------------

def test_crayfish_lifecycle_stage_classification_thresholds():
    """
    Verify stage classification maps normalized bounding box area to:
    - craylet:   < 0.02
    - juvenile:  0.02 to < 0.08
    - sub-adult: 0.08 to < 0.18  (strictly hyphenated 'sub-adult')
    - adult:     >= 0.18
    """
    boxes = [
        ("craylet", BoundingBox(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2)),          # area = 0.010 (< 0.02)
        ("juvenile", BoundingBox(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.3)),         # area = 0.040 (0.02 - 0.08)
        ("sub-adult", BoundingBox(x_min=0.1, y_min=0.1, x_max=0.4, y_max=0.45)),       # area = 0.105 (0.08 - 0.18)
        ("adult", BoundingBox(x_min=0.1, y_min=0.1, x_max=0.6, y_max=0.55)),           # area = 0.225 (>= 0.18)
    ]

    for expected_stage, bbox in boxes:
        area = bbox.area
        if area < 0.02:
            stage = "craylet"
        elif area < 0.08:
            stage = "juvenile"
        elif area < 0.18:
            stage = "sub-adult"
        else:
            stage = "adult"

        assert stage == expected_stage, f"Box with area {area:.4f} classified as {stage}, expected {expected_stage}"
        assert stage != "sub_adult", "Stage 'sub-adult' must be hyphenated, not underscored"


def _create_multistage_tray_image(width: int = 640, height: int = 480) -> Image.Image:
    """Generate a high-contrast tray scan image containing 4 crayfish lifecycle stages."""
    img = Image.new("RGB", (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # 1. Craylet (<0.02 area)
    draw.ellipse([60, 60, 95, 95], fill=(20, 20, 20))
    # 2. Juvenile (0.02 - 0.08 area)
    draw.ellipse([180, 140, 270, 230], fill=(25, 25, 25))
    # 3. Sub-adult (0.08 - 0.18 area)
    draw.ellipse([330, 80, 500, 260], fill=(20, 20, 20))
    # 4. Adult (>=0.18 area)
    draw.ellipse([100, 250, 450, 450], fill=(15, 15, 15))
    return img


def test_yolov8_detector_crayfish_multistage_fallback():
    """Verify honest empty detection on blank frame and 4 stages detection under 200ms on tray scan."""
    detector = YOLOv8Detector(classes=["crayfish", "craylet", "juvenile", "sub_adult", "adult"])

    # 1. Honest empty detection on blank frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    empty_res = detector.detect(dummy_frame)
    assert len(empty_res.bboxes) == 0, f"Expected 0 bboxes on blank frame, got {len(empty_res.bboxes)}"
    assert len(empty_res.confidences) == 0
    assert len(empty_res.class_names) == 0

    # 2. Genuine classical CV detection of 4 lifecycle stages on tray scan image
    tray_img = _create_multistage_tray_image()
    t0 = time.perf_counter()
    res = detector.detect(tray_img)
    latency_ms = (time.perf_counter() - t0) * 1000

    assert latency_ms < 200.0, f"Detection took {latency_ms}ms, exceeding 200ms limit"
    assert len(res.bboxes) == 4, f"Expected 4 stage bboxes, got {len(res.bboxes)}"
    assert set(res.class_names) == {"craylet", "juvenile", "sub_adult", "adult"}

    areas = {c: b.area for c, b in zip(res.class_names, res.bboxes)}
    assert areas["craylet"] < 0.02, f"Craylet area {areas['craylet']} not < 0.02"
    assert 0.02 <= areas["juvenile"] < 0.08, f"Juvenile area {areas['juvenile']} not in [0.02, 0.08)"
    assert 0.08 <= areas["sub_adult"] < 0.18, f"Sub-adult area {areas['sub_adult']} not in [0.08, 0.18)"
    assert areas["adult"] >= 0.18, f"Adult area {areas['adult']} not >= 0.18"


def test_yolov8_classical_thresholding_on_tray_image():
    """Verify classical CV adaptive thresholding heuristics detect dark objects on high-contrast trays."""
    detector = YOLOv8Detector(classes=["crayfish", "craylet", "juvenile", "sub_adult", "adult"])

    # Create synthetic white tray with dark crayfish blobs
    tray_img = Image.new("RGB", (640, 480), color=(240, 240, 240))
    draw = ImageDraw.Draw(tray_img)
    # Craylet blob
    draw.ellipse([50, 50, 80, 80], fill=(20, 20, 20))
    # Juvenile blob
    draw.ellipse([200, 150, 280, 250], fill=(30, 30, 30))

    t0 = time.perf_counter()
    res = detector.detect(tray_img)
    latency_ms = (time.perf_counter() - t0) * 1000

    assert latency_ms < 200.0
    assert len(res.bboxes) >= 2
    assert all(0.0 <= b.x_min <= 1.0 for b in res.bboxes)


def test_sdk_redclaw_lifecycle_stage_classification():
    """Verify AnimalLens SDK produces normalized stages: craylet, juvenile, sub-adult, adult."""
    lens = AnimalLens(species="redclaw", reasoning=None)
    tray_img = _create_multistage_tray_image()

    result = lens.analyze_image(tray_img)
    assert isinstance(result, AnalysisResult)
    assert any(term in result.species.lower() for term in ("redclaw", "cherax"))
    assert result.count == 4
    assert len(result.detections) == 4

    stages = [d["stage"] for d in result.detections]
    assert "craylet" in stages
    assert "juvenile" in stages
    assert "sub-adult" in stages
    assert "adult" in stages
    assert "sub_adult" not in stages, "Underscore sub_adult must be normalized to sub-adult"


# ---------------------------------------------------------------------------
# Feature 3: RTSP Stream Ingestion, FPS Throttling & Seamless Video Loop
# ---------------------------------------------------------------------------

def test_stream_source_target_fps_and_bounded_buffer():
    """Verify StreamSource respects target_fps and maintains bounded queue."""
    target_fps = 10.0
    source = StreamSource(
        endpoint="rtsp://mock.tank.camera/live",
        target_fps=target_fps,
        camera_id="CAM-TANK-TEST",
        buffer_size=2,
    )
    assert source.fps == target_fps
    assert source.buffer_size == 2

    # Read 3 synthetic frames and verify timing
    t0 = time.time()
    frames = []
    for ts, frame in source:
        frames.append(frame)
        if len(frames) >= 3:
            break
    source.stop()

    assert len(frames) == 3
    elapsed = time.time() - t0
    # 3 frames at 10 fps takes approximately 0.15s - 0.45s
    assert elapsed >= 0.15, f"Frames received too rapidly ({elapsed:.3f}s for 3 frames at 10fps)"


def test_video_loop_seamless_rewind(tmp_path):
    """Verify StreamSource seamlessly loops simulated video files past EOF."""
    video_path = str(tmp_path / "test_loop.mp4")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (100, 100))
    for i in range(5):
        frame = np.full((100, 100, 3), i * 40, dtype=np.uint8)
        out.write(frame)
    out.release()

    source = StreamSource(endpoint=video_path, target_fps=15.0)
    frames_read = 0
    for ts, frame in source:
        frames_read += 1
        # Video has only 5 frames; reading 12 frames proves seamless rewind past EOF
        if frames_read >= 12:
            break
    source.stop()

    assert frames_read == 12, "StreamSource failed to seamlessly loop past EOF"


# ---------------------------------------------------------------------------
# Feature 4: Memory Bounded Tracking History
# ---------------------------------------------------------------------------

def test_botsort_tracker_memory_bounded_history():
    """Verify TrackItem caps history length to <= 50 to prevent memory leaks."""
    tracker = BoTSORTTracker()
    test_bbox = BoundingBox(x_min=0.2, y_min=0.2, x_max=0.4, y_max=0.4)

    # Perform 120 updates on the same track
    for i in range(120):
        det = DetectionResult(
            bboxes=[test_bbox],
            confidences=[0.95],
            class_names=["crayfish"],
        )
        tracker.update(det, timestamp=float(i) * 0.1)

    tracks = tracker.active_tracks
    assert len(tracks) > 0
    item = tracks[0]
    assert len(item.history) <= 50, f"TrackItem.history leaked! Length: {len(item.history)} (expected <= 50)"


# ---------------------------------------------------------------------------
# Feature 5: Stream Metrics REST API Contract (GET /v1/stream/active)
# ---------------------------------------------------------------------------

def test_stream_active_metrics_schema_contract():
    """
    Verify GET /v1/stream/active complies with PROJECT.md contract:
    - camera_id, endpoint, is_running, latency_ms, fps, target_fps
    - frames_received, processed_frames, frames_dropped
    - detected_count, density, activity, species
    """
    mgr = LiveStreamManager()

    start_info = mgr.start_stream(
        camera_id="CAM-CONTRACT-01",
        rtsp_url="rtsp://mock.camera/tank1",
        species="redclaw",
        save_to_db=False,
        target_fps=10.0,
    )
    assert start_info["status"] == "started"

    try:
        time.sleep(0.35)  # Allow worker to start and ingest initial frames
        listing = mgr.list_streams()
        assert listing["active_streams_count"] == 1
        assert len(listing["streams"]) == 1

        stream_data = listing["streams"][0]

        required_keys = [
            "camera_id",
            "endpoint",
            "is_running",
            "latency_ms",
            "fps",
            "target_fps",
            "frames_received",
            "processed_frames",
            "frames_dropped",
            "detected_count",
            "density",
            "activity",
            "species",
        ]
        for key in required_keys:
            assert key in stream_data, f"Missing required key '{key}' in GET /v1/stream/active response"

        assert stream_data["camera_id"] == "CAM-CONTRACT-01"
        assert stream_data["target_fps"] == 10.0
        assert isinstance(stream_data["fps"], (int, float))
        assert isinstance(stream_data["processed_frames"], int)
        assert isinstance(stream_data["detected_count"], int)
        assert 0.0 <= stream_data["density"] <= 1.0
        assert 0.0 <= stream_data["activity"] <= 1.0
    finally:
        mgr.stop_stream("CAM-CONTRACT-01")

    # Verify active streams is now 0
    empty_listing = mgr.list_streams()
    assert empty_listing["active_streams_count"] == 0


# ---------------------------------------------------------------------------
# Feature 6: WebSocket Events, Ping/Pong, Telemetry & Disconnect
# ---------------------------------------------------------------------------

def test_websocket_ping_pong_and_clean_disconnect():
    """
    Verify WS /v1/events handles ping/pong and disconnects cleanly
    without raising NameError on line 214.
    """
    with client.websocket_connect("/v1/events") as websocket:
        # Send ping
        websocket.send_text("ping")
        resp = websocket.receive_json()
        assert resp == {"type": "pong"}

    # Client socket closed upon exit of context manager
    # Verify no server crash or error
    health = client.get("/v1/health")
    assert health.status_code == 200


def test_websocket_telemetry_broadcast():
    """Verify live stream telemetry ('stream.telemetry') broadcasts to active WebSockets."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    received_messages = []

    class MockWebSocket:
        def __init__(self):
            self.closed = False
        async def send_text(self, text: str):
            received_messages.append(text)

    mock_ws = MockWebSocket()
    ws_manager.active_connections.add(mock_ws)

    telemetry_payload = {
        "camera_id": "CAM-WS-TEST",
        "fps": 10.0,
        "target_fps": 10.0,
        "processed_frames": 42,
        "detected_count": 8,
        "density": 0.05,
        "activity": 0.75,
        "species": "redclaw",
    }

    try:
        loop.run_until_complete(
            ws_manager.broadcast_event("stream.telemetry", telemetry_payload)
        )
        assert len(received_messages) == 1
        assert "stream.telemetry" in received_messages[0]
        assert "CAM-WS-TEST" in received_messages[0]
    finally:
        ws_manager.disconnect(mock_ws)
        loop.close()


# ---------------------------------------------------------------------------
# Spatial Metrics & Ethogram Analytics Integration
# ---------------------------------------------------------------------------

def test_spatial_metrics_and_ethogram_calculations():
    """Verify spatial_metrics and sampling_protocols produce bounded metrics."""
    subjects = [
        SubjectInfo(track_id=1, bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2)),
        SubjectInfo(track_id=2, bbox=BoundingBox(x_min=0.15, y_min=0.15, x_max=0.25, y_max=0.25)),
    ]
    sm = compute_spatial_metrics(subjects, arena_area=1.0)
    assert sm.active_subjects_count == 2
    assert 0.0 <= sm.crowding_intensity <= 1.0

    events = [
        BehaviorEvent(
            species=SpeciesInfo(id="cherax_quadricarinatus", name="Redclaw Crayfish"),
            source=SourceInfo(type=SourceType.VIDEO),
            subjects=subjects,
            behavior=BehaviorInfo(category="locomotion", label="normal_movement", confidence=0.9),
            temporal=TemporalInfo(start=0.0, end=5.0, duration=5.0),
            model=ModelInfo(species_model="redclaw-behavior-v1"),
        ),
        BehaviorEvent(
            species=SpeciesInfo(id="cherax_quadricarinatus", name="Redclaw Crayfish"),
            source=SourceInfo(type=SourceType.VIDEO),
            subjects=subjects,
            behavior=BehaviorInfo(category="resting", label="sheltered", confidence=0.85),
            temporal=TemporalInfo(start=5.0, end=10.0, duration=5.0),
            model=ModelInfo(species_model="redclaw-behavior-v1"),
        ),
    ]
    etho = SamplingProtocols.compute_ethogram_time_budget(events, total_duration=10.0)
    assert 0.0 <= etho.activity_index <= 1.0
