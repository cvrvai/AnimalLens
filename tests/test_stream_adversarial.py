"""
Milestone 1 Adversarial Stress Test Suite:
Stream Worker, Target FPS Throttling, Video Looping, Tracker Memory Safety, and WebSocket Telemetry.

Authored by: m1_challenger_2 (Adversarial Challenger Agent)
Focus Areas:
1. Target FPS throttling at boundary values (1 fps, 15 fps) and clamping of out-of-bounds inputs (-5, 0, 100).
2. Continuous simulated video loops past EOF under high frame volume.
3. Tracker memory safety (TrackItem.history bounded to <= 50 over 500 updates).
4. WebSocket telemetry broadcast concurrency and rapid disconnect stress.
"""
from __future__ import annotations

import asyncio
import io
import time
from typing import Any, Dict, List
import cv2
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from animallens.core.schemas import BoundingBox
from animallens.perception.base import DetectionResult
from animallens.perception.models.botsort_tracker import BoTSORTTracker, TrackItem
from animallens.server.app import app
from animallens.server.stream_manager import LiveStreamManager, StreamWorker
from animallens.server.websocket import ConnectionManager, ws_manager
from animallens.sources.stream import StreamSource

client = TestClient(app)


# ===========================================================================
# 1. Target FPS Throttling & Out-of-Bound Clamping Tests
# ===========================================================================

def test_fps_throttling_boundary_1_fps():
    """Verify StreamSource at lower boundary (1.0 fps) produces clamped target_fps and paced frames."""
    source = StreamSource(
        endpoint="rtsp://mock.camera/fps1",
        target_fps=1.0,
        camera_id="CAM-FPS-1",
        buffer_size=2,
    )
    try:
        assert source.target_fps == 1.0
        assert source.fps == 1.0
        metrics = source.get_metrics()
        assert metrics["target_fps"] == 1.0
        assert metrics["fps"] == 1.0
    finally:
        source.stop()


def test_fps_throttling_boundary_15_fps():
    """Verify StreamSource at upper boundary (15.0 fps) produces clamped target_fps."""
    source = StreamSource(
        endpoint="rtsp://mock.camera/fps15",
        target_fps=15.0,
        camera_id="CAM-FPS-15",
        buffer_size=2,
    )
    try:
        assert source.target_fps == 15.0
        assert source.fps == 15.0
        metrics = source.get_metrics()
        assert metrics["target_fps"] == 15.0
        assert metrics["fps"] == 15.0
    finally:
        source.stop()


def test_fps_out_of_bounds_clamping_negative_and_zero():
    """Verify out-of-bound target_fps (-5.0 and 0.0) are clamped to minimum 1.0 fps."""
    for bad_fps in [-5.0, -1.0, 0.0]:
        source = StreamSource(
            endpoint="rtsp://mock.camera/bad_fps",
            target_fps=bad_fps,
            camera_id="CAM-CLAMP-MIN",
        )
        try:
            assert source.target_fps == 1.0, f"Expected 1.0 fps clamp for input {bad_fps}, got {source.target_fps}"
            assert source.fps == 1.0
            metrics = source.get_metrics()
            assert metrics["target_fps"] == 1.0
        finally:
            source.stop()


def test_fps_out_of_bounds_clamping_excessive():
    """Verify out-of-bound target_fps (100.0, 1000.0) are clamped to maximum 15.0 fps."""
    for excessive_fps in [16.0, 30.0, 100.0, 1000.0]:
        source = StreamSource(
            endpoint="rtsp://mock.camera/high_fps",
            target_fps=excessive_fps,
            camera_id="CAM-CLAMP-MAX",
        )
        try:
            assert source.target_fps == 15.0, f"Expected 15.0 fps clamp for input {excessive_fps}, got {source.target_fps}"
            assert source.fps == 15.0
            metrics = source.get_metrics()
            assert metrics["target_fps"] == 15.0
        finally:
            source.stop()


def test_stream_worker_target_fps_contract_consistency():
    """
    ADVERSARIAL CHECK:
    Verify StreamWorker preserves target_fps clamping in get_status() and does not leak unclamped raw values.
    Asserts:
    1. StreamWorker(..., target_fps=100.0) clamps to 15.0 fps (status['target_fps'] == 15.0, current_fps == 15.0).
    2. StreamWorker(..., target_fps=-5.0) clamps to 1.0 fps (status['target_fps'] == 1.0, current_fps == 1.0).
    """
    # 1. Upper bound clamping: 100.0 -> 15.0
    worker_high = StreamWorker(
        camera_id="CAM-WORKER-CLAMP-HIGH",
        rtsp_url="rtsp://mock.camera/test_high",
        target_fps=100.0,
    )
    try:
        assert worker_high.target_fps == 15.0, (
            f"Expected worker_high.target_fps == 15.0, got {worker_high.target_fps}"
        )
        assert worker_high.current_fps == 15.0, (
            f"Expected worker_high.current_fps == 15.0, got {worker_high.current_fps}"
        )
        status_high = worker_high.get_status()
        assert status_high["target_fps"] == 15.0, (
            f"StreamWorker.get_status() exposed unclamped target_fps={status_high['target_fps']} "
            f"(expected 15.0)."
        )
        assert status_high["fps"] == 15.0
    finally:
        worker_high.stop()

    # 2. Lower bound clamping: -5.0 -> 1.0
    worker_low = StreamWorker(
        camera_id="CAM-WORKER-CLAMP-LOW",
        rtsp_url="rtsp://mock.camera/test_low",
        target_fps=-5.0,
    )
    try:
        assert worker_low.target_fps == 1.0, (
            f"Expected worker_low.target_fps == 1.0, got {worker_low.target_fps}"
        )
        assert worker_low.current_fps == 1.0, (
            f"Expected worker_low.current_fps == 1.0, got {worker_low.current_fps}"
        )
        status_low = worker_low.get_status()
        assert status_low["target_fps"] == 1.0, (
            f"StreamWorker.get_status() exposed unclamped target_fps={status_low['target_fps']} "
            f"(expected 1.0)."
        )
        assert status_low["fps"] == 1.0
    finally:
        worker_low.stop()


# ===========================================================================
# 2. Continuous Video Loop Stress Testing
# ===========================================================================

def test_video_loop_stress_continuous_rewind(tmp_path):
    """
    Stress test video file looping over 60 iterations of a tiny 3-frame video (20 rewind cycles).
    Verifies that StreamSource does not stall, crash, or drop connections on EOF.
    """
    video_path = str(tmp_path / "adversarial_loop.mp4")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (80, 80))
    for i in range(3):
        frame = np.full((80, 80, 3), i * 70, dtype=np.uint8)
        out.write(frame)
    out.release()

    source = StreamSource(endpoint=video_path, target_fps=15.0, buffer_size=2)
    frames_read = 0
    t0 = time.time()

    for ts, frame in source:
        frames_read += 1
        assert frame is not None
        assert frame.size == (80, 80)
        if frames_read >= 60:
            break
        # Guard against infinite stall
        if time.time() - t0 > 15.0:
            pytest.fail("Video loop reader stalled during continuous rewind stress test")

    source.stop()
    assert frames_read == 60
    assert source.total_frames_received >= 60


def test_stream_source_buffer_congestion_bounded():
    """Verify that slow consumers do not cause unbounded buffer accumulation; oldest frames dropped."""
    source = StreamSource(
        endpoint="rtsp://mock.camera/congestion",
        target_fps=15.0,
        buffer_size=2,
    )
    source.start()
    # Sleep to allow the background thread to push multiple frames while consumer is idle
    time.sleep(0.3)
    source.stop()

    # Queue maxsize must strictly be 2
    assert source._frame_queue.maxsize == 2
    assert source._frame_queue.qsize() <= 2
    # Multiple frames should have been received and excess dropped
    assert source.total_frames_received > 2
    assert source.total_frames_dropped > 0


# ===========================================================================
# 3. Tracker Memory Safety Stress Testing
# ===========================================================================

def test_tracker_history_stress_500_frames():
    """
    Stress-test BoTSORT tracker memory safety over 500 consecutive frame updates.
    Verifies TrackItem.history does not leak memory and is strictly capped at <= 50.
    """
    tracker = BoTSORTTracker()
    bbox = BoundingBox(x_min=0.3, y_min=0.3, x_max=0.5, y_max=0.5)

    for frame_idx in range(500):
        det = DetectionResult(
            bboxes=[bbox],
            confidences=[0.92],
            class_names=["crayfish"],
        )
        tracker.update(det, timestamp=float(frame_idx) * 0.066)

        # Invariant check: all active tracks must have history <= 50
        for track in tracker.tracks:
            assert len(track.history) <= 50, (
                f"Memory leak detected at frame {frame_idx}: TrackItem.history length is {len(track.history)} (> 50)"
            )

    active = tracker.active_tracks
    assert len(active) == 1
    assert len(active[0].history) == 50


def test_tracker_unmatched_tracks_pruned():
    """Verify tracks that stop receiving updates are pruned after max_age frames, preventing memory leaks."""
    tracker = BoTSORTTracker(max_age=10)
    det = DetectionResult(
        bboxes=[BoundingBox(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2)],
        confidences=[0.9],
        class_names=["crayfish"],
    )
    tracker.update(det, timestamp=0.0)
    assert len(tracker.tracks) == 1

    # Send 15 empty frames
    empty_det = DetectionResult(bboxes=[], confidences=[], class_names=[])
    for i in range(1, 16):
        tracker.update(empty_det, timestamp=float(i) * 0.1)

    # Track must be completely pruned
    assert len(tracker.tracks) == 0, f"Expected 0 tracks after exceeding max_age=10, found {len(tracker.tracks)}"


# ===========================================================================
# 4. WebSocket Disconnect Stress & Telemetry Concurrency
# ===========================================================================

def test_websocket_rapid_connect_disconnect_stress():
    """Rapidly connect and disconnect 25 WebSocket clients in succession; verify zero NameErrors or server crashes."""
    for i in range(25):
        with client.websocket_connect("/v1/events") as ws:
            ws.send_text("ping")
            resp = ws.receive_json()
            assert resp == {"type": "pong"}
        # Exiting context manager triggers disconnect

    # Verify server remains healthy
    health = client.get("/v1/health")
    assert health.status_code == 200


@pytest.mark.asyncio
async def test_websocket_broadcast_concurrent_modification_safety():
    """
    ADVERSARIAL CONCURRENCY CHECK:
    Verifies that ws_manager.broadcast_event() does NOT raise 'RuntimeError: Set changed size during iteration'
    when clients disconnect concurrently during active event broadcasting.
    """
    cm = ConnectionManager()

    class SlowMockWebSocket:
        def __init__(self, idx: int):
            self.idx = idx
            self.messages: List[str] = []

        async def send_text(self, text: str):
            # Simulate slight network delay yielding control to event loop
            await asyncio.sleep(0.01)
            self.messages.append(text)

    s1 = SlowMockWebSocket(1)
    s2 = SlowMockWebSocket(2)
    await cm.connect(s1)
    await cm.connect(s2)

    # Coroutine 1: broadcasts an event to active connections
    async def broadcast_task():
        await cm.broadcast_event("test.event", {"hello": "world"})

    # Coroutine 2: concurrent client disconnect while broadcast is sleeping in send_text
    async def disconnect_task():
        await asyncio.sleep(0.005)
        cm.disconnect(s1)

    # Run concurrently - list(self.active_connections) guarantees no RuntimeError
    await asyncio.gather(broadcast_task(), disconnect_task())

    assert s1 not in cm.active_connections
    assert s2 in cm.active_connections
    assert len(s2.messages) == 1


@pytest.mark.asyncio
async def test_stream_worker_resilience_to_broadcast_exceptions():
    """
    ADVERSARIAL STABILITY CHALLENGE:
    Verify whether ConnectionManager safely isolates bad sockets and StreamWorker handles disconnects.
    """
    worker = StreamWorker(
        camera_id="CAM-RESILIENCE",
        rtsp_url="rtsp://mock.camera/resilience",
        target_fps=15.0,
    )

    class CrashingMockWebSocket:
        async def send_text(self, text: str):
            raise ConnectionResetError("Client abruptly severed TCP connection")

    bad_ws = CrashingMockWebSocket()
    ws_manager.active_connections.add(bad_ws)

    try:
        # Broadcasting should safely handle the exception and remove bad_ws
        await ws_manager.broadcast_event("stream.telemetry", {"status": "ok"})
        assert bad_ws not in ws_manager.active_connections
    finally:
        ws_manager.disconnect(bad_ws)
        worker.stop()


@pytest.mark.asyncio
async def test_stream_worker_crash_shielding_under_broadcast_failure():
    """
    ADVERSARIAL STABILITY CHECK:
    Verify that StreamWorker._run_loop isolates WebSocket broadcast exceptions
    and continues running uninterrupted when ws_manager.broadcast_event raises.
    """
    from unittest.mock import patch

    worker = StreamWorker(
        camera_id="CAM-SHIELD-TEST",
        rtsp_url="rtsp://mock.camera/shield",
        target_fps=15.0,
        save_to_db=False,
    )

    dummy_img = Image.new("RGB", (64, 64), color="green")
    mock_frames = [(time.time() + i * 0.066, dummy_img) for i in range(5)]

    class MockStreamSource(StreamSource):
        async def __aiter__(self):
            for f in mock_frames:
                yield f

    worker.stream_source = MockStreamSource(
        endpoint="rtsp://mock.camera/shield",
        target_fps=15.0,
        camera_id="CAM-SHIELD-TEST",
    )

    fail_count = 0

    async def faulty_broadcast(event_type, data):
        nonlocal fail_count
        fail_count += 1
        raise ConnectionResetError("Simulated unhandled WebSocket transmission crash")

    with patch("animallens.server.stream_manager.ws_manager.broadcast_event", side_effect=faulty_broadcast):
        await worker._run_loop()

    # Invariant checks:
    # 1. Faulty broadcast was invoked and shielded
    assert fail_count >= 5, f"Expected at least 5 broadcast attempts, got {fail_count}"
    # 2. Worker processed all 5 frames without terminating early
    assert worker.processed_frames == 5, f"Worker terminated early! processed_frames={worker.processed_frames}"
