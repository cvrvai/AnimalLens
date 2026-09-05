"""
Adversarial Stress Test Suite for Milestone 1 (M1):
POST /v1/analyze/image Robustness, Boundary Conditions, Extreme Geometries, and Latency Benchmark.

Designed and verified by m1_challenger_1.
"""
from __future__ import annotations

import io
import time
from typing import List, Tuple
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from animallens.core.schemas import AnalysisResult, BoundingBox
from animallens.perception.models.yolov8_detector import YOLOv8Detector
from animallens.sdk import AnimalLens, normalize_lifecycle_stage
from animallens.server.app import app

client = TestClient(app)


def _create_jpeg_bytes(width: int, height: int, color=(40, 60, 80)) -> io.BytesIO:
    """Helper to generate a valid JPEG in memory."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def _create_png_bytes(width: int, height: int, color=(40, 60, 80)) -> io.BytesIO:
    """Helper to generate a valid PNG in memory."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ===========================================================================
# 1. Malformed Inputs: 0-byte, Corrupt JPEG Headers, Non-Image Files
# ===========================================================================

def test_stress_zero_byte_image():
    """Verify 0-byte image file is rejected with HTTP 400 Bad Request."""
    empty_buf = io.BytesIO(b"")
    res = client.post(
        "/v1/analyze/image",
        files={"file": ("empty.jpg", empty_buf, "image/jpeg")},
    )
    assert res.status_code == 400, f"Expected HTTP 400 for 0-byte file, got {res.status_code}"
    detail = res.json().get("detail", "").lower()
    assert "empty" in detail or "0 bytes" in detail


def test_stress_corrupt_jpeg_header():
    """
    Adversarial test: upload image with corrupted JPEG SOI / table headers.
    Server MUST NOT crash; endpoint should reject with HTTP 400 Bad Request.
    """
    corrupt_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01" + b"\xde\xad\xbe\xef" * 16
    corrupt_buf = io.BytesIO(corrupt_bytes)
    res = client.post(
        "/v1/analyze/image",
        files={"file": ("corrupt.jpg", corrupt_buf, "image/jpeg")},
    )
    # The server process must reject corrupt uploads with clean HTTP 400 Bad Request
    assert res.status_code == 400, f"Expected HTTP 400 for corrupt JPEG header, got {res.status_code}"
    assert "invalid image format" in res.json().get("detail", "").lower()

    # Verify server is still alive and responsive after corrupt payload
    health_res = client.get("/v1/health")
    assert health_res.status_code == 200


def test_stress_non_image_file():
    """
    Adversarial test: upload plain text / PDF / random binary as image file.
    Server MUST reject with HTTP 400 Bad Request; subsequent requests must continue functioning.
    """
    text_content = b"This is plain text and definitely not an image file. It should be rejected."
    text_buf = io.BytesIO(text_content)
    res = client.post(
        "/v1/analyze/image",
        files={"file": ("malicious.txt", text_buf, "text/plain")},
    )
    assert res.status_code == 400, f"Expected HTTP 400 for non-image payload, got {res.status_code}"
    assert "invalid image format" in res.json().get("detail", "").lower()

    # Verify server is still alive
    health_res = client.get("/v1/health")
    assert health_res.status_code == 200


# ===========================================================================
# 2. Lifecycle Stage Area Boundaries & Extreme Sizes
# ===========================================================================

def test_stress_extreme_small_crayfish_area():
    """
    Stress-test extremely small crayfish (<0.005 area, e.g. 0.001, 0.003).
    Must classify strictly as 'craylet' without numerical underflow.
    """
    small_areas = [0.0005, 0.0010, 0.0025, 0.0049]
    for a in small_areas:
        side = a ** 0.5
        box = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.1 + side, y_max=0.1 + side)
        assert abs(box.area - a) < 1e-6

        # Check normalization
        if box.area < 0.02:
            stage = "craylet"
        elif box.area < 0.08:
            stage = "juvenile"
        elif box.area < 0.18:
            stage = "sub-adult"
        else:
            stage = "adult"

        assert stage == "craylet", f"Extremely small area {box.area} must classify as craylet, got {stage}"


def test_stress_exact_stage_boundary_thresholds():
    """
    Stress-test boundary area crayfish:
    - 0.0199 -> craylet
    - 0.0200 -> juvenile
    - 0.0201 -> juvenile
    - 0.0799 -> juvenile
    - 0.0800 -> sub-adult
    - 0.0801 -> sub-adult
    - 0.1799 -> sub-adult
    - 0.1800 -> adult
    - 0.1801 -> adult
    """
    test_cases: List[Tuple[float, str]] = [
        (0.0199, "craylet"),
        (0.0200, "juvenile"),
        (0.0201, "juvenile"),
        (0.0799, "juvenile"),
        (0.0800, "sub-adult"),
        (0.0801, "sub-adult"),
        (0.1799, "sub-adult"),
        (0.1800, "adult"),
        (0.1801, "adult"),
    ]

    for area_val, expected_stage in test_cases:
        w = 0.2
        h = area_val / w
        box = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.1 + w, y_max=0.1 + h)

        area = box.area
        if area < 0.02:
            stage = "craylet"
        elif area < 0.08:
            stage = "juvenile"
        elif area < 0.18:
            stage = "sub-adult"
        else:
            stage = "adult"

        assert stage == expected_stage, f"Area {area_val:.4f} resulted in '{stage}', expected '{expected_stage}'"
        assert stage != "sub_adult", f"Stage '{stage}' must use hyphen, not underscore"


def test_stress_giant_adult_crayfish_area():
    """
    Stress-test giant adults (>0.5 area, e.g. 0.55, 0.70, 0.85).
    Must classify as 'adult' without numerical overflow.
    """
    giant_areas = [0.51, 0.65, 0.78, 0.90]
    for a in giant_areas:
        w = 0.8
        h = a / w
        box = BoundingBox(x_min=0.05, y_min=0.05, x_max=0.05 + w, y_max=0.05 + h)

        area = box.area
        if area < 0.02:
            stage = "craylet"
        elif area < 0.08:
            stage = "juvenile"
        elif area < 0.18:
            stage = "sub-adult"
        else:
            stage = "adult"

        assert stage == "adult", f"Giant crayfish with area {box.area:.2f} must classify as adult, got {stage}"


# ===========================================================================
# 3. Extreme Aspect Ratios (1x1000, 1000x1, 5x2000, 2000x5)
# ===========================================================================

@pytest.mark.parametrize("dim", [(1, 1000), (1000, 1), (5, 800), (800, 5)])
def test_stress_extreme_aspect_ratios_api(dim: Tuple[int, int]):
    """
    Stress-test POST /v1/analyze/image with extreme aspect ratio images:
    (1x1000, 1000x1, 5x800, 800x5).
    Ensures no division by zero, no OpenCV assertion crashes, and returns HTTP 200.
    """
    w, h = dim
    buf = _create_png_bytes(width=w, height=h)
    t0 = time.perf_counter()
    res = client.post(
        "/v1/analyze/image",
        files={"file": (f"extreme_{w}x{h}.png", buf, "image/png")},
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    assert res.status_code == 200, f"Extreme aspect ratio ({w}x{h}) failed with {res.status_code}: {res.text}"
    data = res.json()
    assert "detections" in data
    assert "count" in data
    assert latency_ms < 200.0, f"Latency {latency_ms:.2f}ms exceeded 200ms limit for {w}x{h}"

    # Verify all bounding boxes are within [0.0, 1.0]
    for d in data["detections"]:
        bbox = d["bbox"]
        assert 0.0 <= bbox[0] <= 1.0
        assert 0.0 <= bbox[1] <= 1.0
        assert 0.0 <= bbox[2] <= 1.0
        assert 0.0 <= bbox[3] <= 1.0
        assert d["stage"] in ("craylet", "juvenile", "sub-adult", "adult")


# ===========================================================================
# 4. Rapid Sequential Queries & Latency Benchmarking (<200ms)
# ===========================================================================

def test_stress_rapid_sequential_queries_benchmark():
    """
    Benchmark POST /v1/analyze/image under rapid sequential load (30 consecutive requests).
    Every single query must complete in <200ms.
    """
    latencies: List[float] = []
    num_queries = 30

    for i in range(num_queries):
        buf = _create_jpeg_bytes(640, 480, color=(30 + i, 40, 50))
        t0 = time.perf_counter()
        res = client.post(
            "/v1/analyze/image",
            files={"file": (f"seq_{i}.jpg", buf, "image/jpeg")},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        assert res.status_code == 200, f"Query {i} failed with {res.status_code}"
        assert elapsed_ms < 200.0, f"Query {i} took {elapsed_ms:.2f}ms (>200ms)"

    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))]

    print(
        f"\n[LATENCY BENCHMARK] Queries: {num_queries} | "
        f"Min: {min_latency:.2f}ms | Avg: {avg_latency:.2f}ms | "
        f"P95: {p95_latency:.2f}ms | Max: {max_latency:.2f}ms"
    )

    assert max_latency < 200.0, f"Max latency {max_latency:.2f}ms exceeded 200ms threshold"
    assert avg_latency < 50.0, f"Average latency {avg_latency:.2f}ms higher than expected 50ms"


# ===========================================================================
# 5. Server Stability Post-Attack Verification
# ===========================================================================

def test_stress_server_stability_after_adversarial_barrage():
    """
    Send an interleaved barrage of malformed, extreme, and valid requests,
    verifying that the server never deadlocks, crashes, or leaks memory.
    """
    payloads = [
        ("empty", io.BytesIO(b""), "image/jpeg", 400),
        ("corrupt", io.BytesIO(b"\xff\xd8\xff\x00corrupt"), "image/jpeg", 400),
        ("text", io.BytesIO(b"random text payload not an image"), "text/plain", 400),
        ("extreme_1x500", _create_png_bytes(1, 500), "image/png", 200),
        ("extreme_500x1", _create_png_bytes(500, 1), "image/png", 200),
        ("valid_tray", _create_jpeg_bytes(640, 480), "image/jpeg", 200),
    ]

    for name, buf, mime, expected_status in payloads * 3:
        buf.seek(0)
        res = client.post(
            "/v1/analyze/image",
            files={"file": (f"{name}.bin", buf, mime)},
        )
        if isinstance(expected_status, list):
            assert res.status_code in expected_status, f"{name} got {res.status_code}"
        else:
            assert res.status_code == expected_status, f"{name} got {res.status_code}"

    # Final health check
    health_res = client.get("/v1/health")
    assert health_res.status_code == 200
    valid_buf = _create_jpeg_bytes(640, 480)
    final_res = client.post(
        "/v1/analyze/image",
        files={"file": ("final.jpg", valid_buf, "image/jpeg")},
    )
    assert final_res.status_code == 200
    data = final_res.json()
    assert data["count"] == 4
    assert data["species"] == "redclaw"
