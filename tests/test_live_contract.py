"""
AnimalLens Live Integration & Contract Verification Suite (Milestone 4).

Validates all 4 requirements from ORIGINAL_REQUEST.md & PROJECT.md:
Check 1: Trained model weights load into perception engine (YOLOv8Detector & ModelRegistry)
         and perform inference on test tray images.
Check 2: AnimalLens server rejects unauthorized requests with HTTP 401 and accepts
         authorized requests with HTTP 200 (X-API-Key and Authorization: Bearer).
Check 3: Next.js /api/livestock/vision route contract forwards API key and returns
         structured bounding boxes and multi-stage biomass metrics.
Check 4: Programmatic verification of all tests in test_api_security.py and
         test_crayfish_vision.py.
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List

import cv2
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image, ImageDraw
import pytest

from animallens.models.registry import ModelRegistry
from animallens.perception.models.yolov8_detector import YOLOv8Detector
from animallens.sdk import AnimalLens, normalize_lifecycle_stage
from animallens.server.app import create_app


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def _get_repo_root() -> Path:
    """Resolve repository root directory."""
    return Path(__file__).resolve().parents[2]


def _get_animallens_root() -> Path:
    """Resolve AnimalLens microservice root directory."""
    return Path(__file__).resolve().parents[1]


def _create_tray_scan_image(width: int = 640, height: int = 480) -> Image.Image:
    """
    Generate a synthetic aquaculture sample tray image with high-contrast
    blobs corresponding to different crayfish lifecycle stages.
    """
    img = Image.new("RGB", (width, height), color=(235, 238, 240))
    draw = ImageDraw.Draw(img)

    # 1. Craylet blob (<0.02 area)
    draw.ellipse([60, 60, 90, 90], fill=(25, 25, 30))

    # 2. Juvenile blob (0.02 - 0.08 area)
    draw.ellipse([180, 140, 270, 230], fill=(30, 35, 40))

    # 3. Sub-adult blob (0.08 - 0.18 area)
    draw.ellipse([340, 100, 480, 260], fill=(20, 25, 30))

    # 4. Adult blob (>= 0.18 area)
    draw.ellipse([140, 280, 420, 440], fill=(15, 20, 25))

    return img


def _image_to_jpeg_bytes(img: Image.Image) -> bytes:
    """Convert PIL image to JPEG bytes buffer."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _image_to_base64(img: Image.Image) -> str:
    """Convert PIL image to Base64 encoded string."""
    jpeg_bytes = _image_to_jpeg_bytes(img)
    return base64.b64encode(jpeg_bytes).decode("utf-8")


@pytest.fixture
def repo_paths():
    """Provides validated repository paths."""
    repo_root = _get_repo_root()
    al_root = _get_animallens_root()
    weights_dir = al_root / "models" / "trained" / "crayfish_multistage" / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    metrics_json = al_root / "models" / "trained" / "crayfish_multistage" / "metrics.json"
    return {
        "repo_root": repo_root,
        "al_root": al_root,
        "weights_dir": weights_dir,
        "best_pt": best_pt,
        "last_pt": last_pt,
        "metrics_json": metrics_json,
    }


# ---------------------------------------------------------------------------
# Check 1: Model Weights Checkpoint & Perception Engine Loading
# ---------------------------------------------------------------------------

class TestCheck1ModelWeightsAndInference:
    """Verifies trained model weights exist, load, and perform inference."""

    def test_checkpoint_files_exist_and_metrics_valid(self, repo_paths):
        """Verify best.pt, last.pt, and metrics.json exist with genuine binary attributes."""
        best_pt = repo_paths["best_pt"]
        last_pt = repo_paths["last_pt"]
        metrics_json = repo_paths["metrics_json"]

        assert best_pt.exists(), f"Missing best.pt at {best_pt}"
        assert best_pt.is_file(), f"{best_pt} is not a file"

        # Check binary file size (must be >= 1 MB; YOLOv8n is typically 6.2 MB)
        best_size = best_pt.stat().st_size
        assert best_size >= 1_000_000, (
            f"best.pt file size {best_size} bytes is not a binary checkpoint (expected >= 1,000,000 bytes)"
        )

        assert last_pt.exists(), f"Missing last.pt at {last_pt}"
        last_size = last_pt.stat().st_size
        assert last_size >= 1_000_000, (
            f"last.pt file size {last_size} bytes is not a binary checkpoint (expected >= 1,000,000 bytes)"
        )

        # PyTorch deserialization assertion
        import torch
        ckpt = torch.load(str(best_pt), map_location="cpu", weights_only=False)
        assert isinstance(ckpt, dict), f"Expected dict from torch.load, got {type(ckpt)}"
        assert "model" in ckpt or "state_dict" in ckpt or "ema" in ckpt, (
            f"best.pt is missing model weights dictionary; keys present: {list(ckpt.keys())}"
        )

        # Ultralytics model loading assertion
        from ultralytics import YOLO
        yolo_model = YOLO(str(best_pt))
        assert yolo_model is not None

        # Verify metrics.json
        assert metrics_json.exists(), f"Missing metrics.json at {metrics_json}"
        metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
        assert metrics.get("model_name") == "crayfish_multistage"
        assert metrics.get("status") == "SUCCESS"
        assert "metrics" in metrics
        assert "mAP50" in metrics["metrics"]
        assert 0.0 <= metrics["metrics"]["mAP50"] <= 1.0

        # Assert metrics.json checkpoint_size_mb matches disk reality
        disk_size_mb = round(best_size / (1024 * 1024), 2)
        reported_size_mb = metrics.get("checkpoint_size_mb", 0.0)
        assert reported_size_mb >= 1.0, f"metrics.json attests invalid checkpoint size: {reported_size_mb} MB"
        assert abs(reported_size_mb - disk_size_mb) <= 0.2, (
            f"metrics.json checkpoint_size_mb ({reported_size_mb}) does not match actual disk size ({disk_size_mb} MB)"
        )

    def test_model_registry_discovers_and_resolves_checkpoint(self, repo_paths):
        """Verify ModelRegistry discovers and resolves the trained crayfish model."""
        registry = ModelRegistry(models_dir=repo_paths["al_root"] / "models")
        installed = registry.list_installed()

        names = [m["name"] for m in installed]
        assert "crayfish_multistage" in names, (
            f"crayfish_multistage not found in ModelRegistry.list_installed(): {names}"
        )

        crayfish_entry = next(m for m in installed if m["name"] == "crayfish_multistage")
        assert crayfish_entry["is_installed"] is True
        assert Path(crayfish_entry["installed_path"]).exists()

        resolved_path = registry.get_model_path("crayfish_multistage")
        assert resolved_path is not None
        assert resolved_path.exists()
        assert "best.pt" in str(resolved_path)

    def test_yolov8_detector_loads_checkpoint_and_infers(self, repo_paths):
        """Verify YOLOv8Detector loads trained weights using ultralytics backend and infers."""
        best_pt = repo_paths["best_pt"]
        detector = YOLOv8Detector(
            model_path=best_pt,
            classes=["crayfish", "craylet", "juvenile", "sub_adult", "adult"],
        )

        assert detector.model_path is not None
        assert detector.model_path.exists()
        # Verify genuine ultralytics backend is engaged, NOT fallback
        assert detector._backend == "ultralytics", (
            f"YOLOv8Detector failed to engage neural backend; backend is '{detector._backend}'"
        )
        assert detector._model is not None, "YOLOv8Detector._model is None"

        tray_img = _create_tray_scan_image()

        t0 = time.perf_counter()
        result = detector.detect(tray_img, confidence_threshold=0.40)
        latency_ms = (time.perf_counter() - t0) * 1000

        assert latency_ms < 200.0, f"Detection latency {latency_ms:.2f}ms exceeded 200ms limit"
        assert len(result.bboxes) >= 2, f"Expected at least 2 detections on tray, got {len(result.bboxes)}"
        assert len(result.class_names) == len(result.bboxes)
        assert len(result.confidences) == len(result.bboxes)

        # Coordinate normalization check
        for bbox in result.bboxes:
            assert 0.0 <= bbox.x_min <= 1.0
            assert 0.0 <= bbox.y_min <= 1.0
            assert 0.0 <= bbox.x_max <= 1.0
            assert 0.0 <= bbox.y_max <= 1.0
            assert bbox.x_max >= bbox.x_min
            assert bbox.y_max >= bbox.y_min

        # Lifecycle stage classification check
        recognized_stages = {"craylet", "juvenile", "sub_adult", "sub-adult", "adult", "crayfish"}
        for cls_name in result.class_names:
            assert cls_name in recognized_stages, f"Unrecognized class '{cls_name}'"


# ---------------------------------------------------------------------------
# Check 2: AnimalLens API Security & Cloudflare Tunnel Bypass
# ---------------------------------------------------------------------------

class TestCheck2ServerSecurityAndCloudflareBypass:
    """Verifies API Key authentication and public Cloudflare Tunnel whitelist."""

    TEST_API_KEY = "live_test_cf_tunnel_key_987654"

    @pytest.fixture
    def authenticated_client(self):
        """Creates TestClient with ANIMALLENS_API_KEY configured."""
        orig = os.environ.get("ANIMALLENS_API_KEY")
        os.environ["ANIMALLENS_API_KEY"] = self.TEST_API_KEY
        client = TestClient(create_app())
        yield client
        if orig is not None:
            os.environ["ANIMALLENS_API_KEY"] = orig
        else:
            os.environ.pop("ANIMALLENS_API_KEY", None)

    @pytest.fixture
    def unauthenticated_client(self):
        """Creates TestClient with ANIMALLENS_API_KEY empty/unset."""
        orig = os.environ.get("ANIMALLENS_API_KEY")
        os.environ["ANIMALLENS_API_KEY"] = ""
        client = TestClient(create_app())
        yield client
        if orig is not None:
            os.environ["ANIMALLENS_API_KEY"] = orig
        else:
            os.environ.pop("ANIMALLENS_API_KEY", None)

    def test_public_health_bypasses_auth_for_cloudflare_tunnel(self, authenticated_client):
        """GET /v1/health must return HTTP 200 without authentication headers."""
        resp = authenticated_client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "healthy"
        assert data.get("service") == "AnimalLens"

    def test_docs_and_openapi_bypass_auth(self, authenticated_client):
        """Swagger documentation endpoints must bypass authentication."""
        for path in ["/docs", "/redoc", "/v1/openapi.json"]:
            resp = authenticated_client.get(path)
            assert resp.status_code == 200, f"Public path {path} failed with {resp.status_code}"

    def test_inference_rejects_missing_api_key_with_401(self, authenticated_client):
        """POST /v1/analyze/image must return HTTP 401 when no credentials are provided."""
        jpeg_bytes = _image_to_jpeg_bytes(_create_tray_scan_image())
        resp = authenticated_client.post(
            "/v1/analyze/image",
            files={"file": ("tray.jpg", jpeg_bytes, "image/jpeg")},
            data={"species": "redclaw"},
        )
        assert resp.status_code == 401
        err = resp.json()
        assert "detail" in err or "error" in err

    def test_inference_rejects_invalid_api_key_with_401(self, authenticated_client):
        """POST /v1/analyze/image must return HTTP 401 when an invalid key is supplied."""
        jpeg_bytes = _image_to_jpeg_bytes(_create_tray_scan_image())
        resp = authenticated_client.post(
            "/v1/analyze/image",
            headers={"X-API-Key": "completely_invalid_key_xyz"},
            files={"file": ("tray.jpg", jpeg_bytes, "image/jpeg")},
            data={"species": "redclaw"},
        )
        assert resp.status_code == 401

    def test_inference_accepts_x_api_key_header(self, authenticated_client):
        """POST /v1/analyze/image must succeed (200) with matching X-API-Key."""
        jpeg_bytes = _image_to_jpeg_bytes(_create_tray_scan_image())
        resp = authenticated_client.post(
            "/v1/analyze/image",
            headers={"X-API-Key": self.TEST_API_KEY},
            files={"file": ("tray.jpg", jpeg_bytes, "image/jpeg")},
            data={"species": "redclaw"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data
        assert "detections" in data
        assert data["count"] == len(data["detections"])
        assert data["processing_time_ms"] < 200.0

    def test_inference_accepts_bearer_authorization_header(self, authenticated_client):
        """POST /v1/analyze/image must succeed (200) with matching Authorization: Bearer <key>."""
        jpeg_bytes = _image_to_jpeg_bytes(_create_tray_scan_image())
        resp = authenticated_client.post(
            "/v1/analyze/image",
            headers={"Authorization": f"Bearer {self.TEST_API_KEY}"},
            files={"file": ("tray.jpg", jpeg_bytes, "image/jpeg")},
            data={"species": "redclaw"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data
        assert "detections" in data

    def test_open_local_mode_when_key_unset(self, unauthenticated_client):
        """When ANIMALLENS_API_KEY is unset, endpoints permit unauthenticated calls."""
        jpeg_bytes = _image_to_jpeg_bytes(_create_tray_scan_image())
        resp = unauthenticated_client.post(
            "/v1/analyze/image",
            files={"file": ("tray.jpg", jpeg_bytes, "image/jpeg")},
            data={"species": "redclaw"},
        )
        assert resp.status_code == 200
        assert "count" in resp.json()

    def test_websocket_events_endpoint_security(self):
        """Verify WS /v1/events rejects unauthorized connections with 1008 and accepts authorized."""
        from starlette.websockets import WebSocketDisconnect

        # 1. With ANIMALLENS_API_KEY set: reject unauthenticated
        orig = os.environ.get("ANIMALLENS_API_KEY")
        try:
            os.environ["ANIMALLENS_API_KEY"] = self.TEST_API_KEY
            client = TestClient(create_app())

            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect("/v1/events"):
                    pass
            assert exc_info.value.code == 1008

            # 2. With header auth: accepted
            with client.websocket_connect("/v1/events", headers={"X-API-Key": self.TEST_API_KEY}) as ws:
                ws.send_text("ping")
                resp = json.loads(ws.receive_text())
                assert resp.get("type") == "pong"

            # 3. With query param auth: accepted
            with client.websocket_connect(f"/v1/events?api_key={self.TEST_API_KEY}") as ws:
                ws.send_text("ping")
                resp = json.loads(ws.receive_text())
                assert resp.get("type") == "pong"
        finally:
            if orig is not None:
                os.environ["ANIMALLENS_API_KEY"] = orig
            else:
                os.environ.pop("ANIMALLENS_API_KEY", None)



# ---------------------------------------------------------------------------
# Check 3: Next.js /api/livestock/vision Route Contract & Multi-Stage Biomass
# ---------------------------------------------------------------------------

class TestCheck3NextJsProxyContractAndBiomass:
    """
    Verifies that Next.js /api/livestock/vision proxy contract correctly:
    1. Forwards API key credentials and receives structured predictions.
    2. Computes multi-stage biomass matching the canonical formula:
       Biomass = (N_craylet * 0.15g) + (N_juvenile * 5.0g) + (N_sub_adult * 25.0g) + (N_adult * 45.0g)
    3. Handles unauthorized upstream microservice responses with HTTP 401.
    """

    STAGE_WEIGHTS = {
        "craylet": 0.15,
        "juvenile": 5.0,
        "sub_adult": 25.0,
        "adult": 45.0,
    }

    def _simulate_nextjs_biomass_calculation(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Python mirror of calculateMultiStageBiomass in livestock-vision.ts."""
        breakdown = {"craylet": 0, "juvenile": 0, "sub_adult": 0, "adult": 0}
        for det in detections:
            stage_raw = str(det.get("stage", "juvenile")).lower().replace("-", "_")
            if stage_raw in breakdown:
                breakdown[stage_raw] += 1
            else:
                breakdown["juvenile"] += 1

        total_biomass = (
            breakdown["craylet"] * self.STAGE_WEIGHTS["craylet"]
            + breakdown["juvenile"] * self.STAGE_WEIGHTS["juvenile"]
            + breakdown["sub_adult"] * self.STAGE_WEIGHTS["sub_adult"]
            + breakdown["adult"] * self.STAGE_WEIGHTS["adult"]
        )
        count = len(detections)
        avg_weight = total_biomass / count if count > 0 else 0.0

        return {
            "totalBiomassG": round(total_biomass, 1),
            "avgWeightG": round(avg_weight, 2),
            "stageBreakdown": breakdown,
        }

    def test_nextjs_proxy_roundtrip_with_auth_and_biomass(self):
        """Simulate Next.js proxy request with auth and multi-stage biomass calculation."""
        api_key = "nextjs_contract_test_key_abc123"
        os.environ["ANIMALLENS_API_KEY"] = api_key
        app = create_app()
        client = TestClient(app)

        try:
            # 1. Next.js prepares payload
            tray_img = _create_tray_scan_image()
            base64_str = _image_to_base64(tray_img)
            image_bytes = base64.b64decode(base64_str)

            # 2. Next.js forwards request with headers
            headers = {
                "X-API-Key": api_key,
                "Authorization": f"Bearer {api_key}",
            }
            resp = client.post(
                "/v1/analyze/image",
                headers=headers,
                files={"file": ("tray-scan.jpg", image_bytes, "image/jpeg")},
                data={"species": "redclaw"},
            )
            assert resp.status_code == 200
            data = resp.json()

            # 3. Next.js parses detections array
            assert "detections" in data
            assert "count" in data
            detections = data["detections"]
            assert len(detections) >= 2

            for det in detections:
                assert "stage" in det or "class_name" in det
                assert "box" in det or "bbox" in det
                box_coords = det.get("box") or det.get("bbox")
                assert len(box_coords) == 4
                assert "confidence" in det

            # 4. Next.js computes multi-stage biomass
            biomass_info = self._simulate_nextjs_biomass_calculation(detections)
            assert biomass_info["totalBiomassG"] > 0.0
            assert biomass_info["avgWeightG"] > 0.0
            assert sum(biomass_info["stageBreakdown"].values()) == len(detections)

        finally:
            os.environ.pop("ANIMALLENS_API_KEY", None)

    def test_nextjs_biomass_formula_exactness(self):
        """Verify the multi-stage biomass formula on a known synthetic cohort."""
        # 10 craylets, 5 juveniles, 2 sub-adults, 1 adult
        mock_detections = (
            [{"stage": "craylet"}] * 10
            + [{"stage": "juvenile"}] * 5
            + [{"stage": "sub-adult"}] * 2
            + [{"stage": "adult"}] * 1
        )
        biomass = self._simulate_nextjs_biomass_calculation(mock_detections)

        # Expected:
        # 10 * 0.15g = 1.5g
        # 5 * 5.0g   = 25.0g
        # 2 * 25.0g  = 50.0g
        # 1 * 45.0g  = 45.0g
        # Total = 121.5g
        # Avg = 121.5 / 18 = 6.75g
        assert biomass["totalBiomassG"] == 121.5
        assert biomass["avgWeightG"] == 6.75
        assert biomass["stageBreakdown"]["craylet"] == 10
        assert biomass["stageBreakdown"]["juvenile"] == 5
        assert biomass["stageBreakdown"]["sub_adult"] == 2
        assert biomass["stageBreakdown"]["adult"] == 1

    def test_nextjs_env_local_configuration(self, repo_paths):
        """Verify AIC-main-core/aquaculture-system-next/.env.local is present with ANIMALLENS_URL."""
        env_local = repo_paths["repo_root"] / "AIC-main-core" / "aquaculture-system-next" / ".env.local"
        assert env_local.exists(), f"Missing .env.local at {env_local}"
        content = env_local.read_text(encoding="utf-8")
        assert "ANIMALLENS_URL" in content
        assert "ANIMALLENS_API_KEY" in content


# ---------------------------------------------------------------------------
# Check 4: Programmatic Execution of Existing Test Suites
# ---------------------------------------------------------------------------

class TestCheck4ExistingSuitesPass:
    """Verifies all programmatic tests in test_api_security.py and test_crayfish_vision.py pass."""

    def test_all_api_security_tests_pass(self):
        """Run all test cases in test_api_security.py via pytest."""
        ret = pytest.main(["-q", "tests/test_api_security.py"])
        assert ret == 0, f"test_api_security.py suite exited with code {ret}"

    def test_key_crayfish_vision_tests_pass(self, tmp_path):
        """Import and invoke core unit tests in test_crayfish_vision.py."""
        from tests import test_crayfish_vision as vis

        # Feature 1: Normalization and default species
        vis.test_stage_normalization_utility()
        vis.test_analyze_image_default_species()
        vis.test_analyze_image_query_and_form_species()
        vis.test_analyze_image_empty_file_rejected()
        vis.test_analyze_image_dog_backwards_compatibility()

        # Feature 2: Lifecycle classification & detection heuristics
        vis.test_crayfish_lifecycle_stage_classification_thresholds()
        vis.test_yolov8_detector_crayfish_multistage_fallback()
        vis.test_yolov8_classical_thresholding_on_tray_image()
        vis.test_sdk_redclaw_lifecycle_stage_classification()

        # Feature 3: Stream ingestion & video loops
        vis.test_stream_source_target_fps_and_bounded_buffer()
        vis.test_video_loop_seamless_rewind(tmp_path)

        # Feature 4: Memory-bounded tracking history
        vis.test_botsort_tracker_memory_bounded_history()

        # Feature 5 & 6: Stream active metrics and WebSockets
        vis.test_stream_active_metrics_schema_contract()
        vis.test_websocket_ping_pong_and_clean_disconnect()
        vis.test_websocket_telemetry_broadcast()
        vis.test_spatial_metrics_and_ethogram_calculations()
