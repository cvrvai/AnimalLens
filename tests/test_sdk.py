"""
Unit tests for high-level AnimalLens developer SDK.
"""
from PIL import Image
import pytest
from animallens import AnimalLens
from animallens.core.schemas import AnalysisResult, BehaviorEvent


def test_sdk_analyze_image():
    lens = AnimalLens(species="redclaw", reasoning=None)
    img = Image.new("RGB", (640, 480), color=(10, 20, 30))

    result = lens.analyze_image(img)
    assert isinstance(result, AnalysisResult)
    assert result.species == "Redclaw Crayfish"
    assert len(result.behaviors) > 0
    assert result.total_frames_analyzed == 1


def test_sdk_analyze_video():
    lens = AnimalLens(species="redclaw", reasoning=None)
    result = lens.analyze_video("dummy_tank.mp4", sample_fps=5.0, max_duration_seconds=4.0)

    assert isinstance(result, AnalysisResult)
    assert result.total_frames_analyzed == 20
    assert len(result.timeline) > 0
    assert "===" not in result.format_timeline_text()  # Formatted text exists


def test_sdk_stream():
    lens = AnimalLens(species="redclaw", reasoning=None)
    events = []

    # Stream 5 events
    for event in lens.stream("rtsp://mock-cam"):
        events.append(event)
        if len(events) >= 5:
            break

    assert len(events) == 5
    assert isinstance(events[0], BehaviorEvent)
    assert events[0].species.id == "cherax_quadricarinatus"
