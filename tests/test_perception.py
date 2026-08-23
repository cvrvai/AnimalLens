"""
Unit tests for perception models, tracking, rolling buffer, and pipeline.
"""
from PIL import Image
import pytest
from animallens.core.schemas import SourceInfo, SourceType
from animallens.perception.base import FramePerceptionData
from animallens.perception.buffer import RollingVideoBuffer
from animallens.perception.development.mock_detector import MockDetector
from animallens.perception.development.mock_tracker import MockTracker
from animallens.perception.development.redclaw_rules import RuleBasedRedclawClassifier
from animallens.perception.pipeline import PerceptionPipeline
from animallens.species.redclaw.adapter import RedclawAdapter


def test_mock_detector():
    detector = MockDetector(num_subjects=2)
    dummy_img = Image.new("RGB", (640, 480))
    res = detector.detect(dummy_img)

    assert len(res.bboxes) == 2
    assert len(res.confidences) == 2
    assert res.confidences[0] > 0.5


def test_mock_tracker():
    detector = MockDetector(num_subjects=2)
    tracker = MockTracker()
    dummy_img = Image.new("RGB", (640, 480))

    # Frame 1
    dets1 = detector.detect(dummy_img)
    tracks1 = tracker.update(dets1, timestamp=0.0)
    assert len(tracks1) == 2
    id1 = tracks1[0].track_id

    # Frame 2
    dets2 = detector.detect(dummy_img)
    tracks2 = tracker.update(dets2, timestamp=0.1)
    assert len(tracks2) == 2
    # Persistent track ID preserved
    assert tracks2[0].track_id == id1


def test_rolling_video_buffer():
    buf = RollingVideoBuffer(capacity_seconds=5.0, fps=10.0)
    detector = MockDetector(num_subjects=1)
    dummy_img = Image.new("RGB", (100, 100))

    for i in range(20):
        ts = i * 0.1
        dets = detector.detect(dummy_img)
        fdata = FramePerceptionData(frame_index=i, timestamp=ts, detections=dets)
        buf.push(timestamp=ts, frame=dummy_img, perception_data=fdata)

    assert len(buf) == 20
    window = buf.get_window(duration_seconds=1.0)
    assert len(window) > 0
    assert window[-1].timestamp >= 1.9


def test_perception_pipeline():
    adapter = RedclawAdapter()
    pipeline = PerceptionPipeline(species_adapter=adapter)
    dummy_img = Image.new("RGB", (640, 480))

    events = pipeline.process_frame(dummy_img, timestamp=0.0)
    assert isinstance(events, list)
    assert len(events) >= 1
    assert events[0].species.id == "cherax_quadricarinatus"
