"""
BoT-SORT tracker implementation with Kalman filter and Hungarian/greedy IoU association.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from animallens.core.schemas import BoundingBox
from animallens.perception.base import BaseTracker, DetectionResult, TrackState
from animallens.perception.models.kalman_filter import KalmanBoxTracker


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Intersection over Union (IoU) matrix between two sets of boxes.
    Boxes format: [x1, y1, x2, y2].
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)

    boxes1 = np.asarray(boxes1, dtype=np.float64)
    boxes2 = np.asarray(boxes2, dtype=np.float64)

    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = np.maximum(b1_x1[:, None], b2_x1[None, :])
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1[None, :])
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2[None, :])
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union_area = b1_area[:, None] + b2_area[None, :] - inter_area
    union_area = np.maximum(union_area, 1e-8)

    return inter_area / union_area


class TrackItem:
    """Internal track container holding KalmanBoxTracker state."""

    def __init__(self, bbox: BoundingBox, confidence: float, class_name: str) -> None:
        self.tracker = KalmanBoxTracker([bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max])
        self.track_id = self.tracker.id
        self.confidence = confidence
        self.class_name = class_name
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.history: List[tuple[float, BoundingBox]] = []
        self.attributes: Dict[str, Any] = {}

    def predict(self) -> List[float]:
        self.age += 1
        return self.tracker.predict()

    def update(self, bbox: BoundingBox, confidence: float, timestamp: float) -> None:
        self.tracker.update([bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max])
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.history.append((timestamp, bbox))
        if len(self.history) > 50:
            self.history = self.history[-50:]

    @property
    def current_bbox(self) -> BoundingBox:
        s = self.tracker.get_state()
        return BoundingBox(
            x_min=float(s[0]),
            y_min=float(s[1]),
            x_max=float(s[2]),
            y_max=float(s[3]),
            is_normalized=True,
        )

    @property
    def velocity(self) -> float:
        return self.tracker.get_velocity()


class BoTSORTTracker(BaseTracker):
    """
    Robust BoT-SORT Multi-Object Tracker.
    Associates detections across consecutive frames using Kalman motion prediction and IoU matching.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 1,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[TrackItem] = []
        self.frame_count = 0

    def update(
        self,
        detections: DetectionResult,
        timestamp: float = 0.0,
        frame: Optional[Any] = None,
    ) -> List[TrackState]:
        self.frame_count += 1

        # 1. Predict new positions of existing tracks
        for t in self.tracks:
            t.predict()

        det_boxes = []
        for b in detections.bboxes:
            det_boxes.append([b.x_min, b.y_min, b.x_max, b.y_max])

        matched_tracks = set()
        matched_dets = set()

        if self.tracks and det_boxes:
            track_boxes = np.array([t.tracker.get_state() for t in self.tracks])
            iou_mat = compute_iou_matrix(track_boxes, np.array(det_boxes))

            # Greedy matching on highest IoU
            while True:
                if iou_mat.size == 0:
                    break
                max_val = np.max(iou_mat)
                if max_val < self.iou_threshold:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if t_idx in matched_tracks or d_idx in matched_dets:
                    iou_mat[t_idx, d_idx] = -1.0
                    continue

                matched_tracks.add(int(t_idx))
                matched_dets.add(int(d_idx))

                self.tracks[t_idx].update(
                    detections.bboxes[d_idx],
                    confidence=detections.confidences[d_idx] if d_idx < len(detections.confidences) else 0.9,
                    timestamp=timestamp,
                )
                iou_mat[t_idx, :] = -1.0
                iou_mat[:, d_idx] = -1.0

        # Unmatched detections become new tracks
        for d_idx, bbox in enumerate(detections.bboxes):
            if d_idx not in matched_dets:
                conf = detections.confidences[d_idx] if d_idx < len(detections.confidences) else 0.9
                cls_name = detections.class_names[d_idx] if d_idx < len(detections.class_names) else "animal"
                new_track = TrackItem(bbox=bbox, confidence=conf, class_name=cls_name)
                new_track.history.append((timestamp, bbox))
                self.tracks.append(new_track)

        # Increment time_since_update for unmatched tracks and prune old ones
        active_tracks: List[TrackItem] = []
        for idx, t in enumerate(self.tracks):
            if idx not in matched_tracks and idx < len(self.tracks) - len(detections.bboxes) + len(matched_dets):
                t.time_since_update += 1
            if t.time_since_update <= self.max_age:
                active_tracks.append(t)
        self.tracks = active_tracks

        # Return confirmed tracks as TrackState
        results: List[TrackState] = []
        for t in self.tracks:
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                results.append(
                    TrackState(
                        track_id=t.track_id,
                        animal_id=f"animal_{t.track_id}",
                        current_bbox=t.current_bbox,
                        history_bboxes=t.history[-10:],
                        velocity=t.velocity,
                        confidence=t.confidence,
                        attributes={"class_name": t.class_name, **t.attributes},
                    )
                )

        return results

    @property
    def active_tracks(self) -> List[TrackItem]:
        """Returns confirmed active tracks."""
        return [t for t in self.tracks if t.hits >= self.min_hits or self.frame_count <= self.min_hits]

    def reset(self) -> None:
        """Reset internal tracking state."""
        self.tracks = []
        self.frame_count = 0
