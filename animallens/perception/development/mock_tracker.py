"""
Mock / Development Multi-Object Tracker.
Maintains persistent track IDs and computes movement velocities.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from animallens.core.schemas import BoundingBox
from animallens.perception.base import BaseTracker, DetectionResult, TrackState


class MockTracker(BaseTracker):
    """
    Development tracker using greedy nearest-centroid matching.
    """

    def __init__(self, max_distance_threshold: float = 0.30) -> None:
        self.max_distance_threshold = max_distance_threshold
        self._tracks: Dict[int, TrackState] = {}
        self._next_id: int = 1

    def _centroid_dist(self, b1: BoundingBox, b2: BoundingBox) -> float:
        c1 = b1.center
        c2 = b2.center
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    def update(
        self,
        detections: DetectionResult,
        timestamp: float,
        frame: Optional[Any] = None,
    ) -> List[TrackState]:
        unmatched_detections = list(range(len(detections.bboxes)))
        updated_tracks: List[TrackState] = []

        # Try to match to existing tracks
        for track_id, track in list(self._tracks.items()):
            best_idx = None
            best_dist = float("inf")

            for det_idx in unmatched_detections:
                dist = self._centroid_dist(track.current_bbox, detections.bboxes[det_idx])
                if dist < self.max_distance_threshold and dist < best_dist:
                    best_dist = dist
                    best_idx = det_idx

            if best_idx is not None:
                det_box = detections.bboxes[best_idx]
                conf = detections.confidences[best_idx]
                unmatched_detections.remove(best_idx)

                # Compute velocity
                dt = max(0.001, timestamp - (track.history_bboxes[-1][0] if track.history_bboxes else timestamp - 0.033))
                velocity = best_dist / dt

                track.history_bboxes.append((timestamp, track.current_bbox))
                if len(track.history_bboxes) > 60:
                    track.history_bboxes.pop(0)

                track.current_bbox = det_box
                track.confidence = conf
                track.velocity = velocity
                track.age_frames += 1
                track.missed_frames = 0
                updated_tracks.append(track)
            else:
                track.missed_frames += 1
                if track.missed_frames > 15:
                    del self._tracks[track_id]

        # Initialize new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det_box = detections.bboxes[det_idx]
            conf = detections.confidences[det_idx]
            new_track = TrackState(
                track_id=self._next_id,
                animal_id=f"RC-{self._next_id:03d}",
                current_bbox=det_box,
                history_bboxes=[(timestamp, det_box)],
                velocity=0.0,
                confidence=conf,
                age_frames=1,
            )
            self._tracks[self._next_id] = new_track
            self._next_id += 1
            updated_tracks.append(new_track)

        return updated_tracks

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
