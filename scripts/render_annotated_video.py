"""
Render annotated detection video and keyframe images with bounding boxes,
track IDs, confidence tags, and ethogram behavior banners.
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from animallens.perception.models.botsort_tracker import BoTSORTTracker
from animallens.perception.models.yolov8_detector import YOLOv8Detector
from animallens.perception.temporal.classifier import TemporalBehaviorClassifier
from animallens.species.registry import species_registry


def render_detections(
    video_path: str = r"E:\AnimalLens\data\Red Claw Crayfish Great Alternative to Lobster and Crab - YouTube - Google Chrome 2026-08-23 12-12-10.mp4",
    output_video: str = "data/annotated_detection_output.mp4",
    weights_path: str = "data/trained_models/redclaw-behavior-v1.pt",
    max_frames: int = 150,
) -> list[str]:
    detector = YOLOv8Detector(model_path=weights_path, conf_threshold=0.30)
    tracker = BoTSORTTracker()
    classifier = TemporalBehaviorClassifier()
    species = species_registry.get("redclaw")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    out_file = Path(output_video)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps / 2, (w, h))

    saved_images = []
    frame_idx = 0
    stride = 2  # Sample every 2nd frame

    print(f"Rendering detections on {video_path} (Resolution: {w}x{h})...")

    # Colors
    box_color = (0, 220, 50)       # Vibrant Green for crayfish
    banner_color = (30, 30, 30)     # Dark Gray for top HUD banner
    text_color = (255, 255, 255)

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        if frame_idx % stride == 0:
            timestamp = frame_idx / fps
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # 1. Run YOLOv8 detection
            detections = detector.detect(pil_img, confidence_threshold=0.30)

            # 2. Run BoT-SORT multi-object tracking
            tracks = tracker.update(detections, timestamp=timestamp, frame=frame_rgb)

            # 3. Draw Top HUD Banner
            annotated = frame.copy()
            cv2.rectangle(annotated, (0, 0), (w, 75), banner_color, -1)
            cv2.putText(
                annotated,
                f"AnimalLens Vision AI | Species: Cherax quadricarinatus | Tracks: {len(tracks)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 230, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Time: {timestamp:.2f}s | Detector: YOLOv8-Custom (conf >= 30%) | Tracker: BoT-SORT",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # 4. Draw bounding boxes and track badges
            for trk in tracks:
                bx1 = int(trk.current_bbox.x_min * w)
                by1 = int(trk.current_bbox.y_min * h)
                bx2 = int(trk.current_bbox.x_max * w)
                by2 = int(trk.current_bbox.y_max * h)

                # Draw corner brackets / bounding box
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), box_color, 2)

                # Label tag
                label_text = f"ID #{trk.track_id}: Redclaw ({trk.confidence * 100:.0f}%)"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (bx1, by1 - th - 8), (bx1 + tw + 10, by1), box_color, -1)
                cv2.putText(
                    annotated,
                    label_text,
                    (bx1 + 5, by1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            # Write frame to video
            writer.write(annotated)

            # Save keyframe snapshots
            if frame_idx in (20, 60, 100, 140):
                snap_path = Path("data") / f"annotated_detection_frame_{frame_idx}.jpg"
                cv2.imwrite(str(snap_path), annotated)
                saved_images.append(str(snap_path))
                print(f"Saved detection snapshot: {snap_path.resolve()}")

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"\nRendered detection video saved to: {out_file.resolve()}")
    return saved_images


if __name__ == "__main__":
    render_detections()
