"""
Render annotated dynamic detection video for moving Greyhound dog with trajectory motion trails.
"""
from collections import defaultdict, deque
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from animallens.perception.models.botsort_tracker import BoTSORTTracker
from animallens.perception.models.yolov8_detector import YOLOv8Detector


def render_running_dog(
    video_path: str = "data/raw/videos/dog_running.mp4",
    output_video: str = "data/outputs/videos/annotated_greyhound_running.mp4",
) -> list[str]:
    detector = YOLOv8Detector(conf_threshold=0.30)
    tracker = BoTSORTTracker()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 270)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    # Upscale output for crisp display
    out_w, out_h = w * 2, h * 2

    out_file = Path(output_video)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (out_w, out_h))

    saved_images = []
    frame_idx = 0

    # Store trajectory history per track ID: {track_id: deque([(cx, cy), ...])}
    track_histories = defaultdict(lambda: deque(maxlen=30))

    box_color = (0, 200, 255)  # Bright Amber / Gold
    trail_color = (0, 255, 120)  # Bright Neon Green trajectory trail
    banner_color = (20, 20, 20)

    print(f"Rendering running Greyhound motion tracking on {video_path}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # 1. Detect
        detections = detector.detect(pil_img, confidence_threshold=0.30)

        # 2. Track
        tracks = tracker.update(detections, timestamp=timestamp, frame=frame_rgb)

        # Resize for visualization
        annotated = cv2.resize(frame, (out_w, out_h))

        # Top HUD Banner
        cv2.rectangle(annotated, (0, 0), (out_w, 65), banner_color, -1)
        cv2.putText(
            annotated,
            f"AnimalLens Canine AI | Greyhound Gallop | Active Tracks: {len(tracks)}",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Time: {timestamp:.2f}s | Locomotion Tracking | BoT-SORT Kalman",
            (15, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # 3. Draw tracks & motion trails
        for trk in tracks:
            bx1 = int(trk.current_bbox.x_min * out_w)
            by1 = int(trk.current_bbox.y_min * out_h)
            bx2 = int(trk.current_bbox.x_max * out_w)
            by2 = int(trk.current_bbox.y_max * out_h)

            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2
            track_histories[trk.track_id].append((cx, cy))

            # Draw trajectory path
            pts = list(track_histories[trk.track_id])
            for i in range(1, len(pts)):
                thickness = int(np.sqrt(30 / float(len(pts) - i + 1)) * 1.5)
                cv2.line(annotated, pts[i - 1], pts[i], trail_color, thickness)

            # Draw bounding box
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), box_color, 2)

            tag = f"ID #{trk.track_id}: Running Dog ({trk.confidence * 100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (bx1, by1 - th - 8), (bx1 + tw + 8, by1), box_color, -1)
            cv2.putText(
                annotated,
                tag,
                (bx1 + 4, by1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        writer.write(annotated)

        # Save snapshots at key movement frames
        if frame_idx in (50, 120, 200, 280):
            snap_dir = Path("data/outputs/snapshots")
            snap_dir.mkdir(parents=True, exist_ok=True)
            snap_path = snap_dir / f"annotated_greyhound_frame_{frame_idx}.jpg"
            cv2.imwrite(str(snap_path), annotated)
            saved_images.append(str(snap_path))
            print(f"Saved running dog snapshot: {snap_path.resolve()}")

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Rendered Greyhound running video saved to: {out_file.resolve()}")
    return saved_images


if __name__ == "__main__":
    render_running_dog()
