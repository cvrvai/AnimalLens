"""
Render annotated detection video and keyframe snapshots for Dog (Canis lupus familiaris).
"""
from pathlib import Path
import cv2
from PIL import Image
from animallens.perception.models.botsort_tracker import BoTSORTTracker
from animallens.perception.models.yolov8_detector import YOLOv8Detector


def render_dog_video(
    video_path: str = "data/dog_sample.mp4",
    output_video: str = "data/annotated_dog_output.mp4",
) -> list[str]:
    detector = YOLOv8Detector(conf_threshold=0.35)
    tracker = BoTSORTTracker()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    out_file = Path(output_video)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (w, h))

    saved_images = []
    frame_idx = 0

    box_color = (255, 140, 0)      # Deep Orange/Gold for Dogs
    banner_color = (25, 25, 25)

    print(f"Rendering canine detection video on {video_path}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # 1. Run YOLOv8 detection
        detections = detector.detect(pil_img, confidence_threshold=0.35)

        # 2. Run BoT-SORT tracker
        tracks = tracker.update(detections, timestamp=timestamp, frame=frame_rgb)

        annotated = frame.copy()
        # Top HUD Banner
        cv2.rectangle(annotated, (0, 0), (w, 75), banner_color, -1)
        cv2.putText(
            annotated,
            f"AnimalLens Vision AI | Species: Canis lupus familiaris (Dog) | Active Tracks: {len(tracks)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Time: {timestamp:.2f}s | Model: YOLOv8 COCO Pretrained | Tracker: BoT-SORT Kalman",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Bounding boxes
        for trk in tracks:
            bx1 = int(trk.current_bbox.x_min * w)
            by1 = int(trk.current_bbox.y_min * h)
            bx2 = int(trk.current_bbox.x_max * w)
            by2 = int(trk.current_bbox.y_max * h)

            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), box_color, 3)

            tag = f"ID #{trk.track_id}: Dog ({trk.confidence * 100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(annotated, (bx1, by1 - th - 10), (bx1 + tw + 12, by1), box_color, -1)
            cv2.putText(
                annotated,
                tag,
                (bx1 + 6, by1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        writer.write(annotated)

        # Save snapshot
        if frame_idx in (10, 60):
            snap_path = Path("data") / f"annotated_dog_frame_{frame_idx}.jpg"
            cv2.imwrite(str(snap_path), annotated)
            saved_images.append(str(snap_path))
            print(f"Saved dog detection snapshot: {snap_path.resolve()}")

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Rendered annotated dog video saved to: {out_file.resolve()}")
    return saved_images


if __name__ == "__main__":
    render_dog_video()
