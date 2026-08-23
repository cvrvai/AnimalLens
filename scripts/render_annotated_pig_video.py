"""
Renders an annotated output video with bounding boxes, confidence tags, and ethological HUD
using our custom trained pig model (best.pt). Also exports keyframe artifacts for visual inspection.
"""
from pathlib import Path
import cv2
import numpy as np

from animallens.core.schemas import BoundingBox
from animallens.perception.models.yolov8_detector import YOLOv8Detector
from animallens.species.pig.adapter import PigAdapter


def main():
    print("=" * 70)
    print("  Rendering Annotated Pig Video & Keyframe Artifacts")
    print("=" * 70)

    weights_path = Path("models/trained/pig_behavior_v1/weights/best.pt")
    video_path = Path("data/raw/videos/pig_farm_pen.mp4")
    out_dir = Path("data/annotated")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = out_dir / "pig_annotated.mp4"

    artifact_dir = Path(r"C:\Users\EDEN\.gemini\antigravity\brain\4d7fdf63-274a-474f-9316-dcd505b80856")

    detector = YOLOv8Detector(model_path=weights_path, classes=["pig"], conf_threshold=0.30)
    adapter = PigAdapter()

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Input Video: {video_path} ({width}x{height}, {fps:.1f} FPS, {total_frames} frames)")
    print(f"📦 Model: {weights_path}")
    print(f"🎬 Output Video: {out_video_path}")

    # Use mp4v or XVID codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    frame_idx = 0
    exported_artifacts = []
    sample_artifact_frames = [30, 150, 600, 1200, 1800]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        res = detector.detect(frame, confidence_threshold=0.30)
        time_sec = frame_idx / fps

        # Draw HUD banner at top
        hud_bg = frame.copy()
        cv2.rectangle(hud_bg, (0, 0), (width, 55), (20, 20, 20), -1)
        cv2.addWeighted(hud_bg, 0.75, frame, 0.25, 0, frame)

        # Header info
        cv2.putText(frame, f"AnimalLens Swine Vision AI | Time: {time_sec:05.1f}s | FPS: {fps:.0f}", (15, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        if res.bboxes:
            for i, box in enumerate(res.bboxes):
                conf = res.confidences[i]
                posture = adapter.classify_recumbency(box, velocity=0.0)

                # Bounding box pixel coords
                x1 = int(box.x_min * width)
                y1 = int(box.y_min * height)
                x2 = int(box.x_max * width)
                y2 = int(box.y_max * height)

                # Draw main box
                color = (0, 220, 100)  # emerald green
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Corner brackets for tech HUD styling
                d = 20
                cv2.line(frame, (x1, y1), (x1 + d, y1), (0, 255, 255), 4)
                cv2.line(frame, (x1, y1), (x1, y1 + d), (0, 255, 255), 4)
                cv2.line(frame, (x2, y1), (x2 - d, y1), (0, 255, 255), 4)
                cv2.line(frame, (x2, y1), (x2, y1 + d), (0, 255, 255), 4)
                cv2.line(frame, (x1, y2), (x1 + d, y2), (0, 255, 255), 4)
                cv2.line(frame, (x1, y2), (x1, y2 - d), (0, 255, 255), 4)
                cv2.line(frame, (x2, y2), (x2 - d, y2), (0, 255, 255), 4)
                cv2.line(frame, (x2, y2), (x2 - d, y2), (0, 255, 255), 4)

                # Clean swine label tag
                label_text = f"Pig #{i+1} (Sow): {posture.upper()} ({conf:.0%})"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - 26)), (x1 + tw + 10, y1), (0, 220, 100), -1)
                cv2.putText(frame, label_text, (x1 + 5, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

            # Top HUD details
            primary_posture = adapter.classify_recumbency(res.bboxes[0], velocity=0.0)
            behavior_desc = "NESTING / SUBSTRATE ROOTING" if primary_posture == "rooting_nesting" else primary_posture.upper()
            cv2.putText(frame, f"Behavior: {behavior_desc} | Species: Sus scrofa (Pig) | Count: {len(res.bboxes)}", (15, 47),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA)

        out_writer.write(frame)

        # Save keyframe snapshots for artifact presentation
        if frame_idx in sample_artifact_frames:
            snapshot_name = f"pig_detection_frame_{frame_idx}.jpg"
            snapshot_path = artifact_dir / snapshot_name
            cv2.imwrite(str(snapshot_path), frame)
            exported_artifacts.append(snapshot_path)
            print(f"   📸 Saved snapshot artifact: {snapshot_name} (frame {frame_idx}, {time_sec:.1f}s)")

        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"   Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames:.0%})...")

    cap.release()
    out_writer.release()

    file_size_mb = out_video_path.stat().st_size / 1024 / 1024
    print("\n" + "=" * 70)
    print(f"✅ Video Annotation Finished!")
    print(f"   Annotated Video File: {out_video_path} ({file_size_mb:.1f} MB)")
    print(f"   Saved {len(exported_artifacts)} Keyframe Image Artifacts")
    print("=" * 70)


if __name__ == "__main__":
    main()
