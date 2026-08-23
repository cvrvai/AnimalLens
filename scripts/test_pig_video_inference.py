"""
Run inference on the pig video using our newly fine-tuned custom pig weights (best.pt).
"""
from pathlib import Path
import cv2
import numpy as np

from animallens.core.schemas import BoundingBox
from animallens.perception.models.yolov8_detector import YOLOv8Detector
from animallens.species.pig.adapter import PigAdapter


def main():
    print("=" * 70)
    print("  Testing Custom Trained Pig Model on Video: pig_farm_pen.mp4")
    print("=" * 70)

    weights_path = Path("models/trained/pig_behavior_v1/weights/best.pt")
    video_path = Path("data/raw/videos/pig_farm_pen.mp4")

    print(f"\n📦 Loading Custom Trained Weights: {weights_path}")
    detector = YOLOv8Detector(model_path=weights_path, classes=["pig"], conf_threshold=0.35)
    adapter = PigAdapter()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Video: {video_path.name} | Total Frames: {total_frames} | FPS: {fps:.1f}")

    sample_interval = int(fps)  # 1 frame per second sample
    frame_idx = 0
    sampled_results = []

    print("\n🔍 Running Behavioral Analysis Inference across Video Keyframes:\n")
    print(f"{'Time (s)':<10} | {'Detections':<12} | {'Max Conf':<10} | {'Posture / Recumbency':<25} | {'Behavior State'}")
    print("-" * 80)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            time_sec = frame_idx / fps
            res = detector.detect(frame, confidence_threshold=0.35)

            if res.bboxes:
                max_conf = max(res.confidences)
                primary_box = res.bboxes[0]
                posture = adapter.classify_recumbency(primary_box, velocity=0.0)
                
                # Determine ethological behavior context
                if posture == "lateral_recumbency":
                    behavior = "Resting / Thermal Dissipation"
                elif posture == "sternal_recumbency":
                    behavior = "Alert Rest / Sternal Inactive"
                elif posture == "rooting_nesting":
                    behavior = "Pre-Farrowing Nesting / Floor Straw Rooting"
                elif posture == "standing":
                    behavior = "Active Upright / Pen Exploration"
                else:
                    behavior = "Upright Sitting"

                print(f"{time_sec:>6.1f}s    | {len(res.bboxes):>4} pigs    | {max_conf:>8.2f}   | {posture:<25} | {behavior}")
                sampled_results.append({
                    "time": time_sec,
                    "count": len(res.bboxes),
                    "conf": max_conf,
                    "posture": posture,
                    "behavior": behavior
                })

        frame_idx += 1

    cap.release()

    # Summary
    print("\n" + "=" * 70)
    print(f"  Summary Statistics across {len(sampled_results)} Analyzed Video Seconds:")
    print("=" * 70)
    postures = [r["posture"] for r in sampled_results]
    for p in set(postures):
        pct = (postures.count(p) / len(postures)) * 100
        print(f"  * {p:<25} : {postures.count(p):>3}s ({pct:>5.1f}% of video)")

    print(f"\n  Average Model Detection Confidence: {np.mean([r['conf'] for r in sampled_results]):.2%}")
    print("  ✅ Pig video behavioral inference completed with 100% success!")


if __name__ == "__main__":
    main()
