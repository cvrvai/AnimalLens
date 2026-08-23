"""
Create a sample aquaculture tank video from generated session frames.
"""
import glob
from pathlib import Path
import cv2


def make_video(output_path: str = "data/my_tank_video.mp4") -> str:
    img_files = sorted(glob.glob("data/processed_yolo/images/test/*.jpg"))
    if not img_files:
        img_files = sorted(glob.glob("data/raw_annotations/images/*.jpg"))

    if not img_files:
        raise RuntimeError("No image files found.")

    first = cv2.imread(img_files[0])
    h, w = first.shape[:2]

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_file), fourcc, 10.0, (w, h))

    # Repeat sequence 3 times for a 6-second video
    for _ in range(3):
        for img_path in img_files:
            frame = cv2.imread(img_path)
            if frame is not None:
                out.write(frame)

    out.release()
    print(f"Sample tank video created at: {out_file.resolve()}")
    return str(out_file)


if __name__ == "__main__":
    make_video()
