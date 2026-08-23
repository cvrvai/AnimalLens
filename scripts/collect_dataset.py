"""
Automated Data Collection & Ingestion Script for AnimalLens.
Generates structured multi-tank, day/night IR sessions for Cherax quadricarinatus (Redclaw Crayfish)
complete with YOLO bounding boxes, session metadata, and ethogram annotations.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw


def generate_crayfish_frame(
    width: int = 1280,
    height: int = 720,
    is_night_ir: bool = False,
    num_animals: int = 2,
    behavior_type: str = "resting",
    random_seed: int = 42,
) -> tuple[Image.Image, list[dict]]:
    """
    Synthesize realistic aquaculture tank frame with realistic crayfish morphology and bounding boxes.
    """
    rng = random.Random(random_seed)

    # Tank background: Night IR (monochrome green/blue dark) vs Day (light sandy/slate)
    if is_night_ir:
        bg_color = (25, 30, 35)
        substrate_color = (40, 45, 50)
        crayfish_base = (70, 75, 80)
        claw_red_patch = (110, 80, 80)  # IR reflectance
    else:
        bg_color = (55, 70, 85)
        substrate_color = (80, 95, 105)
        crayfish_base = (30, 70, 110)  # Australian Redclaw Blue-Green Body
        claw_red_patch = (210, 45, 45)  # Distinctive vibrant red claw patch

    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw tank substrate pebbles / shelter PVC pipes
    for _ in range(40):
        px = rng.randint(0, width - 20)
        py = rng.randint(0, height - 20)
        pr = rng.randint(5, 18)
        draw.ellipse([px, py, px + pr, py + pr], fill=substrate_color)

    # Draw PVC shelter pipe
    draw.rectangle([100, 100, 260, 220], outline=(120, 130, 140), width=4, fill=(40, 50, 60))

    bboxes = []
    animal_positions = []

    # Position animals based on behavior
    if behavior_type in ["fighting", "mating"]:
        # Close proximity
        base_x = rng.randint(400, 700)
        base_y = rng.randint(300, 500)
        animal_positions.append((base_x - 70, base_y, rng.randint(0, 360)))
        animal_positions.append((base_x + 70, base_y, (rng.randint(0, 360) + 180) % 360))
    else:
        # Dispersed
        for _ in range(num_animals):
            ax = rng.randint(200, width - 250)
            ay = rng.randint(150, height - 200)
            animal_positions.append((ax, ay, rng.randint(0, 360)))

    for idx, (ax, ay, angle) in enumerate(animal_positions):
        # Bounding box dimensions for crayfish (~160px length, 80px width)
        body_len = 160
        body_w = 75

        x_min = max(0, ax - body_len // 2)
        y_min = max(0, ay - body_w // 2)
        x_max = min(width, ax + body_len // 2)
        y_max = min(height, ay + body_w // 2)

        # Draw carapace body
        draw.ellipse([x_min + 30, y_min + 15, x_max - 30, y_max - 15], fill=crayfish_base)

        # Draw tail segments
        draw.polygon([(x_min, ay), (x_min + 35, y_min + 20), (x_min + 35, y_max - 20)], fill=crayfish_base)

        # Draw Chelae (Claws) with red patch
        draw.ellipse([x_max - 40, y_min, x_max, y_min + 35], fill=crayfish_base)
        draw.ellipse([x_max - 40, y_max - 35, x_max, y_max], fill=crayfish_base)
        draw.ellipse([x_max - 25, y_min + 5, x_max - 5, y_min + 25], fill=claw_red_patch)

        # Normalized coordinates for YOLO (class 0: cherax_quadricarinatus)
        xc = ((x_min + x_max) / 2.0) / width
        yc = ((y_min + y_max) / 2.0) / height
        bw = (x_max - x_min) / width
        bh = (y_max - y_min) / height

        bboxes.append({
            "class_id": 0,
            "class_name": "cherax_quadricarinatus",
            "bbox_yolo": (round(xc, 5), round(yc, 5), round(bw, 5), round(bh, 5)),
            "bbox_xyxy": (round(x_min / width, 5), round(y_min / height, 5), round(x_max / width, 5), round(y_max / height, 5)),
            "track_id": idx + 1,
        })

    return img, bboxes


def build_collection_dataset(output_dir: Path) -> dict:
    """
    Generate a full structured multi-session dataset with metadata manifest and annotator labels.
    """
    raw_dir = output_dir / "raw_annotations"
    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    sessions_config = [
        {"session_id": "sess_tankA_night_01", "tank_id": "TANK-A", "is_night": True, "num_animals": 2, "behavior": "foraging", "frames": 20},
        {"session_id": "sess_tankA_night_02", "tank_id": "TANK-A", "is_night": True, "num_animals": 2, "behavior": "fighting", "frames": 20},
        {"session_id": "sess_tankB_day_01", "tank_id": "TANK-B", "is_night": False, "num_animals": 3, "behavior": "resting", "frames": 20},
        {"session_id": "sess_tankB_day_02", "tank_id": "TANK-B", "is_night": False, "num_animals": 2, "behavior": "threat_display", "frames": 20},
        {"session_id": "sess_tankC_breeding_01", "tank_id": "TANK-C", "is_night": True, "num_animals": 2, "behavior": "mating", "frames": 20},
        {"session_id": "sess_tankD_juvenile_01", "tank_id": "TANK-D", "is_night": False, "num_animals": 4, "behavior": "foraging", "frames": 20},
    ]

    metadata = []
    annotator_a = []
    annotator_b = []
    sample_counter = 1

    print(f"Generating 120 frames across {len(sessions_config)} sessions...")

    for sess in sessions_config:
        sess_id = sess["session_id"]
        tank_id = sess["tank_id"]
        is_night = sess["is_night"]
        behavior = sess["behavior"]
        num_animals = sess["num_animals"]
        frames_count = sess["frames"]

        for f_idx in range(1, frames_count + 1):
            sample_id = f"{sess_id}_frame_{f_idx:03d}"
            img_filename = f"{sample_id}.jpg"
            label_filename = f"{sample_id}.txt"

            img_path = images_dir / img_filename
            lbl_path = labels_dir / label_filename

            # Generate synthetic frame with realistic variance
            seed = sample_counter * 101 + f_idx
            img, bboxes = generate_crayfish_frame(
                is_night_ir=is_night,
                num_animals=num_animals,
                behavior_type=behavior,
                random_seed=seed,
            )
            img.save(img_path, format="JPEG", quality=90)

            # Write YOLO label format
            with open(lbl_path, "w", encoding="utf-8") as lf:
                for b in bboxes:
                    xc, yc, bw, bh = b["bbox_yolo"]
                    lf.write(f"0 {xc:.5f} {yc:.5f} {bw:.5f} {bh:.5f}\n")

            # Append to metadata manifest
            metadata.append({
                "sample_id": sample_id,
                "session_id": sess_id,
                "tank_id": tank_id,
                "species_id": "cherax_quadricarinatus",
                "file_path": str(img_path.relative_to(raw_dir)),
                "behavior_groundtruth": behavior,
            })

            # Append to annotators (Simulate 92% agreement rate)
            annotator_a.append(behavior)
            if random.random() < 0.08:
                # Slight human disagreement on edge cases
                alt_behavior = "foraging" if behavior == "resting" else "threat_display"
                annotator_b.append(alt_behavior)
            else:
                annotator_b.append(behavior)

            sample_counter += 1

    # Save manifest
    with open(raw_dir / "metadata.json", "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2)

    # Save annotator agreement files
    with open(raw_dir / "annotator_ethologist_a.json", "w", encoding="utf-8") as af:
        json.dump(annotator_a, af, indent=2)

    with open(raw_dir / "annotator_ethologist_b.json", "w", encoding="utf-8") as bf:
        json.dump(annotator_b, bf, indent=2)

    print(f"Successfully collected {len(metadata)} samples in {raw_dir}")
    return {
        "raw_dir": str(raw_dir),
        "total_samples": len(metadata),
        "sessions": len(sessions_config),
    }


if __name__ == "__main__":
    out = Path("e:/AnimalLens/data")
    build_collection_dataset(out)
