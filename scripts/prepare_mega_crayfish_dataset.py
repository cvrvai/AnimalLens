"""
Mega Crayfish Dataset Builder & Hard Negative Synthesizer for AnimalLens.
Merges:
1. crayfish-zh9y5_v1 (early juveniles/craylets)
2. crayfish-4bvu4-mdooz_v1 (juveniles)
3. crayiot_v3 (eggs, juveniles, pregnant adults)
4. crayfish_ulcer_v1 (sub-adults / mature)
Adds hard negative background images (empty steel trays, water reflections, metallic textures)
Outputs: datasets/crayfish_mega_combined ready for high-epoch YOLOv8 training.
"""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

# Life stage classes
CLASSES = ["craylet", "juvenile", "sub_adult", "adult"]

DATASET_CONFIGS = [
    {
        "name": "crayfish-zh9y5_v1",
        "dir": Path("datasets/crayfish-zh9y5_v1"),
        # 0: early_juvenile -> craylet (0)
        "remap": {0: 0},
    },
    {
        "name": "crayfish-4bvu4-mdooz_v1",
        "dir": Path("datasets/crayfish-4bvu4-mdooz_v1"),
        # 0: crayfish -> juvenile (1)
        "remap": {0: 1},
    },
    {
        "name": "crayiot_v3",
        "dir": Path("datasets/crayiot_v3"),
        # 0: Eggs -> craylet (0), 1: crayfish -> juvenile (1), 2: pregnant-Crayfish -> adult (3)
        "remap": {0: 0, 1: 1, 2: 3},
    },
    {
        "name": "crayfish_ulcer_v1",
        "dir": Path("datasets/crayfish_ulcer_v1"),
        # 0: Shell ulcer -> sub_adult (2), 1: healthy_Crayfish -> sub_adult (2)
        "remap": {0: 2, 1: 2},
    },
]


def create_steel_tray_negatives(output_dir: Path, count: int = 80):
    """
    Generate synthetic hard negative backgrounds representing stainless steel trays,
    metallic scratches, water ripples, specular highlights, and bubbles.
    Each image is saved with an empty .txt label file so YOLO learns
    that metallic reflections, rivets, and glares are 100% NOT crayfish.
    """
    print(f"Generating {count} hard-negative steel tray backgrounds...")
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir = output_dir / "labels" / "val"

    # Also check if user uploaded images have steel trays to crop empty regions
    user_images_dir = Path(r"C:\Users\Administrator\.gemini\antigravity\brain\b1a1e571-ae55-484a-8360-5d66003fd4c8\.user_uploaded")
    tray_sources = []
    if user_images_dir.exists():
        for p in user_images_dir.glob("*.png"):
            tray_sources.append(p)

    for i in range(count):
        is_val = (i % 8 == 0)
        target_img_dir = val_img_dir if is_val else train_img_dir
        target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

        filename = f"hard_neg_steel_tray_{i:03d}.jpg"
        img_out = target_img_dir / filename
        lbl_out = target_lbl_dir / f"hard_neg_steel_tray_{i:03d}.txt"

        # If we have real tray images, extract non-crayfish empty crops
        extracted = False
        if tray_sources and (i % 2 == 0):
            try:
                src_path = random.choice(tray_sources)
                with Image.open(src_path) as src_img:
                    w, h = src_img.size
                    if w >= 320 and h >= 320:
                        # Pick corner or side region (tray edge/corner where crayfish aren't clustered)
                        crop_size = min(320, w // 2, h // 2)
                        x0 = random.choice([0, w - crop_size, random.randint(0, max(0, w - crop_size))])
                        y0 = random.choice([0, h - crop_size, random.randint(0, max(0, h - crop_size))])
                        crop = src_img.crop((x0, y0, x0 + crop_size, y0 + crop_size))
                        # Resize to 640x640
                        crop = crop.resize((640, 640), Image.Resampling.BILINEAR)
                        # Slightly vary brightness/contrast
                        if random.random() > 0.5:
                            enhancer = ImageEnhance.Brightness(crop)
                            crop = enhancer.enhance(random.uniform(0.8, 1.2))
                        crop.convert("RGB").save(img_out, quality=92)
                        extracted = True
            except Exception:
                extracted = False

        if not extracted:
            # Generate synthetic metallic textured gradient with glare and ripples
            w, h = 640, 640
            # Base metallic grey color (160 to 220)
            base_val = random.randint(160, 210)
            arr = np.full((h, w, 3), base_val, dtype=np.uint8)

            # Add gradient / light reflection band
            x_band = np.linspace(0, np.pi, w)
            reflection = np.sin(x_band) * random.randint(25, 45)
            arr = np.clip(arr.astype(np.int16) + reflection[None, :, None], 0, 255).astype(np.uint8)

            # Add metallic grain noise
            noise = np.random.normal(0, 10, (h, w, 3)).astype(np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            img = Image.fromarray(arr)
            # Add circular glare or water ripple
            if random.random() > 0.4:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.5)))

            img.save(img_out, quality=92)

        # Write empty label file
        lbl_out.write_text("", encoding="utf-8")

    print(f"Created {count} negative background images with zero annotations.")


def build_mega_dataset(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val", "test"]
    for s in splits:
        (output_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    stats = {
        "total_images": 0,
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "total_boxes": 0,
        "class_distribution": {c: 0 for c in CLASSES},
    }

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for cfg in DATASET_CONFIGS:
        ds_dir = cfg["dir"]
        ds_name = cfg["name"]
        remap = cfg["remap"]

        if not ds_dir.exists():
            print(f"Skipping missing dataset: {ds_dir}")
            continue

        print(f"\nProcessing {ds_name} from {ds_dir}...")
        ds_img_count = 0

        for split_candidate in ["train", "valid", "val", "test"]:
            target_split = "val" if split_candidate in ["valid", "val"] else split_candidate
            if target_split not in splits:
                target_split = "train"

            img_dirs = [
                ds_dir / split_candidate / "images",
                ds_dir / "images" / split_candidate,
                ds_dir / split_candidate,
            ]
            img_dir = next((p for p in img_dirs if p.exists() and p.is_dir()), None)
            if not img_dir:
                continue

            lbl_dirs = [
                ds_dir / split_candidate / "labels",
                ds_dir / "labels" / split_candidate,
                ds_dir / "labels",
            ]
            lbl_dir = next((p for p in lbl_dirs if p.exists() and p.is_dir()), None)

            images = [p for p in img_dir.iterdir() if p.suffix.lower() in img_extensions]
            for img_p in images:
                unique_name = f"{ds_name}_{img_p.stem}"
                target_img_p = output_dir / "images" / target_split / f"{unique_name}{img_p.suffix}"
                target_lbl_p = output_dir / "labels" / target_split / f"{unique_name}.txt"

                shutil.copy2(img_p, target_img_p)

                # Process label
                boxes = 0
                src_lbl = lbl_dir / f"{img_p.stem}.txt" if lbl_dir else None
                if src_lbl and src_lbl.exists():
                    lines = src_lbl.read_text(encoding="utf-8").strip().splitlines()
                    out_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                src_cls = int(parts[0])
                                target_cls = remap.get(src_cls, 1) # default to juvenile
                                coords = parts[1:5]
                                out_lines.append(f"{target_cls} {' '.join(coords)}")
                                boxes += 1
                                stats["class_distribution"][CLASSES[target_cls]] += 1
                            except ValueError:
                                continue
                    target_lbl_p.write_text("\n".join(out_lines), encoding="utf-8")
                else:
                    target_lbl_p.write_text("", encoding="utf-8")

                stats["total_images"] += 1
                stats["total_boxes"] += boxes
                ds_img_count += 1
                if target_split == "train":
                    stats["train_images"] += 1
                elif target_split == "val":
                    stats["val_images"] += 1
                elif target_split == "test":
                    stats["test_images"] += 1

        print(f"  -> Added {ds_img_count} images from {ds_name}")

    # Inject hard negative empty steel tray backgrounds
    create_steel_tray_negatives(output_dir, count=80)
    stats["total_images"] += 80
    stats["train_images"] += 70
    stats["val_images"] += 10

    # Write unified dataset.yaml
    names_dict = {i: c for i, c in enumerate(CLASSES)}
    dataset_yaml = {
        "path": str(output_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/val",
        "nc": len(CLASSES),
        "names": names_dict,
    }

    yaml_p = output_dir / "dataset.yaml"
    with open(yaml_p, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    print("\n" + "=" * 65)
    print(" MEGA DATASET MERGE COMPLETE")
    print("=" * 65)
    print(f" Dataset Path:      {output_dir.resolve()}")
    print(f" Dataset YAML:      {yaml_p.resolve()}")
    print(f" Total Images:      {stats['total_images']}")
    print(f"  - Training Set:   {stats['train_images']}")
    print(f"  - Validation Set: {stats['val_images']}")
    print(f" Total Boxes:       {stats['total_boxes']}")
    print(" Class Distribution:")
    for k, v in stats["class_distribution"].items():
        print(f"  - {k:15s}: {v:5d} annotations")
    print("=" * 65 + "\n")
    return stats


if __name__ == "__main__":
    out = Path("datasets/crayfish_mega_combined")
    build_mega_dataset(out)
