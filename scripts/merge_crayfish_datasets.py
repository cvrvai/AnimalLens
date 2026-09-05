"""
Multi-Stage Crayfish Dataset Merger & Harmonizer for AnimalLens.
Merges multiple YOLOv8 datasets (across all life stages: craylet, juvenile, adult, broodstock),
resolves filename collisions, re-maps class indices, and outputs a unified dataset.yaml ready for training.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml


# Standard Life Stage Taxonomy
CRAYFISH_STAGES = [
    "craylet",      # Stage 3 early juvenile (~0.15g)
    "juvenile",     # Mid juvenile (5g - 20g)
    "sub_adult",    # 25g - 45g
    "adult",        # 50g+ / Broodstock
]

STAGE_KEYWORDS_MAP = {
    "craylet": 0,
    "early_juvenile": 0,
    "baby": 0,
    "larvae": 0,
    "stage3": 0,
    "stage_3": 0,

    "juvenile": 1,
    "med_juvenile": 1,
    "nursery": 1,

    "sub_adult": 2,
    "subadult": 2,
    "growout": 2,

    "adult": 3,
    "broodstock": 3,
    "breeder": 3,
    "mature": 3,
    "male": 3,
    "female": 3,

    "crayfish": 0,
    "cherax_quadricarinatus": 0,
    "redclaw": 0,
}


def find_dataset_yaml(dataset_dir: Path) -> Optional[Path]:
    for candidate in ["data.yaml", "dataset.yaml", "data.yml"]:
        p = dataset_dir / candidate
        if p.exists():
            return p
    yamls = list(dataset_dir.glob("*.yaml")) + list(dataset_dir.glob("*.yml"))
    return yamls[0] if yamls else None


def load_dataset_classes(dataset_dir: Path) -> Dict[int, str]:
    """Parse class index -> name mapping from dataset YAML."""
    yaml_path = find_dataset_yaml(dataset_dir)
    if not yaml_path:
        return {0: "crayfish"}

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        names = cfg.get("names", [])
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, list):
            return {i: str(n) for i, n in enumerate(names)}
    except Exception:
        pass
    return {0: "crayfish"}


def merge_datasets(
    input_dirs: List[Path],
    output_dir: Path,
    mode: str = "unified",  # 'unified' or 'multistage'
    custom_class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Merge multiple YOLOv8 dataset directories into one."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "unified":
        target_classes = ["crayfish"]
    elif custom_class_names:
        target_classes = custom_class_names
    else:
        target_classes = CRAYFISH_STAGES

    print(f"\n========================================================")
    print(f" CRAYFISH DATASET MERGER & HARMONIZER")
    print(f" Mode:           {mode.upper()}")
    print(f" Target Classes: {target_classes}")
    print(f" Output Target:  {output_dir.resolve()}")
    print(f" Datasets to merge: {len(input_dirs)}")
    print(f"========================================================\n")

    # Setup directories
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
        "class_distribution": {c: 0 for c in target_classes},
    }

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for ds_idx, in_dir in enumerate(input_dirs):
        ds_name = in_dir.name
        source_classes = load_dataset_classes(in_dir)
        print(f"[{ds_idx+1}/{len(input_dirs)}] Processing dataset '{ds_name}'...")
        print(f"     Source Classes: {source_classes}")

        # Build class remap table for this dataset
        remap_table: Dict[int, int] = {}
        for src_id, src_name in source_classes.items():
            if mode == "unified":
                remap_table[src_id] = 0
            else:
                # Search matching stage keyword
                src_lower = src_name.lower().replace(" ", "_")
                mapped_id = None
                for kw, target_id in STAGE_KEYWORDS_MAP.items():
                    if kw in src_lower:
                        mapped_id = min(target_id, len(target_classes) - 1)
                        break
                if mapped_id is None:
                    # Default to closest class or 0
                    mapped_id = 0
                remap_table[src_id] = mapped_id

        # Scan for images and labels in splits
        for split_candidate in ["train", "valid", "val", "test"]:
            target_split = "val" if split_candidate in ["valid", "val"] else split_candidate
            if target_split not in splits:
                target_split = "train"

            # Check potential image locations
            img_dirs_to_check = [
                in_dir / split_candidate / "images",
                in_dir / "images" / split_candidate,
                in_dir / split_candidate,
            ]
            img_dir = next((p for p in img_dirs_to_check if p.exists() and p.is_dir()), None)
            if not img_dir:
                continue

            # Corresponding label dir
            lbl_dirs_to_check = [
                in_dir / split_candidate / "labels",
                in_dir / "labels" / split_candidate,
                in_dir / "labels",
            ]
            lbl_dir = next((p for p in lbl_dirs_to_check if p.exists() and p.is_dir()), None)

            images = [p for p in img_dir.iterdir() if p.suffix.lower() in img_extensions]
            for img_path in images:
                # Create unique filename to prevent collisions across datasets
                unique_stem = f"ds{ds_idx+1}_{img_path.stem}"
                target_img_path = output_dir / "images" / target_split / f"{unique_stem}{img_path.suffix}"
                target_lbl_path = output_dir / "labels" / target_split / f"{unique_stem}.txt"

                # Copy image
                shutil.copy2(img_path, target_img_path)

                # Process label
                box_count = 0
                src_lbl_file = lbl_dir / f"{img_path.stem}.txt" if lbl_dir else None
                if src_lbl_file and src_lbl_file.exists():
                    out_lines = []
                    lines = src_lbl_file.read_text(encoding="utf-8").strip().splitlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                src_cls = int(parts[0])
                                new_cls = remap_table.get(src_cls, 0)
                                coords = parts[1:5]
                                out_lines.append(f"{new_cls} {' '.join(coords)}")
                                box_count += 1
                                c_name = target_classes[new_cls]
                                stats["class_distribution"][c_name] += 1
                            except ValueError:
                                continue
                    if out_lines:
                        target_lbl_path.write_text("\n".join(out_lines), encoding="utf-8")
                else:
                    # Empty label file for background images
                    target_lbl_path.write_text("", encoding="utf-8")

                stats["total_images"] += 1
                stats["total_boxes"] += box_count
                if target_split == "train":
                    stats["train_images"] += 1
                elif target_split == "val":
                    stats["val_images"] += 1
                elif target_split == "test":
                    stats["test_images"] += 1

    # Write unified dataset.yaml
    names_dict = {i: c for i, c in enumerate(target_classes)}
    dataset_yaml_data = {
        "path": str(output_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test" if stats["test_images"] > 0 else "images/val",
        "nc": len(target_classes),
        "names": names_dict,
    }

    yaml_file = output_dir / "dataset.yaml"
    with open(yaml_file, "w", encoding="utf-8") as yf:
        yaml.dump(dataset_yaml_data, yf, default_flow_style=False, sort_keys=False)

    print("\n" + "=" * 60)
    print(" MERGED DATASET SUMMARY")
    print("=" * 60)
    print(f" Output Path:     {output_dir.resolve()}")
    print(f" Config File:     {yaml_file.resolve()}")
    print(f" Total Images:    {stats['total_images']}")
    print(f"  * Train:        {stats['train_images']}")
    print(f"  * Validation:   {stats['val_images']}")
    print(f"  * Test:         {stats['test_images']}")
    print(f" Total Boxes:     {stats['total_boxes']}")
    print(" Class Breakdown:")
    for cls_name, cnt in stats["class_distribution"].items():
        print(f"  - {cls_name:15s}: {cnt:5d} bounding boxes")
    print("=" * 60 + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Merge and harmonize multi-stage crayfish datasets for YOLOv8 training.")
    parser.add_argument("--inputs", "-i", nargs="+", required=True, help="List of dataset directories to merge")
    parser.add_argument("--output", "-o", default="datasets/crayfish_merged", help="Destination output directory")
    parser.add_argument("--mode", "-m", choices=["unified", "multistage"], default="unified",
                        help="'unified' (single crayfish class) or 'multistage' (craylet, juvenile, sub_adult, adult)")
    parser.add_argument("--classes", "-c", nargs="+", default=None, help="Optional explicit custom class list")

    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs if Path(p).exists()]
    if not input_paths:
        print(f"Error: None of the provided input paths exist: {args.inputs}")
        return

    merge_datasets(
        input_dirs=input_paths,
        output_dir=Path(args.output),
        mode=args.mode,
        custom_class_names=args.classes,
    )


if __name__ == "__main__":
    main()
