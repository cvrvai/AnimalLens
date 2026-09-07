"""
AnimalLens AI Auto-Labeler for Roboflow Unassigned Datasets using Ollama Vision.

Automates the detection and bounding box generation for early juvenile crayfish
and uploads annotations directly to Roboflow or exports to local YOLO format.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    _script_dir = Path(__file__).resolve().parent
    # Load AnimalLens .env
    load_dotenv(_script_dir.parent / ".env")
    # Load aquaculture-system-next .env
    load_dotenv(_script_dir.parent.parent / "AIC-main-core" / "aquaculture-system-next" / ".env")
    load_dotenv(Path.cwd() / ".env")
except ImportError:
    pass

ROBOFLOW_API_URL = "https://api.roboflow.com"


def normalize_box(raw: dict) -> Optional[Tuple[float, float, float, float]]:
    """Convert box to normalized [ymin, xmin, ymax, xmax] bounded in [0.0, 1.0]."""
    if "box_2d" in raw and isinstance(raw["box_2d"], list) and len(raw["box_2d"]) == 4:
        ymin, xmin, ymax, xmax = [float(v) for v in raw["box_2d"]]
    elif "bbox" in raw and isinstance(raw["bbox"], list) and len(raw["bbox"]) == 4:
        xmin, ymin, xmax, ymax = [float(v) for v in raw["bbox"]]
    elif all(k in raw for k in ("ymin", "xmin", "ymax", "xmax")):
        ymin, xmin, ymax, xmax = float(raw["ymin"]), float(raw["xmin"]), float(raw["ymax"]), float(raw["xmax"])
    else:
        return None

    max_val = max(ymin, xmin, ymax, xmax)
    if max_val > 1.5:
        scale = 1000.0 if max_val > 100.0 else 100.0
        ymin /= scale
        xmin /= scale
        ymax /= scale
        xmax /= scale

    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin

    ymin = max(0.0, min(1.0, ymin))
    xmin = max(0.0, min(1.0, xmin))
    ymax = max(0.0, min(1.0, ymax))
    xmax = max(0.0, min(1.0, xmax))

    if ymax - ymin < 0.005 or xmax - xmin < 0.005:
        return None

    return ymin, xmin, ymax, xmax


def boxes_to_yolo(boxes: List[Tuple[float, float, float, float]]) -> str:
    """Convert [ymin, xmin, ymax, xmax] list to YOLO format string."""
    lines = []
    for ymin, xmin, ymax, xmax in boxes:
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin
        if width <= 0 or height <= 0:
            continue
        lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return "\n".join(lines)


def fetch_batches(api_key: str, workspace: str, project: str) -> List[dict]:
    url = f"{ROBOFLOW_API_URL}/{workspace}/{project}/batches?api_key={api_key}"
    req = urllib.request.Request(url, headers={"User-Agent": "AnimalLens-AutoLabeler/1.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        return data.get("batches", [])


def fetch_batch_image_ids(api_key: str, workspace: str, project: str, batch_id: str, limit: int = 50) -> List[str]:
    url = f"{ROBOFLOW_API_URL}/{workspace}/{project}/annotation-batches/{batch_id}/images?api_key={api_key}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "AnimalLens-AutoLabeler/1.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        return data.get("imageIds", [])


def fetch_image_details(api_key: str, workspace: str, project: str, image_id: str) -> Optional[dict]:
    url = f"{ROBOFLOW_API_URL}/{workspace}/{project}/images/{image_id}?api_key={api_key}"
    req = urllib.request.Request(url, headers={"User-Agent": "AnimalLens-AutoLabeler/1.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            return data.get("image")
    except Exception as e:
        print(f"  [Warning] Could not fetch image {image_id}: {e}")
        return None


def call_ollama_detect(
    image_bytes: bytes,
    api_key: str,
    endpoint: str = "https://ollama.com",
    model: str = "llama3.2-vision",
) -> List[Tuple[float, float, float, float]]:
    """Send image to Ollama vision model and return normalized bounding boxes."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    system_prompt = (
        "You are an expert aquaculture computer vision detector specializing in Redclaw Crayfish. "
        "Detect every visible juvenile crayfish in this tray photo. "
        "Return strictly a JSON object with 'detections': array of { 'label': 'early_juvenile', 'box_2d': [ymin, xmin, ymax, xmax] } "
        "where coordinates are normalized 0-1000."
    )

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Detect all juvenile crayfish. Return ONLY JSON.", "images": [b64]}
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.1}
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    chat_url = f"{endpoint.rstrip('/')}/api/chat"
    req = urllib.request.Request(chat_url, data=payload, headers=headers)

    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode())

    content = data.get("message", {}).get("content") or data.get("response") or "{}"
    parsed = {}
    try:
        parsed = json.loads(content)
    except Exception:
        clean = content.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(clean)
        except Exception:
            s = clean.find("{")
            e = clean.rfind("}")
            if s != -1 and e > s:
                try:
                    parsed = json.loads(clean[s:e+1])
                except Exception:
                    pass

    raw_list = parsed.get("detections") or parsed.get("objects") or parsed.get("boxes") or []
    boxes = []
    for item in raw_list:
        if isinstance(item, dict):
            nb = normalize_box(item)
            if nb:
                boxes.append(nb)
    return boxes


def upload_annotation_to_roboflow(
    api_key: str,
    project: str,
    image_id: str,
    image_name: str,
    yolo_string: str,
) -> bool:
    """Upload YOLO annotation directly to Roboflow project."""
    annotation_filename = f"{Path(image_name).stem}.txt"
    url = f"{ROBOFLOW_API_URL}/dataset/{project}/annotate/{image_id}?api_key={api_key}&name={urllib.request.quote(annotation_filename)}&overwrite=true"
    payload = json.dumps({
        "annotationFile": yolo_string,
        "labelmap": {"0": "early_juvenile"}
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return bool(data.get("success", True))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"  [Roboflow Upload Error {e.code}] {err_body}")
        return False
    except Exception as e:
        print(f"  [Roboflow Upload Error] {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="AnimalLens AI Auto-Labeler for Roboflow with Ollama")
    parser.add_argument("--batch", type=str, help="Roboflow Batch ID to process (or 'all')", default=None)
    parser.add_argument("--ollama-key", type=str, help="Ollama Cloud API Key", default=None)
    parser.add_argument("--endpoint", type=str, help="Ollama Endpoint", default="https://ollama.com")
    parser.add_argument("--model", type=str, help="Ollama Vision Model", default="gemma4:31b")
    parser.add_argument("--limit", type=int, help="Max images to process", default=50)
    parser.add_argument("--save-local", action="store_true", help="Save copy to local YOLO format", default=True)
    args = parser.parse_args()

    # Auto-map local model to Ollama Cloud model if endpoint is cloud
    if "ollama.com" in args.endpoint and args.model in ("llama3.2-vision", "qwen2.5-vl"):
        args.model = "gemma4:31b"

    rf_key = os.environ.get("ROBOFLOW_API_KEY", "WuOsvQe8dthqEbBCpGj0").strip()
    rf_workspace = os.environ.get("ROBOFLOW_WORKSPACE", "cc-aryuc").strip()
    rf_project = os.environ.get("ROBOFLOW_PROJECT", "crayfish-zh9y5").strip()

    ollama_key = (args.ollama_key or os.environ.get("OLLAMA_API_KEY") or "").strip()
    if not ollama_key and not args.endpoint.startswith("http://localhost") and not args.endpoint.startswith("http://127.0.0.1"):
        print("[Error] Ollama Cloud API Key is required. Pass --ollama-key or set OLLAMA_API_KEY.")
        sys.exit(1)

    print(f"=== AnimalLens AI Auto-Labeler ===")
    print(f"Roboflow Project: {rf_workspace}/{rf_project}")
    print(f"Ollama Model:     {args.model} ({args.endpoint})")

    # 1. Fetch batches
    print("\nFetching Roboflow batches...")
    batches = fetch_batches(rf_key, rf_workspace, rf_project)
    if not batches:
        print("No batches found in project.")
        return

    print("Available Batches:")
    for b in batches:
        print(f"  - [{b['id']}] {b['name']} ({b.get('images', 0)} images)")

    selected_batch_id = args.batch
    if not selected_batch_id:
        # Default to Mobile Upload
        mobile_b = next((b for b in batches if "mobile" in b["name"].lower()), batches[0])
        selected_batch_id = mobile_b["id"]
        print(f"\nAuto-selected batch: [{selected_batch_id}] {mobile_b['name']}")

    # 2. Fetch images
    print(f"\nFetching images for batch {selected_batch_id} (limit={args.limit})...")
    image_ids = fetch_batch_image_ids(rf_key, rf_workspace, rf_project, selected_batch_id, limit=args.limit)
    print(f"Found {len(image_ids)} images in batch.")

    local_images_dir = Path(_script_dir.parent / "datasets" / "crayfish_labeled" / "images")
    local_labels_dir = Path(_script_dir.parent / "datasets" / "crayfish_labeled" / "labels")
    if args.save_local:
        local_images_dir.mkdir(parents=True, exist_ok=True)
        local_labels_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    total_boxes = 0

    for i, img_id in enumerate(image_ids, 1):
        print(f"\n[{i}/{len(image_ids)}] Processing image {img_id}...")
        img_info = fetch_image_details(rf_key, rf_workspace, rf_project, img_id)
        if not img_info:
            continue

        orig_url = img_info.get("urls", {}).get("original")
        name = img_info.get("name") or f"photo-{img_id}.jpg"
        if not orig_url:
            print("  [Warning] No original URL for image.")
            continue

        # Check if already annotated
        if img_info.get("annotation"):
            print(f"  [Skip] Already annotated ({name}).")
            continue

        # Download image bytes
        try:
            req = urllib.request.Request(orig_url, headers={"User-Agent": "AnimalLens-AutoLabeler/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                img_bytes = resp.read()
        except Exception as e:
            print(f"  [Error] Failed to download {orig_url}: {e}")
            continue

        # Detect with Ollama
        print(f"  Calling Ollama {args.model} detection...")
        t0 = time.time()
        try:
            boxes = call_ollama_detect(img_bytes, ollama_key, args.endpoint, args.model)
            dur = time.time() - t0
            print(f"  Detected {len(boxes)} crayfish in {dur:.2f}s.")
        except Exception as e:
            print(f"  [Ollama Detection Error] {e}")
            continue

        yolo_txt = boxes_to_yolo(boxes)

        # Upload to Roboflow
        rf_ok = upload_annotation_to_roboflow(rf_key, rf_project, img_id, name, yolo_txt)
        if rf_ok:
            print(f"  Uploaded to Roboflow ✓")
        else:
            print(f"  Uploaded to Roboflow failed ✗")

        # Save locally
        if args.save_local:
            base_stem = Path(name).stem
            (local_labels_dir / f"{base_stem}.txt").write_text(yolo_txt, encoding="utf-8")
            (local_images_dir / name).write_bytes(img_bytes)
            print(f"  Saved locally to {local_labels_dir / f'{base_stem}.txt'}")

        success_count += 1
        total_boxes += len(boxes)

    print("\n==========================================")
    print(f"Completed! Successfully labeled {success_count}/{len(image_ids)} images.")
    print(f"Total crayfish bounding boxes generated: {total_boxes}")
    print("==========================================")


if __name__ == "__main__":
    main()
