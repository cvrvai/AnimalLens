"""
Script to pull all unassigned and raw images from Roboflow project cc-aryuc/crayfish-zh9y5.
Downloads images from all batches (Mobile Upload, Uploaded 09/05, Reassigned 09/07)
and saves them to AnimalLens/datasets/crayfish_unassigned/ with full metadata index.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROBOFLOW_API_URL = "https://api.roboflow.com"
API_KEY = "WuOsvQe8dthqEbBCpGj0"
WORKSPACE = "cc-aryuc"
PROJECT = "crayfish-zh9y5"

OUTPUT_DIR = Path("e:/Projects/personal project/AnimalLens/datasets/crayfish_unassigned")
IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"
METADATA_FILE = OUTPUT_DIR / "metadata.json"


def fetch_batches() -> list[dict]:
    url = f"{ROBOFLOW_API_URL}/{WORKSPACE}/{PROJECT}/batches?api_key={API_KEY}"
    req = urllib.request.Request(url, headers={"User-Agent": "AnimalLens-Downloader/1.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        return data.get("batches", [])


def fetch_batch_image_ids(batch_id: str) -> list[str]:
    image_ids = []
    after = None
    while True:
        url = f"{ROBOFLOW_API_URL}/{WORKSPACE}/{PROJECT}/annotation-batches/{batch_id}/images?api_key={API_KEY}&limit=50"
        if after:
            url += f"&after={urllib.request.quote(after)}"
        req = urllib.request.Request(url, headers={"User-Agent": "AnimalLens-Downloader/1.0"})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            ids = data.get("imageIds", [])
            image_ids.extend(ids)
            after = data.get("nextPageToken")
            if not after or not ids:
                break
    return image_ids


def fetch_image_details(image_id: str) -> dict | None:
    url = f"{ROBOFLOW_API_URL}/{WORKSPACE}/{PROJECT}/images/{image_id}?api_key={API_KEY}"
    req = urllib.request.Request(url, headers={"User-Agent": "AnimalLens-Downloader/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            return data.get("image")
    except Exception as e:
        print(f"  [Warning] Failed fetching info for {image_id}: {e}")
        return None


def download_single_image(img_info: dict, batch_name: str) -> dict | None:
    image_id = img_info.get("id")
    raw_name = img_info.get("name") or f"photo-{image_id}.jpg"
    orig_url = img_info.get("urls", {}).get("original")
    if not orig_url:
        return None

    # Sanitize filename
    safe_name = f"{image_id}_{Path(raw_name).name}"
    target_path = IMAGES_DIR / safe_name

    # Download if not already downloaded
    if not target_path.exists() or target_path.stat().st_size == 0:
        try:
            req = urllib.request.Request(orig_url, headers={"User-Agent": "AnimalLens-Downloader/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                target_path.write_bytes(data)
        except Exception as e:
            print(f"  [Error] Downloading {safe_name}: {e}")
            return None

    return {
        "id": image_id,
        "filename": safe_name,
        "original_name": raw_name,
        "batch": batch_name,
        "local_path": str(target_path.resolve()),
        "url": orig_url,
        "width": img_info.get("metadata", {}).get("width"),
        "height": img_info.get("metadata", {}).get("height"),
        "annotation": img_info.get("annotation"),
        "is_annotated": bool(img_info.get("annotation")),
    }


def main():
    print("=== Roboflow Dataset Puller ===")
    print(f"Project: {WORKSPACE}/{PROJECT}")
    print(f"Target Directory: {OUTPUT_DIR}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch batches
    print("\n[Step 1/3] Fetching batches from Roboflow...")
    batches = fetch_batches()
    print(f"Found {len(batches)} batches:")
    total_batch_images = 0
    batch_map = {}
    for b in batches:
        num = b.get("images", 0)
        total_batch_images += num
        batch_map[b["id"]] = b.get("name", b["id"])
        print(f"  - [{b['id']}] {b.get('name')} ({num} images)")

    print(f"Total unassigned images across batches: {total_batch_images}")

    # 2. Fetch all image IDs
    print("\n[Step 2/3] Fetching image IDs for each batch...")
    all_images_to_fetch = []
    for bid, bname in batch_map.items():
        print(f"  Querying batch: {bname}...")
        ids = fetch_batch_image_ids(bid)
        print(f"    -> {len(ids)} image IDs found.")
        for img_id in ids:
            all_images_to_fetch.append((img_id, bname))

    print(f"\nTotal image IDs to download: {len(all_images_to_fetch)}")

    # 3. Fetch details and download images in parallel
    print(f"\n[Step 3/3] Downloading images with multi-threading (10 workers)...")
    downloaded_records = []
    t0 = time.time()

    def worker(item):
        img_id, bname = item
        info = fetch_image_details(img_id)
        if not info:
            return None
        return download_single_image(info, bname)

    completed = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(worker, item): item for item in all_images_to_fetch}
        for f in as_completed(futures):
            res = f.result()
            completed += 1
            if res:
                downloaded_records.append(res)
            if completed % 25 == 0 or completed == len(all_images_to_fetch):
                print(f"  Progress: {completed}/{len(all_images_to_fetch)} images ({len(downloaded_records)} succeeded)")

    duration = time.time() - t0
    print(f"\nDownload finished in {duration:.1f}s.")
    print(f"Successfully downloaded {len(downloaded_records)}/{len(all_images_to_fetch)} images.")

    # 4. Save metadata index
    METADATA_FILE.write_text(
        json.dumps(
            {
                "project": f"{WORKSPACE}/{PROJECT}",
                "pulled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_images": len(downloaded_records),
                "images": downloaded_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Metadata index saved to: {METADATA_FILE}")
    print("\nAll done!")


if __name__ == "__main__":
    main()
