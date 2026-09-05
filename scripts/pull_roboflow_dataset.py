"""
Automated Roboflow Dataset Puller for AnimalLens.
Supports both Roboflow Python SDK and Direct REST API fallback.
Downloads datasets in YOLOv8 format with automated extraction and validation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse
import urllib.request
import urllib.error
import yaml

try:
    from dotenv import load_dotenv
    _script_dir = Path(__file__).resolve().parent
    load_dotenv(_script_dir.parent / ".env")
    load_dotenv(Path.cwd() / ".env")
except ImportError:
    pass


def save_key_to_env(api_key: str) -> Path:
    """Save ROBOFLOW_API_KEY into .env file in AnimalLens root."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    lines = []
    found = False
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ROBOFLOW_API_KEY="):
                lines.append(f"ROBOFLOW_API_KEY={api_key}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"ROBOFLOW_API_KEY={api_key}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_path


def parse_roboflow_url(url_str: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Parse Roboflow URL to extract (workspace, project, version)."""
    p = urlparse(url_str.strip())
    parts = [part for part in p.path.strip("/").split("/") if part]
    workspace = parts[0] if len(parts) > 0 else None
    project = parts[1] if len(parts) > 1 else None
    version = None
    if len(parts) > 2 and parts[2].isdigit():
        version = int(parts[2])
    return workspace, project, version


def fetch_project_metadata(api_key: str, workspace: str, project: str) -> Optional[dict]:
    """Fetch project details and available versions from Roboflow REST API."""
    url = f"https://api.roboflow.com/{workspace}/{project}?api_key={api_key}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "AnimalLens-Roboflow-Downloader/1.0"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        print(f"[Roboflow API Notice] HTTP {e.code}: {error_body}")
        return None
    except Exception as e:
        print(f"[Roboflow API Notice] Could not fetch project metadata: {e}")
        return None


def pull_via_sdk(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    format_type: str,
    target_dir: Path,
) -> bool:
    """Attempt download using official Roboflow Python SDK."""
    try:
        from roboflow import Roboflow
        print(f"[SDK] Initializing Roboflow client...")
        rf = Roboflow(api_key=api_key)
        ws = rf.workspace(workspace)
        proj = ws.project(project)
        ver = proj.version(version)
        print(f"[SDK] Downloading version {version} in '{format_type}' format to {target_dir}...")
        dataset = ver.download(format_type, location=str(target_dir))
        return True
    except ImportError:
        print("[SDK] roboflow package not found or import failed. Using Direct REST API fallback...")
        return False
    except Exception as e:
        print(f"[SDK Warning] SDK download encountered: {e}. Switching to Direct REST API...")
        return False


def pull_via_direct_api(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    format_type: str,
    target_dir: Path,
) -> bool:
    """Download using Roboflow Direct REST API."""
    api_url = f"https://api.roboflow.com/{workspace}/{project}/{version}/{format_type}?api_key={api_key}"
    print(f"[REST API] Requesting download URL from: https://api.roboflow.com/{workspace}/{project}/{version}/{format_type}")

    req = urllib.request.Request(
        api_url,
        headers={"User-Agent": "AnimalLens-Roboflow-Downloader/1.0"}
    )

    try:
        with urllib.request.urlopen(req) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8")
        print(f"[REST API Error] HTTP {e.code}: {error_msg}")
        return False
    except Exception as e:
        print(f"[REST API Error] Request failed: {e}")
        return False

    # Retrieve zip download link
    zip_url = None
    if isinstance(resp_data, dict):
        if "export" in resp_data and "link" in resp_data["export"]:
            zip_url = resp_data["export"]["link"]
        elif "link" in resp_data:
            zip_url = resp_data["link"]

    if not zip_url:
        print(f"[REST API Error] Unexpected response format from Roboflow: {resp_data}")
        return False

    print(f"[REST API] Downloading dataset archive from Roboflow...")
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
        tmp_zip_path = Path(tmp_file.name)

    try:
        def report_progress(block_num: int, block_size: int, total_size: int):
            if total_size > 0:
                percent = min(100.0, (block_num * block_size / total_size) * 100)
                sys.stdout.write(f"\r  * Downloading: {percent:.1f}% ({block_num * block_size // 1024} KB / {total_size // 1024} KB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(zip_url, tmp_zip_path, reporthook=report_progress)
        print("\n[REST API] Download complete. Extracting dataset files...")

        with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)

        print(f"[REST API] Successfully extracted to: {target_dir.resolve()}")
        return True
    finally:
        if tmp_zip_path.exists():
            tmp_zip_path.unlink()


def verify_and_report_dataset(target_dir: Path) -> dict:
    """Inspect and report the extracted YOLO dataset structure."""
    yaml_candidates = list(target_dir.glob("*.yaml")) + list(target_dir.glob("*.yml"))
    data_yaml_path = target_dir / "data.yaml"
    if not data_yaml_path.exists() and yaml_candidates:
        data_yaml_path = yaml_candidates[0]

    report = {
        "path": str(target_dir.resolve()),
        "yaml_file": str(data_yaml_path.resolve()) if data_yaml_path.exists() else None,
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "classes": [],
    }

    # Count images in splits
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for split in ["train", "valid", "val", "test"]:
        split_dir = target_dir / split / "images"
        if not split_dir.exists():
            split_dir = target_dir / "images" / split
        if not split_dir.exists():
            split_dir = target_dir / split

        if split_dir.exists():
            count = sum(1 for p in split_dir.iterdir() if p.suffix.lower() in img_exts)
            if "train" in split:
                report["train_images"] += count
            elif "val" in split:
                report["val_images"] += count
            elif "test" in split:
                report["test_images"] += count

    # If no validation set exists, split ~15-20% from train into valid
    if report["val_images"] == 0 and report["train_images"] > 3:
        train_img_dir = target_dir / "train" / "images"
        train_lbl_dir = target_dir / "train" / "labels"
        val_img_dir = target_dir / "valid" / "images"
        val_lbl_dir = target_dir / "valid" / "labels"
        val_img_dir.mkdir(parents=True, exist_ok=True)
        val_lbl_dir.mkdir(parents=True, exist_ok=True)

        all_train_imgs = sorted([p for p in train_img_dir.iterdir() if p.suffix.lower() in img_exts])
        num_to_val = max(1, int(len(all_train_imgs) * 0.18))
        import shutil
        for img_p in all_train_imgs[-num_to_val:]:
            lbl_p = train_lbl_dir / f"{img_p.stem}.txt"
            shutil.move(str(img_p), str(val_img_dir / img_p.name))
            if lbl_p.exists():
                shutil.move(str(lbl_p), str(val_lbl_dir / lbl_p.name))
        report["train_images"] -= num_to_val
        report["val_images"] += num_to_val

    # Normalize data.yaml paths for Ultralytics YOLOv8 compatibility
    if data_yaml_path.exists():
        try:
            with open(data_yaml_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg["path"] = str(target_dir.resolve()).replace("\\", "/")
            cfg["train"] = "train/images"
            cfg["val"] = "valid/images" if (target_dir / "valid").exists() else "train/images"
            if "test" in cfg and not (target_dir / "test").exists():
                del cfg["test"]
            with open(data_yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False)

            names = cfg.get("names", [])
            if isinstance(names, dict):
                report["classes"] = [names[k] for k in sorted(names.keys())]
            elif isinstance(names, list):
                report["classes"] = names
        except Exception:
            pass

    print("\n" + "=" * 60)
    print(" ROBOFLOW DATASET SUMMARY ")
    print("=" * 60)
    print(f" Directory:        {report['path']}")
    print(f" Dataset YAML:     {report['yaml_file']}")
    print(f" Train Images:     {report['train_images']}")
    print(f" Validation Imgs:  {report['val_images']}")
    print(f" Test Images:      {report['test_images']}")
    print(f" Total Images:     {report['train_images'] + report['val_images'] + report['test_images']}")
    print(f" Classes ({len(report['classes'])}):     {', '.join(report['classes']) if report['classes'] else 'None declared'}")
    print("=" * 60 + "\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="Pull Roboflow dataset for AnimalLens YOLOv8 training.")
    parser.add_argument("--url", "-u", default=None, help="Roboflow project or version URL (e.g. https://app.roboflow.com/cc-aryuc/crayfish-zh9y5/train)")
    parser.add_argument("--workspace", "-w", default=None, help="Roboflow workspace ID (e.g. cc-aryuc)")
    parser.add_argument("--project", "-p", default=None, help="Roboflow project ID (e.g. crayfish-zh9y5 or crayfish-4bvu4-mdooz)")
    parser.add_argument("--version", "-v", type=int, default=None, help="Dataset version number (e.g. 1)")
    parser.add_argument("--format", "-f", default="yolov8", help="Export format (default: yolov8)")
    parser.add_argument("--api-key", "-k", default=os.getenv("ROBOFLOW_API_KEY", ""), help="Roboflow Private API Key")
    parser.add_argument("--set-key", default=None, help="Save your Roboflow Private API Key to .env and exit")
    parser.add_argument("--output", "-o", default=None, help="Target output folder (default: datasets/{project}_v{version})")

    args = parser.parse_args()

    # Handle --set-key
    if args.set_key:
        saved_file = save_key_to_env(args.set_key.strip())
        print(f"\n[OK] Successfully saved ROBOFLOW_API_KEY to: {saved_file.resolve()}")
        print("You can now pull datasets without passing --api-key!\n")
        sys.exit(0)

    workspace = args.workspace
    project = args.project
    version = args.version

    # Parse URL if provided
    if args.url:
        parsed_ws, parsed_proj, parsed_ver = parse_roboflow_url(args.url)
        if parsed_ws and not workspace:
            workspace = parsed_ws
        if parsed_proj and not project:
            project = parsed_proj
        if parsed_ver and not version:
            version = parsed_ver

    # Defaults from env or fallback
    if not workspace:
        workspace = os.getenv("ROBOFLOW_WORKSPACE", "cc-aryuc")
    if not project:
        project = os.getenv("ROBOFLOW_PROJECT", "crayfish-zh9y5")

    api_key = args.api_key.strip()
    if not api_key:
        api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()

    if not api_key:
        for env_path in [Path(".env"), Path("../.env"), Path("animallens/.env")]:
            if env_path.exists():
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("ROBOFLOW_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            api_key = val
                            break
            if api_key:
                break

    if not api_key:
        print("\n" + "=" * 65)
        print(" [!] ROBOFLOW PRIVATE API KEY REQUIRED")
        print("=" * 65)
        print(f" 1. Open: https://app.roboflow.com/{workspace}/settings/api")
        print(" 2. Copy your Private API Key.")
        print(" 3. Paste it below, or add to AnimalLens/.env: ROBOFLOW_API_KEY=...")
        print("=" * 65 + "\n")
        try:
            api_key = input("Enter your Roboflow API key (or Ctrl+C to cancel): ").strip()
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(1)

        if api_key:
            save_ans = input("Save this API key to .env for future commands? [Y/n]: ").strip().lower()
            if save_ans in ("", "y", "yes"):
                save_key_to_env(api_key)
                print("[OK] Saved to AnimalLens/.env!")

    if not api_key:
        print("Error: No API key provided. Exiting.")
        sys.exit(1)

    # If version not specified or from /train URL, query project to find available versions
    if not version:
        print(f"Querying project metadata for '{workspace}/{project}'...")
        meta = fetch_project_metadata(api_key, workspace, project)
        if meta:
            # In Roboflow REST API, top-level 'versions' is a list, while project['versions'] is an integer count
            raw_versions = meta.get("versions", [])
            if not isinstance(raw_versions, list):
                raw_versions = []

            proj_data = meta.get("project", {})
            if isinstance(proj_data, dict):
                v_count = proj_data.get("versions", 0)
                if isinstance(v_count, int) and v_count > 0 and not raw_versions:
                    raw_versions = list(range(1, v_count + 1))

            version_numbers = []
            for v_item in raw_versions:
                if isinstance(v_item, dict):
                    v_id = str(v_item.get("id", ""))
                    num_part = v_id.split("/")[-1] if "/" in v_id else str(v_item.get("version", v_item.get("name", "")))
                    if num_part.isdigit():
                        version_numbers.append(int(num_part))
                elif isinstance(v_item, int):
                    version_numbers.append(v_item)

            if version_numbers:
                version = max(version_numbers)
                print(f"[Roboflow Project] Found existing version(s): {sorted(version_numbers)}. Using latest: Version {version}.")
            else:
                print(f"[Roboflow Project] Project '{workspace}/{project}' has no versions generated yet.")
                print("Generating Version 1 automatically via Roboflow API (640x640 YOLO format)...")
                try:
                    from roboflow import Roboflow
                    rf = Roboflow(api_key=api_key)
                    proj = rf.workspace(workspace).project(project)
                    settings = {
                        "preprocessing": {"auto-orient": True, "resize": {"width": 640, "height": 640, "format": "Stretch to"}},
                        "augmentation": {}
                    }
                    version = proj.generate_version(settings=settings)
                    print(f"[Roboflow API] Successfully generated Version {version}!")
                except Exception as gen_err:
                    print(f"\n[!] Automatic generation failed: {gen_err}")
                    print(" Please visit https://app.roboflow.com/{workspace}/{project}, click 'Generate', and create Version 1.")
                    sys.exit(1)
        else:
            # Fallback to default version 1
            version = 1

    target_dir = Path(args.output) if args.output else Path("datasets") / f"{project}_v{version}"
    print(f"\nPulling dataset: {workspace}/{project} (Version {version}) into {target_dir.resolve()}...")

    # Try SDK first, fall back to REST API
    success = pull_via_sdk(
        api_key=api_key,
        workspace=workspace,
        project=project,
        version=version,
        format_type=args.format,
        target_dir=target_dir,
    )

    if not success:
        success = pull_via_direct_api(
            api_key=api_key,
            workspace=workspace,
            project=project,
            version=version,
            format_type=args.format,
            target_dir=target_dir,
        )

    if not success:
        print("[Error] Failed to pull dataset from Roboflow.")
        sys.exit(1)

    verify_and_report_dataset(target_dir)


if __name__ == "__main__":
    main()
