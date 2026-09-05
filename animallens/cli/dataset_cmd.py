"""
CLI commands for dataset management, anti-leakage splitting, and Cohen's Kappa validation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer
from animallens.datasets.converter import BBoxConverter, DatasetExporter
from animallens.datasets.kappa import CohenKappaValidator
from animallens.datasets.partitioner import AntiLeakagePartitioner, DatasetSample

console = Console()
dataset_app = typer.Typer(
    name="dataset",
    help="Dataset management, anti-leakage splitting, and inter-rater reliability tools",
    add_completion=False,
    no_args_is_help=True,
)


@dataset_app.command(name="kappa")
def calculate_kappa_cmd(
    annotator1_file: str = typer.Argument(..., help="Path to JSON file containing Annotator 1 label array"),
    annotator2_file: str = typer.Argument(..., help="Path to JSON file containing Annotator 2 label array"),
    threshold: float = typer.Option(0.75, "--threshold", "-t", help="Minimum required Kappa threshold"),
) -> None:
    """Compute Cohen's Kappa coefficient between two annotators."""
    try:
        with open(annotator1_file, "r", encoding="utf-8") as f:
            a1 = json.load(f)
        with open(annotator2_file, "r", encoding="utf-8") as f:
            a2 = json.load(f)
    except Exception as e:
        console.print(f"[bold red]Failed to load annotation files:[/bold red] {e}")
        raise typer.Exit(code=1)

    report = CohenKappaValidator.compute_kappa(a1, a2, threshold=threshold)

    table = Table(title="Inter-Rater Reliability Report (Cohen's Kappa)", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")

    table.add_row("Cohen's Kappa (κ)", f"{report.cohen_kappa:.4f}")
    table.add_row("Observed Agreement (Po)", f"{report.observed_agreement * 100:.1f}%")
    table.add_row("Chance Agreement (Pe)", f"{report.chance_agreement * 100:.1f}%")
    table.add_row("Sample Count", str(report.sample_count))
    table.add_row("Interpretation", report.interpretation)
    status_text = "[bold green]PASS (Ready for Training)[/bold green]" if report.is_valid_for_training else "[bold red]FAIL (Needs Refinement)[/bold red]"
    table.add_row("Training Qualification", status_text)

    console.print(table)


@dataset_app.command(name="split")
def split_dataset_cmd(
    input_json: str = typer.Argument(..., help="JSON file containing array of dataset samples"),
    output_dir: str = typer.Option("datasets/splits", "--out", "-o", help="Directory to save split manifests"),
    train_ratio: float = typer.Option(0.70, "--train", help="Train partition ratio"),
    val_ratio: float = typer.Option(0.15, "--val", help="Validation partition ratio"),
    test_ratio: float = typer.Option(0.15, "--test", help="Test partition ratio"),
) -> None:
    """Partition dataset samples using session/tank grouping with 0% temporal leakage."""
    try:
        with open(input_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        samples = [DatasetSample(**item) for item in raw]
    except Exception as e:
        console.print(f"[bold red]Failed to load dataset manifest:[/bold red] {e}")
        raise typer.Exit(code=1)

    partitioner = AntiLeakagePartitioner(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    result = partitioner.split_by_session(samples)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "train.json", "w", encoding="utf-8") as f:
        json.dump([s.model_dump() for s in result.train_samples], f, indent=2)
    with open(out_path / "val.json", "w", encoding="utf-8") as f:
        json.dump([s.model_dump() for s in result.val_samples], f, indent=2)
    with open(out_path / "test.json", "w", encoding="utf-8") as f:
        json.dump([s.model_dump() for s in result.test_samples], f, indent=2)

    # Automatically construct YOLOv8 directory structure if source files exist
    import shutil
    base_raw_dir = Path(input_json).parent

    for split_name, samples_list in [("train", result.train_samples), ("val", result.val_samples), ("test", result.test_samples)]:
        split_img_dir = out_path / "images" / split_name
        split_lbl_dir = out_path / "labels" / split_name
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for s in samples_list:
            src_img = base_raw_dir / s.file_path
            if src_img.exists():
                shutil.copy(src_img, split_img_dir / src_img.name)
                src_lbl = base_raw_dir / "labels" / f"{src_img.stem}.txt"
                if src_lbl.exists():
                    shutil.copy(src_lbl, split_lbl_dir / src_lbl.name)

    # Generate dataset.yaml
    yaml_text = DatasetExporter.generate_yolo_yaml(
        dataset_dir=out_path,
        class_names=["cherax_quadricarinatus"],
        train_path="images/train",
        val_path="images/val",
        test_path="images/test",
    )
    with open(out_path / "dataset.yaml", "w", encoding="utf-8") as yf:
        yf.write(yaml_text)

    table = Table(title="Anti-Leakage Grouped Partition Summary", border_style="green")
    table.add_column("Split", style="bold")
    table.add_column("Sample Count", justify="right")
    table.add_column("Unique Sessions", justify="right")

    table.add_row("Train", str(len(result.train_samples)), str(len(result.train_sessions)))
    table.add_row("Validation", str(len(result.val_samples)), str(len(result.val_sessions)))
    table.add_row("Test", str(len(result.test_samples)), str(len(result.test_sessions)))
    table.add_row("[bold]Total[/bold]", f"[bold]{result.total_count}[/bold]", "-")

    console.print(table)
    console.print(f"\n[green]YOLO Dataset & Manifests saved to:[/green] {out_path.resolve()}\n")


@dataset_app.command(name="pull-roboflow")
def pull_roboflow_cmd(
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Roboflow project or version URL"),
    workspace: str = typer.Option("cc-aryuc", "--workspace", "-w", help="Roboflow workspace ID"),
    project: str = typer.Option("crayfish-zh9y5", "--project", "-p", help="Roboflow project ID"),
    version: Optional[int] = typer.Option(None, "--version", "-v", help="Dataset version number"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="Roboflow Private API Key"),
    output: Optional[str] = typer.Option(None, "--out", "-o", help="Destination folder"),
) -> None:
    """Pull dataset from Roboflow in YOLOv8 format via CLI."""
    import os
    from scripts.pull_roboflow_dataset import (
        pull_via_direct_api,
        pull_via_sdk,
        verify_and_report_dataset,
        parse_roboflow_url,
        fetch_project_metadata,
    )

    if url:
        parsed_ws, parsed_proj, parsed_ver = parse_roboflow_url(url)
        if parsed_ws:
            workspace = parsed_ws
        if parsed_proj:
            project = parsed_proj
        if parsed_ver and not version:
            version = parsed_ver

    resolved_key = api_key or os.getenv("ROBOFLOW_API_KEY", "")
    if not resolved_key:
        console.print(f"\n[bold yellow]No API key provided.[/bold yellow] Get your key from: [cyan]https://app.roboflow.com/{workspace}/settings/api[/cyan]")
        resolved_key = typer.prompt("Roboflow API Key", hide_input=True)

    if not version:
        meta = fetch_project_metadata(resolved_key, workspace, project)
        if meta:
            proj_data = meta.get("project", meta)
            raw_versions = proj_data.get("versions", [])
            version_numbers = []
            for v_item in raw_versions:
                if isinstance(v_item, dict):
                    v_id = v_item.get("id", "")
                    num_part = v_id.split("/")[-1] if "/" in v_id else v_item.get("name", "")
                    if num_part.isdigit():
                        version_numbers.append(int(num_part))
                elif isinstance(v_item, int):
                    version_numbers.append(v_item)
            if version_numbers:
                version = max(version_numbers)
                console.print(f"[bold green]Auto-detected latest generated version:[/bold green] Version {version}")
            else:
                console.print(f"\n[bold red]Project '{workspace}/{project}' has no generated versions yet![/bold red]")
                console.print("Please go to Roboflow, click 'Generate' on the left menu, and create Version 1 before exporting.")
                raise typer.Exit(code=1)
        else:
            version = 1

    target_dir = Path(output) if output else Path("datasets") / f"{project}_v{version}"
    console.print(f"[bold cyan]Pulling Roboflow dataset:[/bold cyan] {workspace}/{project} (v{version}) -> {target_dir}...")

    success = pull_via_sdk(resolved_key, workspace, project, version, "yolov8", target_dir)
    if not success:
        success = pull_via_direct_api(resolved_key, workspace, project, version, "yolov8", target_dir)

    if success:
        console.print(f"\n[bold green]Dataset successfully downloaded and ready for training![/bold green]")
        verify_and_report_dataset(target_dir)
    else:
        console.print(f"\n[bold red]Failed to download dataset. Please verify API key and network connection.[/bold red]")
        raise typer.Exit(code=1)


@dataset_app.command(name="merge")
def merge_datasets_cmd(
    inputs: List[str] = typer.Argument(..., help="Dataset directories to merge"),
    output: str = typer.Option("datasets/crayfish_merged", "--out", "-o", help="Target output folder"),
    mode: str = typer.Option("unified", "--mode", "-m", help="Merge mode: 'unified' (single class) or 'multistage' (life stages)"),
) -> None:
    """Merge and harmonize multiple datasets across life stages into a unified YOLOv8 dataset."""
    from scripts.merge_crayfish_datasets import merge_datasets

    input_paths = [Path(p) for p in inputs if Path(p).exists()]
    if not input_paths:
        console.print(f"[bold red]Error:[/bold red] None of the input paths exist: {inputs}")
        raise typer.Exit(code=1)

    merge_datasets(input_paths, Path(output), mode=mode)
    console.print(f"\n[bold green]Datasets successfully merged into:[/bold green] {Path(output).resolve()}\n")
