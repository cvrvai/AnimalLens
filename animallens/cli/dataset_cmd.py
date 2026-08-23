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

    table = Table(title="Anti-Leakage Grouped Partition Summary", border_style="green")
    table.add_column("Split", style="bold")
    table.add_column("Sample Count", justify="right")
    table.add_column("Unique Sessions", justify="right")

    table.add_row("Train", str(len(result.train_samples)), str(len(result.train_sessions)))
    table.add_row("Validation", str(len(result.val_samples)), str(len(result.val_sessions)))
    table.add_row("Test", str(len(result.test_samples)), str(len(result.test_sessions)))
    table.add_row("[bold]Total[/bold]", f"[bold]{result.total_count}[/bold]", "-")

    console.print(table)
    console.print(f"\n[green]Manifests saved to:[/green] {out_path.resolve()}\n")
