"""
CLI command for 1-click model training and transfer learning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer

console = Console()


def train_model_cli(
    video: Optional[Path] = typer.Option(None, "--video", "-v", help="Path to raw video clip for auto-dataset extraction"),
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d", help="Path to existing dataset.yaml config"),
    base_model: str = typer.Option("yolov8s.pt", "--base-model", "-m", help="Base starting weights"),
    epochs: int = typer.Option(50, "--epochs", "-e", help="Training epoch count"),
    batch: int = typer.Option(16, "--batch", "-b", help="Batch size"),
    imgsz: int = typer.Option(640, "--imgsz", help="Image resolution"),
    device: str = typer.Option("cpu", "--device", help="Training device (cpu, 0, mps)"),
    output_dir: Path = typer.Option(Path("models/trained"), "--output", "-o", help="Project output directory"),
    name: str = typer.Option("custom_canine_v1", "--name", "-n", help="Experiment name"),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume from last checkpoint"),
) -> None:
    """
    Train or fine-tune custom YOLOv8 perception models directly from video or dataset.
    """
    from animallens.training.dataset_builder import VideoDatasetBuilder
    from animallens.training.trainer import ModelTrainer

    console.print(Panel(
        f"[bold cyan]AnimalLens 1-Click Deep Learning Model Trainer[/bold cyan]\n\n"
        f"  * Base Model:       [green]{base_model}[/green]\n"
        f"  * Epochs:           [green]{epochs}[/green]\n"
        f"  * Batch Size:       [green]{batch}[/green]\n"
        f"  * Device:           [green]{device}[/green]\n"
        f"  * Project Output:   [green]{output_dir / name}[/green]",
        title="Training Configuration",
        border_style="cyan",
    ))

    # 1. If video is provided, build dataset automatically
    dataset_yaml = dataset
    if video:
        if not video.exists():
            console.print(f"[bold red]Error:[/bold red] Video file not found: {video}")
            raise typer.Exit(code=1)

        console.print(f"\n[cyan]Extracting training keyframes from:[/cyan] {video.name}...")
        builder = VideoDatasetBuilder(output_dir=output_dir / "dataset")
        frames = builder.extract_keyframes(video, sample_fps=2.0)
        console.print(f"[green]Extracted {len(frames)} frames.[/green] Generating initial pseudo-labels...")
        builder.generate_pseudo_labels()
        dataset_yaml = builder.write_yaml_config(classes=["dog"])

    if not dataset_yaml or not dataset_yaml.exists():
        console.print("[bold red]Error:[/bold red] Please provide either --video or --dataset with a valid dataset.yaml.")
        raise typer.Exit(code=1)

    # 2. Run Model Training
    console.print(f"\n[bold green]Starting transfer learning fine-tuning...[/bold green]\n")
    trainer = ModelTrainer(base_model=base_model, project_dir=output_dir, experiment_name=name)
    report = trainer.train(
        dataset_yaml=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        resume=resume,
    )

    # 3. Print Results Summary Table
    table = Table(title="Training Results & Checkpoint Registry", border_style="green")
    table.add_column("Metric / Checkpoint", style="cyan")
    table.add_column("Value / Path", style="bold white")

    table.add_row("Status", f"[bold green]{report.status}[/bold green]")
    table.add_row("Epochs Completed", str(report.epochs_completed))
    table.add_row("Validation mAP@50", f"{report.map50:.3f}")
    table.add_row("Validation mAP@50-95", f"{report.map50_95:.3f}")
    table.add_row("Best Weights Checkpoint", str(report.best_weights_path))
    table.add_row("Last Savepoint (Resume)", str(report.last_weights_path))
    if report.onnx_weights_path:
        table.add_row("ONNX Edge Model", str(report.onnx_weights_path))

    console.print(table)
    console.print(f"\n[bold green]Model training complete! Checkpoint saved to {report.best_weights_path}[/bold green]\n")
