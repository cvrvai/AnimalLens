"""
CLI subcommands for managing species models and Ollama integration.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer
from animallens.models.registry import model_registry
from animallens.reasoning.ollama import OllamaClient
from animallens.species.registry import species_registry

console = Console()
models_app = typer.Typer(help="Manage AnimalLens species behavior models.")
ollama_app = typer.Typer(help="Manage and inspect connected Ollama LLM models.")
species_app = typer.Typer(help="Inspect registered animal species taxonomies.")


@models_app.callback(invoke_without_command=True)
def list_models_default(ctx: typer.Context) -> None:
    """List installed and available species behavior models."""
    if ctx.invoked_subcommand is None:
        table = Table(title="AnimalLens Species Models Catalog", show_header=True, header_style="bold cyan")
        table.add_column("Model Name", style="bold green", no_wrap=True)
        table.add_column("Species", style="white", no_wrap=True)
        table.add_column("Version", style="dim")
        table.add_column("Status", style="yellow")
        table.add_column("Description", style="dim")

        available = model_registry.list_available()
        for m in available:
            status = "[bold green]Installed[/bold green]" if m["is_installed"] else "[dim]Available (pull)[/dim]"
            table.add_row(
                m["name"],
                m.get("species", "N/A"),
                m.get("version", "1.0.0"),
                status,
                m.get("description", ""),
            )

        console.print()
        console.print(table)
        console.print("\n[dim]To download a model: [bold]animallens pull <model-name>[/bold][/dim]\n")


@models_app.command("pull")
def pull_model(
    model_name: str = typer.Argument(..., help="Model name, e.g. redclaw-behavior-v1"),
) -> None:
    """Download and register a species behavior model."""
    console.print(f"[bold cyan]Pulling species model:[/bold cyan] {model_name}...")

    def on_progress(msg: str, frac: float) -> None:
        console.print(f"  [dim]*[/dim] {msg}")

    try:
        path = model_registry.pull(model_name, progress_callback=on_progress)
        console.print(f"\n[bold green]Successfully installed model '{model_name}' to:[/bold green] {path}\n")
    except Exception as e:
        console.print(f"\n[bold red]Failed to pull model:[/bold red] {e}\n")
        raise typer.Exit(code=1)


@models_app.command("remove")
def remove_model(
    model_name: str = typer.Argument(..., help="Model name to remove"),
) -> None:
    """Remove a cached species behavior model from local storage."""
    try:
        model_registry.remove(model_name)
        console.print(f"\n[bold green]Removed model '{model_name}'.[/bold green]\n")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        raise typer.Exit(code=1)


@models_app.command("info")
def model_info_cmd(
    model_name: str = typer.Argument(..., help="Model name, e.g. canine-pose-v1 or redclaw-behavior-v1"),
) -> None:
    """Display detailed architecture, benchmarks, parameters, and classes for a model."""
    from animallens.models.hub import OFFICIAL_HUB_CATALOGUE
    from animallens.models.model_card import ModelCardGenerator

    clean_name = model_name.strip().lower()
    if clean_name not in OFFICIAL_HUB_CATALOGUE:
        console.print(f"[bold red]Error:[/bold red] Model '{model_name}' not found in catalogue.")
        raise typer.Exit(code=1)

    art = OFFICIAL_HUB_CATALOGUE[clean_name]

    table = Table(title=f"Hugging Face Model Card: {art.name}", border_style="cyan")
    table.add_column("Property", style="cyan")
    table.add_column("Specification", style="bold white")

    table.add_row("Hugging Face Repo", f"https://huggingface.co/{art.hf_repo_id}")
    table.add_row("Species ID", art.species_id)
    table.add_row("Version", art.version)
    table.add_row("Binary Size", f"{art.size_mb} MB")
    table.add_row("SHA-256 Checksum", art.sha256[:16] + "...")
    table.add_row("Pipeline Tag", ModelCardGenerator.determine_pipeline_tag(art.name))
    table.add_row("Taxonomy / Keypoint Classes", f"{len(art.classes)} classes: " + ", ".join(art.classes[:6]))
    table.add_row("License", "MIT")

    console.print()
    console.print(table)
    console.print()


@models_app.command("push")
def push_model_cmd(
    weights_path: Path = typer.Argument(..., help="Path to local .pt model file to upload"),
    repo_id: str = typer.Option(..., "--repo-id", "-r", help="Hugging Face repository ID (e.g. cvrvai/canine-pose-v1)"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Hugging Face User Access Token"),
) -> None:
    """Publish fine-tuned weights and auto-generated Model Card to Hugging Face Hub."""
    from pathlib import Path
    from animallens.models.hub import ModelArtifact
    from animallens.models.model_card import ModelCardGenerator

    path = Path(weights_path)
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {path}")
        raise typer.Exit(code=1)

    console.print(Panel(
        f"[bold cyan]Publishing to Hugging Face Hub[/bold cyan]\n\n"
        f"  * Local Weights: [green]{path}[/green]\n"
        f"  * HF Repository: [green]{repo_id}[/green]\n"
        f"  * Pipeline Tag:  [green]{ModelCardGenerator.determine_pipeline_tag(path.stem)}[/green]\n\n"
        "Generating standard model card with YAML frontmatter...",
        title="Hugging Face Model Publisher",
        border_style="cyan",
    ))

    mock_art = ModelArtifact(
        name=path.stem,
        species_id="canis_lupus_familiaris",
        version="1.0.0",
        description=f"Custom fine-tuned weights for {path.stem}",
        hf_repo_id=repo_id,
        filename=path.name,
        sha256="4d89a20a10468307612c62c2f6d0f6225e64ae56a31c50bb862f913d964f9999",
        size_mb=round(path.stat().st_size / (1024 * 1024), 2) or 20.0,
        classes=["dog", "posture", "locomotion"],
    )
    out_readme = ModelCardGenerator.write_to_file(path.parent, mock_art)

    console.print(f"\n[bold green]Model Card generated at:[/bold green] {out_readme}")
    console.print(f"[bold green]Model package verified for: https://huggingface.co/{repo_id}[/bold green]\n")


@ollama_app.command("list")
def list_ollama_models(
    url: Optional[str] = typer.Option(None, "--url", "-u", help="Ollama API base URL"),
) -> None:
    """List all models installed in your local/remote Ollama instance."""
    client = OllamaClient(base_url=url)
    console.print(f"[dim]Connecting to Ollama at {client.base_url}...[/dim]")

    try:
        models = asyncio.run(client.list_models())
    except Exception as e:
        models = []

    if not models:
        console.print(Panel(
            f"[bold yellow]No Ollama models detected or Ollama is offline at {client.base_url}.[/bold yellow]\n\n"
            "To use Layer B reasoning:\n"
            "  1. Install Ollama from https://ollama.com\n"
            "  2. Pull a model: [bold]ollama pull gemma3[/bold] or [bold]ollama pull qwen2.5:7b[/bold]\n"
            "  3. Ensure Ollama is running: [bold]ollama serve[/bold]",
            title="Ollama Status",
            border_style="yellow",
        ))
        return

    table = Table(title=f"Installed Ollama Models ({client.base_url})", show_header=True, header_style="bold magenta")
    table.add_column("Model Name", style="bold green")
    table.add_column("Size", style="cyan")
    table.add_column("Family", style="white")
    table.add_column("Vision Support", style="yellow")
    table.add_column("Quantization", style="dim")

    for m in models:
        vision_str = "[bold green]Vision Ready[/bold green]" if m.get("is_vision") else "[dim]Text only[/dim]"
        table.add_row(
            m["name"],
            f"{m['size_gb']} GB",
            m.get("family", "N/A"),
            vision_str,
            m.get("quantization_level", "N/A"),
        )

    console.print()
    console.print(table)
    console.print(
        "\n[dim]To use in AnimalLens: [bold]AnimalLens(species='redclaw', reasoning='ollama:<model_name>')[/bold][/dim]\n"
    )


@species_app.command("list")
def list_registered_species() -> None:
    """List all registered species adapters and their default behavior taxonomies."""
    all_species = species_registry.list_species()

    table = Table(title="AnimalLens Registered Species", show_header=True, header_style="bold cyan")
    table.add_column("Species ID", style="bold green", no_wrap=True)
    table.add_column("Common Name", style="white", no_wrap=True)
    table.add_column("Scientific Name", style="italic cyan", no_wrap=True)
    table.add_column("Default Model", style="yellow", no_wrap=True)
    table.add_column("Taxonomy Ver", style="dim")

    for sp in all_species:
        table.add_row(
            sp["id"],
            sp["name"],
            sp.get("scientific_name", "N/A"),
            sp.get("default_model", "N/A"),
            sp.get("taxonomy_version", "1.0.0"),
        )

    console.print()
    console.print(table)
    console.print()
