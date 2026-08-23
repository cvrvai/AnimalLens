"""
CLI subcommand for analyzing media files and streams with AnimalLens.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer
from animallens.core.schemas import AnalysisResult
from animallens.sdk import AnimalLens

console = Console()


def analyze_source_cli(
    source_path: str = typer.Argument(..., help="Path or URL to image, video file, or camera"),
    species: str = typer.Option("redclaw", "--species", "-s", help="Species identifier (e.g. redclaw, pig)"),
    reasoning: Optional[str] = typer.Option(None, "--reasoning", "-r", help="Optional reasoning provider, e.g. 'ollama:gemma3'"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Optional file path to save structured JSON output"),
    format_type: str = typer.Option("table", "--format", "-f", help="Output format: table, json, or timeline"),
    sample_fps: float = typer.Option(5.0, "--fps", help="Frame sampling rate for recorded video"),
) -> None:
    """Analyze an image or recorded video for animal behaviors."""
    console.print(f"\n[bold cyan]AnimalLens Analysis[/bold cyan]")
    console.print(f"  [dim]Source:[/dim]    {source_path}")
    console.print(f"  [dim]Species:[/dim]   {species}")
    console.print(f"  [dim]Reasoning:[/dim] {reasoning or 'None (Layer A only)'}\n")

    try:
        lens = AnimalLens(species=species, reasoning=reasoning)
        result: AnalysisResult = lens.analyze(source_path, sample_fps=sample_fps)
    except Exception as e:
        console.print(f"[bold red]Analysis failed:[/bold red] {e}")
        raise typer.Exit(code=1)

    if format_type == "json":
        json_str = json.dumps(result.model_dump(mode="json"), indent=2)
        console.print(json_str)
    elif format_type == "timeline":
        console.print(Panel(result.format_timeline_text(), title=f"Timeline for {species}"))
    else:
        # Table output
        table = Table(title=f"Detected Behaviors ({len(result.behaviors)} events)", show_header=True, header_style="bold magenta")
        table.add_column("Timestamp", style="cyan", width=12, no_wrap=True)
        table.add_column("Category", style="yellow", width=18, no_wrap=True)
        table.add_column("Behavior Label", style="bold green", width=20, no_wrap=True)
        table.add_column("Confidence", style="white", width=12, no_wrap=True)
        table.add_column("Subjects", style="dim", width=10, no_wrap=True)

        for event in result.behaviors:
            conf_str = f"{event.behavior.confidence:.1%}"
            if event.behavior.is_uncertain:
                conf_str += " [dim red](uncertain)[/dim red]"
            table.add_row(
                f"{event.temporal.start:.1f}s - {event.temporal.end:.1f}s",
                event.behavior.category,
                event.behavior.label,
                conf_str,
                str(len(event.subjects)),
            )

        console.print(table)

        if result.reasoning and result.reasoning.explanation:
            console.print()
            console.print(Panel(
                f"[bold cyan]Summary:[/bold cyan] {result.reasoning.summary}\n\n"
                f"[bold cyan]Ethological Context:[/bold cyan]\n{result.reasoning.explanation}\n\n"
                + (f"[bold cyan]Recommendations:[/bold cyan]\n" + "\n".join(f"* {r}" for r in result.reasoning.recommendations) if result.reasoning.recommendations else ""),
                title=f"Layer B Reasoning ({result.reasoning.provider})",
                border_style="cyan",
            ))

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2)
        console.print(f"\n[bold green]Saved structured results to:[/bold green] {out_path}")

    console.print()
