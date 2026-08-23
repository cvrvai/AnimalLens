"""
Main CLI entrypoint for AnimalLens.
Built with Typer and Rich.
"""
from __future__ import annotations

import sys
from typing import Optional

# Reconfigure windows terminal encoding if possible
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
import typer
from animallens.cli.analyze_cmd import analyze_source_cli
from animallens.cli.doctor import print_doctor_report
from animallens.cli.models_cmd import models_app, ollama_app, species_app
from animallens.core.config import settings

console = Console()
app = typer.Typer(
    name="animallens",
    help="AnimalLens — Open Animal Behavior Intelligence Platform",
    add_completion=False,
    no_args_is_help=True,
)

# Register Sub-Apps
app.add_typer(models_app, name="models")
app.add_typer(ollama_app, name="ollama")
app.add_typer(species_app, name="species")

# Add Root Level Direct Aliases for Developer Friendliness
app.command(name="analyze")(analyze_source_cli)


@app.command(name="doctor")
def doctor_cmd() -> None:
    """Run system diagnostics on Python, GPU, OpenCV, FFmpeg, and Ollama."""
    print_doctor_report()


@app.command(name="pull")
def pull_alias(
    model_name: str = typer.Argument(..., help="Model name, e.g. redclaw-behavior-v1"),
) -> None:
    """Download and register a species behavior model (alias for 'animallens models pull')."""
    from animallens.models.registry import model_registry

    console.print(f"[bold cyan]Pulling species model:[/bold cyan] {model_name}...")
    try:
        path = model_registry.pull(
            model_name,
            progress_callback=lambda msg, frac: console.print(f"  [dim]*[/dim] {msg}"),
        )
        console.print(f"\n[bold green]Successfully installed model '{model_name}' to:[/bold green] {path}\n")
    except Exception as e:
        console.print(f"\n[bold red]Failed to pull model:[/bold red] {e}\n")
        raise typer.Exit(code=1)


@app.command(name="remove")
def remove_alias(
    model_name: str = typer.Argument(..., help="Model name to remove"),
) -> None:
    """Remove a species behavior model from local cache (alias for 'animallens models remove')."""
    from animallens.models.registry import model_registry

    try:
        model_registry.remove(model_name)
        console.print(f"\n[bold green]Removed model '{model_name}'.[/bold green]\n")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}\n")
        raise typer.Exit(code=1)


@app.command(name="serve")
def serve_cmd(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host address to bind"),
    port: int = typer.Option(8088, "--port", "-p", help="Port to listen on"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
) -> None:
    """Start local AnimalLens REST and WebSocket API server."""
    console.print(Panel(
        f"[bold green]AnimalLens API Server starting on http://{host}:{port}[/bold green]\n\n"
        f"  * Interactive OpenAPI Docs: [cyan]http://localhost:{port}/docs[/cyan]\n"
        f"  * WebSocket Realtime Feed:  [cyan]ws://localhost:{port}/v1/events[/cyan]\n"
        f"  * Health Endpoint:          [cyan]http://localhost:{port}/v1/health[/cyan]",
        title="AnimalLens Server",
        border_style="green",
    ))
    try:
        import uvicorn
        uvicorn.run("animallens.server.app:app", host=host, port=port, reload=reload)
    except KeyboardInterrupt:
        console.print("\n[dim]Server stopped.[/dim]")
    except Exception as e:
        console.print(f"[bold red]Server startup error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command(name="version")
def version_cmd() -> None:
    """Print the installed AnimalLens version."""
    console.print(f"AnimalLens version: [bold cyan]{settings.app_version}[/bold cyan]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
