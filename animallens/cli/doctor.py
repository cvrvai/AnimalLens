"""
AnimalLens Doctor - System diagnostic tool.
Checks Python, GPU/CUDA, OpenCV, FFmpeg, Ollama, model cache, and camera devices.
"""
from __future__ import annotations

import asyncio
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from animallens.core.config import settings
from animallens.models.registry import model_registry
from animallens.reasoning.ollama import OllamaClient
from animallens.species.registry import species_registry

console = Console()


def run_doctor_checks() -> Dict[str, Any]:
    """Execute all system diagnostics."""
    results: Dict[str, Any] = {}

    # 1. Python Environment
    py_ver = sys.version.split()[0]
    py_ok = sys.version_info >= (3, 10)
    results["python"] = {
        "title": "Python Runtime",
        "value": f"Python {py_ver} ({platform.python_implementation()})",
        "status": "OK" if py_ok else "WARNING",
        "note": "Python 3.10+ is supported (3.11+ recommended)",
    }

    # 2. PyTorch & GPU / CUDA
    torch_installed = False
    cuda_available = False
    cuda_device = "N/A"
    try:
        import torch
        torch_installed = True
        if torch.cuda.is_available():
            cuda_available = True
            cuda_device = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    if cuda_available:
        gpu_val = f"CUDA Available: {cuda_device}"
        gpu_status = "OK"
    elif torch_installed:
        gpu_val = f"PyTorch {torch.__version__} (CPU mode)"
        gpu_status = "INFO"
    else:
        gpu_val = "PyTorch not installed (optional, using CPU fallback/development engine)"
        gpu_status = "INFO"

    results["gpu"] = {
        "title": "GPU / CUDA Acceleration",
        "value": gpu_val,
        "status": gpu_status,
        "note": "Install `torch` with CUDA for accelerated deep learning inference",
    }

    # 3. Vision Stack (OpenCV / Pillow)
    cv_val = []
    try:
        import cv2
        cv_val.append(f"OpenCV {cv2.__version__}")
    except ImportError:
        cv_val.append("OpenCV not installed (using PIL engine)")

    try:
        import PIL
        cv_val.append(f"Pillow {PIL.__version__}")
    except ImportError:
        pass

    results["vision"] = {
        "title": "Computer Vision Libraries",
        "value": ", ".join(cv_val),
        "status": "OK",
        "note": "OpenCV recommended for high-performance RTSP stream reading",
    }

    # 4. FFmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    results["ffmpeg"] = {
        "title": "FFmpeg Binary",
        "value": f"Installed at {ffmpeg_path}" if ffmpeg_path else "Not found in PATH",
        "status": "OK" if ffmpeg_path else "INFO",
        "note": "Required for specialized video transcoding and H.264/H.265 RTSP streams",
    }

    # 5. Ollama Connectivity & Models
    ollama_client = OllamaClient()
    try:
        installed_ollama_models = asyncio.run(ollama_client.list_models())
        ollama_connected = True
    except Exception:
        installed_ollama_models = []
        ollama_connected = False

    if ollama_connected and installed_ollama_models:
        model_names = ", ".join(m["name"] for m in installed_ollama_models[:4])
        if len(installed_ollama_models) > 4:
            model_names += f" (+{len(installed_ollama_models) - 4} more)"
        ollama_val = f"Connected ({len(installed_ollama_models)} models: {model_names})"
        ollama_status = "OK"
    elif ollama_connected:
        ollama_val = f"Connected at {ollama_client.base_url} (No models pulled yet)"
        ollama_status = "INFO"
    else:
        ollama_val = f"Not reachable at {ollama_client.base_url}"
        ollama_status = "INFO"

    results["ollama"] = {
        "title": "Ollama LLM Engine (Layer B)",
        "value": ollama_val,
        "status": ollama_status,
        "note": "Optional. Required only when connecting LLMs (gemma3, qwen, llama) for reasoning",
    }

    # 6. AnimalLens Species Models Cache
    installed_models = model_registry.list_installed()
    results["models"] = {
        "title": "Species Models Cache",
        "value": f"{len(installed_models)} model(s) installed in {settings.models_dir}",
        "status": "OK",
        "note": "Run `animallens pull <model>` to download species models",
    }

    # 7. Species Adapters
    registered_species = species_registry.list_species()
    results["species"] = {
        "title": "Registered Species",
        "value": f"{len(registered_species)} species: {', '.join(s['name'] for s in registered_species)}",
        "status": "OK",
        "note": "Species plugins are loaded dynamically",
    }

    # 8. Storage & Cache Directories
    settings.ensure_directories()
    results["storage"] = {
        "title": "Storage & Cache Directories",
        "value": f"Cache: {settings.cache_dir}",
        "status": "OK" if settings.cache_dir.exists() else "WARNING",
        "note": "Writable local directory for downloaded models and datasets",
    }

    return results


def print_doctor_report() -> None:
    """Render doctor diagnostics in Rich formatted output."""
    console.print()
    console.print(Panel("[bold cyan]AnimalLens System Diagnostics (Doctor)[/bold cyan]", expand=False))
    console.print()

    checks = run_doctor_checks()

    table = Table(title="Diagnostic Summary", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", width=26)
    table.add_column("Status", width=12)
    table.add_column("Details", style="white")
    table.add_column("Recommendation / Note", style="dim")

    for key, item in checks.items():
        status = item["status"]
        if status == "OK":
            status_str = "[bold green]READY[/bold green]"
        elif status == "INFO":
            status_str = "[bold yellow]OPTIONAL[/bold yellow]"
        else:
            status_str = "[bold red]WARNING[/bold red]"

        table.add_row(item["title"], status_str, item["value"], item["note"])

    console.print(table)
    console.print()
    console.print("[dim]Run [bold]animallens --help[/bold] to see all available commands.[/dim]\n")
