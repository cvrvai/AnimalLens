<div align="center">

# AnimalLens
### Open Animal Behavior Intelligence Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

**AnimalLens** is an open-source animal behavior intelligence platform for images, recorded video, and real-time camera streams, with optional local LLM reasoning through Ollama.

> **AnimalLens is an open-source framework for building and deploying animal behavior intelligence. Redclaw crayfish (*Cherax quadricarinatus*) is the first species implementation.**

</div>

---

## 60-Second Quick Start

### Installation

```bash
pip install animallens
```

### 3-Line Behavior Analysis

```python
from animallens import AnimalLens

lens = AnimalLens(species="redclaw", reasoning="ollama:gemma3")
result = lens.analyze("tank.mp4")

print(result.format_timeline_text())
```

Output:
```text
00:05:22 Feeding (conf: 0.91)
00:14:01 Aggression (conf: 0.86)
00:21:32 Social interaction (conf: 0.78)
00:21:48 Mating (conf: 0.92)
```

---

## System Architecture

AnimalLens strictly separates **Layer A (Vision & Temporal Intelligence)** from **Layer B (Optional LLM Reasoning)**.

```text
                                +-------------------------------------------+
                                |               AnimalLens SDK              |
                                |     from animallens import AnimalLens     |
                                +-------------------------------------------+
                                                      |
                  +-----------------------------------+-----------------------------------+
                  |                                   |                                   |
           [ Python SDK ]                      [ CLI (Typer) ]                     [ REST & WS API ]
        ai = AnimalLens(...)                animallens analyze ...                 POST /v1/analyze
        ai.analyze_video(...)                animallens serve ...                   WS /v1/events
                  |                                   |                                   |
                  +-----------------------------------+-----------------------------------+
                                                      |
                                                      v
                                        +----------------------------+
                                        |    AnimalLens Core Engine  |
                                        +----------------------------+
                                                      |
                       +------------------------------+------------------------------+
                       |                                                             |
                       v                                                             v
        +-----------------------------+                               +-----------------------------+
        |  Layer A: Vision Perception |                               |  Layer B: Reasoning (Opt)   |
        |  (100% LLM-Independent)     |                               |  (Ollama / Any Model)       |
        +-----------------------------+                               +-----------------------------+
        |  1. Source Ingestion        |                               |  * Ollama Client (gemma3,   |
        |     (Image/Video/RTSP)      |                               |    qwen, llama, etc.)       |
        |  2. Object Detection        |                               |  * Text & Vision Routing    |
        |  3. Multi-Object Tracking   |                               |  * Explanations & Summaries |
        |  4. Pose / Movement Engine  |                               |  * Biological Insights      |
        |  5. Temporal Classification |                               +-----------------------------+
        +-----------------------------+                                              ^
                       |                                                             |
                       v                                                             |
        +-----------------------------+                                              |
        |     BehaviorEvent JSON      |----------------------------------------------+
        | (Standard Pydantic Schema)  |
        +-----------------------------+
```

### Core Principles
1. **Layer A Works 100% Offline without LLMs**: Fast computer vision models detect animals, track persistent IDs, and classify temporal behaviors into structured JSON.
2. **Layer B Connects to Any Ollama Model**: Send structured behavior events to any local or remote Ollama model (`gemma3`, `qwen2.5`, `llama3.2`, `mistral`, etc.) for natural language summaries, ethological explanations, and recommendations.
3. **Pluggable Species Adapters**: Species-specific taxonomy and feature extraction live in decoupled adapters (`species/redclaw`, `species/pig`). Adding a new species never requires modifying the core engine.
4. **Never Force False Positives**: Ambiguous or low-confidence interactions are tagged as `unknown` with uncertainty flags for active learning and expert human review.

---

## Usage Examples

### 1. Single Image Analysis

```python
from animallens import AnimalLens

lens = AnimalLens(species="redclaw", reasoning=None)
result = lens.analyze_image("specimen.jpg")

for event in result.behaviors:
    print(f"[{event.behavior.category}] {event.behavior.label} (Conf: {event.behavior.confidence:.2f})")
    print(f"Tracked Subjects: {len(event.subjects)}")
```

### 2. Recorded Video & Timeline Generation

```python
from animallens import AnimalLens

lens = AnimalLens(species="redclaw", reasoning="ollama:gemma3")
result = lens.analyze_video("tank_recording.mp4", sample_fps=5.0)

print(result.format_timeline_text())

# Access structured Pydantic schema
if result.behaviors:
    first_event = result.behaviors[0]
    print(first_event.model_dump_json(indent=2))
```

### 3. Real-Time RTSP Stream & Webcams

```python
from animallens import AnimalLens

lens = AnimalLens(species="redclaw")

# Connect to live camera or USB index (0)
for event in lens.stream("rtsp://admin:pass@192.168.1.100:554/live"):
    print(f"[{event.temporal.start:05.1f}s] Detected: {event.behavior.category}.{event.behavior.label}")
```

### 4. Conversational Behavior Q&A

```python
from animallens import AnimalLens

lens = AnimalLens(species="redclaw", reasoning="ollama:gemma3")
lens.analyze_video("tank.mp4")

answer = lens.ask("Were there any aggressive encounters or signs of stress during this session?")
print(answer)
```

---

## Standard Behavior Event Schema

Every detected behavior produces a strongly-typed schema:

```json
{
  "schema_version": "1.0",
  "event_id": "evt_01928",
  "timestamp": 1724400000.0,
  "species": {
    "id": "cherax_quadricarinatus",
    "name": "Redclaw Crayfish",
    "scientific_name": "Cherax quadricarinatus",
    "taxonomy_version": "1.0.0"
  },
  "source": {
    "type": "camera",
    "camera_id": "CAM-001"
  },
  "subjects": [
    {
      "track_id": 17,
      "animal_id": "F-003",
      "velocity": 0.04
    },
    {
      "track_id": 23,
      "animal_id": "M-002",
      "velocity": 0.05
    }
  ],
  "behavior": {
    "category": "reproduction",
    "label": "mating",
    "confidence": 0.93,
    "is_uncertain": false
  },
  "temporal": {
    "start": 42.1,
    "end": 74.4,
    "duration": 32.3
  },
  "model": {
    "species_model": "redclaw-behavior-v1",
    "version": "1.0.0",
    "detector": "yolov8-redclaw-v1",
    "classifier": "redclaw-temporal-v1"
  },
  "reasoning": {
    "provider": "ollama:gemma3",
    "model": "gemma3",
    "summary": "Observed copulatory interaction between two mature Cherax quadricarinatus.",
    "recommendations": [
      "Maintain stable water temperature at 26-28C",
      "Ensure sufficient shelter tiles to prevent post-copulatory aggression"
    ]
  }
}
```

---

## CLI Reference

AnimalLens includes a full-featured CLI:

```bash
# Run system diagnostics (Python, GPU, CUDA, OpenCV, FFmpeg, Ollama)
animallens doctor

# List registered species
animallens species list

# List & manage species behavior models
animallens models
animallens pull redclaw-behavior-v1
animallens remove redclaw-behavior-v1

# Discover installed Ollama LLMs
animallens ollama list

# Analyze media files
animallens analyze specimen.jpg --species redclaw
animallens analyze tank.mp4 --species redclaw --reasoning "ollama:gemma3" --format timeline

# Start REST & WebSocket server
animallens serve --host 0.0.0.0 --port 8088
```

---

## REST & WebSocket API

Start the server:
```bash
animallens serve
```

Interactive OpenAPI documentation is available at `http://localhost:8088/docs`.

### API Endpoints
| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/v1/health` | Service health status |
| `GET` | `/v1/species` | List supported species & taxonomies |
| `GET` | `/v1/models` | List installed and catalog models |
| `POST`| `/v1/models/pull` | Download model weights to cache |
| `GET` | `/v1/ollama/models` | List local Ollama models |
| `POST`| `/v1/analyze/image` | Multipart image analysis |
| `POST`| `/v1/analyze/video` | Multipart video analysis & timeline |
| `WS`  | `/v1/events` | Real-time WebSocket behavior feed |
| `GET` | `/v1/events/sse` | Server-Sent Events behavior stream |

---

## Docker Deployment

Run with Docker Compose:

```bash
docker compose up -d
```

The API will be available at `http://localhost:8088`.

---

## Extending AnimalLens (Adding a New Species)

Adding a new species (e.g. `pig`, `chicken`, `fish`) requires **zero changes** to AnimalLens Core. Simply create a new folder under `animallens/species/<species_id>/`:

```text
animallens/species/pig/
├── __init__.py
├── config.yaml
├── taxonomy.yaml
└── adapter.py
```

Then register it:
```python
from animallens.species.registry import species_registry
from animallens.species.pig.adapter import PigAdapter

species_registry.register_adapter("pig", PigAdapter())
```

---

## Documentation Index

Detailed technical, mathematical, and execution documentation is available in the [`docs/`](docs/) directory:

1. [Mathematical Foundations & Operations Research](docs/01_mathematical_foundations.md): Multi-agent spatial graphs, Markov state transitions, and Pareto buffer optimization.
2. [Scientific Research Methodology](docs/02_scientific_research_methodology.md): Quantitative ethology sampling protocols (*Altmann 1974*), anti-leakage grouped partitioning, and Cohen's Kappa.
3. [System Architecture](docs/03_system_architecture.md): Modular Layer A vision perception and Layer B reasoning provider interfaces.
4. [Developer Guide & API Reference](docs/04_developer_guide.md): Python SDK, REST & WebSocket API, and custom species tutorial.
5. [AI SDLC & Project Execution Playbook](docs/05_sdlc_and_execution_playbook.md): 7-Stage ML-SDLC framework, 12-week timeline, and founder task checklist.

---

## Roadmap & Release Milestones

- [x] **v0.1 — Phase 1** (Current): Platform architecture, Layer A/B decoupling, Redclaw species adapter, Developer Python SDK, Typer CLI (`doctor`, `models`, `analyze`), FastAPI REST/WS server, and rolling video buffer.
- [ ] **v0.2**: Real Redclaw detector & tracker (YOLOv8 + BoT-SORT fine-tuned on aquaculture datasets).
- [ ] **v0.3**: Temporal behavior recognition & action segmentation.
- [ ] **v0.4**: Realtime camera behavior intelligence with low-latency event buffering.
- [ ] **v0.5**: Redclaw breeding & reproduction behavior models (mating, courtship, egg-bearing).
- [ ] **v0.6**: Automated Hugging Face Hub model distribution & caching.
- [ ] **v0.7**: Multi-species expansion (`pig-behavior-v1`).
- [ ] **v1.0**: Stable multi-species AnimalLens API & production benchmarks.

---

## Contributing & License

Contributions are welcome! Please check [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions.

Licensed under the **Apache License, Version 2.0**.
