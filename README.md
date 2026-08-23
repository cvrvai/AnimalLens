<div align="center">

# 🐕 AnimalLens
### Open-Source Canine Behavior & Ethology Intelligence Library

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg)](https://opencv.org/)

**AnimalLens** is an open-source Python library for real-time animal computer vision, multi-object tracking, kinematics analysis, and ethological behavior classification.

Built for veterinary researchers, smart animal shelters, and AI developers. Features out-of-the-box **Domestic Dog (*Canis lupus familiaris*)** posture, locomotion, and welfare tracking.

</div>

---

## 📦 Installation

Install directly from GitHub via `pip`:

```bash
pip install git+https://github.com/cvrvai/AnimalLens.git
```

*Requirements: Python 3.10+, PyTorch, Ultralytics, OpenCV, NumPy, Pydantic.*

---

## 🚀 Quickstart

Run behavioral analysis on any dog video in 3 lines of Python:

```python
from animallens import AnimalLens

# 1. Initialize AnimalLens library
lens = AnimalLens(species="dog")

# 2. Analyze recorded video (automatically runs YOLOv8 + BoT-SORT Kalman tracking)
result = lens.analyze("path/to/dog_video.mp4")

# 3. Print human-readable ethogram timeline
print(result.format_timeline_text())
```

### Output:
```text
00:00:00 Posture.standing (conf: 0.88)
00:00:01 Locomotion.walk (conf: 0.89)
00:00:02 Locomotion.running_gallop (conf: 0.96)
00:00:04 Social_behavior.play_bow (conf: 0.93)
```

---

## 📖 Developer Guide & Python API

### 1. Analyzing Recorded Videos & Extracting Kinematics

```python
from animallens import AnimalLens

lens = AnimalLens(species="dog")
result = lens.analyze_video("dog_running.mp4", sample_fps=10.0)

print(f"Total Frames Analyzed: {result.total_frames_analyzed}")
print(f"Duration: {result.duration_seconds:.2f}s")

# Iterate over structured behavior events
for event in result.behaviors:
    print(f"[{event.temporal.start:.1f}s - {event.temporal.end:.1f}s] {event.behavior.label}")
    print(f"  Category:   {event.behavior.category}")
    print(f"  Confidence: {event.behavior.confidence:.1%}")
    print(f"  Track IDs:  {[s.track_id for s in event.subjects]}")
```

---

### 2. Real-Time Camera & RTSP Stream Tracking

Process live 60+ FPS webcam or RTSP camera feeds with zero lag:

```python
from animallens import AnimalLens

lens = AnimalLens(species="dog")

# Stream live from local USB webcam (0) or RTSP stream URL
for event in lens.stream(0, target_fps=30.0):
    print(f"[{event.temporal.start:.2f}s] Active Behavior: {event.behavior.label}")
```

---

### 3. Single Image Detection & Posture Analysis

```python
from PIL import Image
from animallens import AnimalLens

lens = AnimalLens(species="dog")
img = Image.open("dog_photo.jpg")

result = lens.analyze_image(img)

# Access detected bounding boxes and postures
for event in result.behaviors:
    for subject in event.subjects:
        print(f"Dog Track #{subject.track_id}: Bounding Box = {subject.bbox}")
```

---

### 4. Low-Level Tracking & Kinematics Engine

For developers building custom computer vision pipelines, use `AnimalTracker` and `BehavioralClassifier` directly:

```python
from animallens.tracking.tracker import AnimalTracker
from animallens.behavior.classifier import BehavioralClassifier
from animallens.perception.models.yolov8_detector import YOLOv8Detector
from PIL import Image

detector = YOLOv8Detector()
tracker = AnimalTracker(species_prefix="DOG", pixel_to_meter_ratio=2.5)
classifier = BehavioralClassifier()

# 1. Detect bounding boxes
img = Image.open("frame.jpg")
detections = detector.detect(img)

# 2. Update Kalman tracker & calculate velocities
telemetry = tracker.update_frame(detections, timestamp=0.033, dt=0.033)

for subject in telemetry.subjects:
    print(f"Subject: {subject.display_id}")
    print(f"  Velocity:      {subject.velocity_mps:.2f} m/s ({subject.velocity_mps * 3.6:.1f} km/h)")
    print(f"  Heading:       {subject.heading_degrees:.1f}°")
    print(f"  Acceleration:  {subject.acceleration_mps2:.2f} m/s²")

    # 3. Classify behavior from kinematics
    behavior = classifier.classify_subject(subject, frame_telemetry=telemetry)
    print(f"  Behavior:      {behavior.human_readable} (Welfare Score: {behavior.welfare_score}/100)")
```

---

### 5. Multimodal LLM Reasoning with Ollama

Connect Layer B reasoning to any local Ollama model (`gemma3`, `llama3.2`, `qwen2.5`) for biological explanations:

```python
from animallens import AnimalLens

# Enable Ollama Layer B reasoning
lens = AnimalLens(species="dog", reasoning="ollama:gemma3")
lens.analyze_video("dog_training.mp4")

# Ask natural language questions about observed behaviors
explanation = lens.ask("Did the dog display any stress, fatigue, or lameness during this run?")
print(explanation)
```

---

### 6. Embedding as a FastAPI Microservice

AnimalLens provides a production-ready FastAPI application:

```python
# main.py
from animallens.server.app import app

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

#### Available Endpoints:
* `POST /v1/analyze/video`: Multipart video analysis returning structured JSON telemetry.
* `POST /v1/analyze/image`: Single-frame image detection.
* `GET /v1/health`: Returns API status and GPU/CUDA device info.
* `WS /v1/events`: Real-time WebSocket streaming feed.

Or start the server via CLI:
```bash
animallens serve --port 8000 --host 0.0.0.0 --device cpu
```

---

## 🔬 Supported Canine Ethogram Categories

| Category | Behaviors / Actions | Kinematic Criteria |
| :--- | :--- | :--- |
| **Locomotion** | `running_gallop`, `trot`, `walk` | Velocity $> 3.5\text{ m/s}$ (Gallop), $1.2-3.5\text{ m/s}$ (Trot), $0.3-1.2\text{ m/s}$ (Walk) |
| **Posture** | `standing`, `sitting`, `lying_sternal`, `sleeping` | Aspect ratio analysis & stationary velocity $< 0.3\text{ m/s}$ |
| **Social Behavior** | `play_bow`, `following`, `sniffing_conspecific`, `greeting` | Inter-Individual Distance ($\text{IID} < 0.6\text{m}$) & posture alignment |
| **Agonistic / Defense** | `aggressive_lunge`, `defensive_retreat`, `growling_stance` | High negative approach rate & rapid acceleration spikes |

---

## 🏛️ System Architecture

```text
                                +-------------------------------------------+
                                |               AnimalLens SDK              |
                                |     from animallens import AnimalLens     |
                                +-------------------------------------------+
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
        |  (100% LLM-Independent)     |                               |  (Ollama / Local LLM)       |
        +-----------------------------+                               +-----------------------------+
        |  1. YOLOv8 Object Detection |                               |  * Ollama Client (gemma3,   |
        |  2. BoT-SORT Kalman MOT     |                               |    llama3.2, qwen2.5)       |
        |  3. Kinematics Engine       |                               |  * Biological Summaries     |
        |  4. Ethogram Classification |                               |  * Veterinary Q&A           |
        +-----------------------------+                               +-----------------------------+
                       |                                                             ^
                       v                                                             |
        +-----------------------------+                                              |
        |     BehaviorEvent JSON      |----------------------------------------------+
        | (Standard Pydantic Schema)  |
        +-----------------------------+
```

---

## 📄 Standard Output Schema

Every analyzed event produces a typed Pydantic JSON structure:

```json
{
  "schema_version": "1.0",
  "event_id": "evt_8a92f1b0",
  "timestamp": 1787474800.0,
  "species": {
    "id": "canis_lupus_familiaris",
    "name": "Domestic Dog",
    "scientific_name": "Canis lupus familiaris"
  },
  "subjects": [
    {
      "track_id": 1,
      "display_id": "DOG-01",
      "velocity_mps": 14.0,
      "velocity_kmh": 50.4,
      "heading_degrees": 84.5,
      "bbox": {
        "x_min": 0.35,
        "y_min": 0.42,
        "x_max": 0.58,
        "y_max": 0.76
      }
    }
  ],
  "behavior": {
    "category": "locomotion",
    "label": "running_gallop",
    "human_readable": "High-Speed Gallop Sprint",
    "confidence": 0.96
  },
  "temporal": {
    "start": 2.4,
    "end": 8.1,
    "duration": 5.7
  }
}
```

---

## 🤝 Contributing & License

Contributions, bug reports, and new species adapters are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

Licensed under the **Apache License, Version 2.0**.
