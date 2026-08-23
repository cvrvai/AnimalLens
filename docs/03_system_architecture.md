# System Architecture & Layered Intelligence

AnimalLens decouples raw visual perception and kinematics from high-level natural language reasoning.

---

## 1. Two-Layer Intelligence Model

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

---

## 2. Layer A: Computer Vision & Temporal Pipeline
* **BaseDetector**: Extracts bounding boxes $B = [x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}]$ and class probabilities.
* **BaseTracker**: Matches detections across consecutive frames to maintain consistent `track_id` sequences and compute velocity vectors.
* **RollingVideoBuffer**: Retains recent temporal windows ($5 - 20$ seconds) in an efficient ring buffer.
* **BaseBehaviorClassifier**: Analyzes spatial trajectories, inter-individual distance, and temporal state sequences.

---

## 3. Layer B: Pluggable Reasoning Provider
* **OllamaReasoningProvider**: Translates structured `BehaviorEvent` JSON into natural language ethological summaries, distress evaluations, and management advice.
* **Multimodal Capability Detection**: Automatically detects whether an Ollama model supports vision (e.g. `gemma3`, `llama3.2-vision`, `llava`) and routes raw keyframes accordingly.
