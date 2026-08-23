# System Architecture & 4-Stage Multimodal Ethology Intelligence Pipeline

**AnimalLens** is a production-grade, multimodal computer vision and ethological intelligence platform. It solves the critical bottleneck of modern video AI by decoupling **high-throughput edge perception ($>60\text{ FPS}$ | $<15\text{ms}$ latency)** from **asynchronous cognitive multimodal reasoning (Ollama / Gemma 3)**.

---

## 1. Dual-Stream Decoupled Processing Model

```text
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ FAST STREAM: Layer A Edge Vision & Kinematics (>60 FPS | <15ms Latency)     │
 │ Input Frame ──▶ YOLOv8 ──▶ Kalman MOT ──▶ Differential Math ──▶ Ethogram   │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │ (Selective Anomaly Triage)
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ SLOW STREAM: Layer B Cognitive Reasoning (Async | 1-2s Deep Inference)     │
 │ Structured Telemetry ──▶ Ollama (Gemma 3) ──▶ Veterinary Diagnosis ──▶ DB  │
 └─────────────────────────────────────────────────────────────────────────────┘
```

1. **Fast Stream (Layer A)** runs locally on edge CPUs or GPUs with **100% mathematical determinism** and zero API costs.
2. **Slow Stream (Layer B)** is triggered **asynchronously only during anomalies, aggression spikes, or high uncertainty** to generate biological insights.

---

## 2. The 4-Stage SOTA Animal Perception & Behavior Pipeline

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
        |  Stage 1: Spatial Pose &    |                               |  * Gated Uncertainty Triage |
        |           Contour Detection |                               |  * Ollama Client (gemma3,   |
        |  Stage 2: BoT-SORT Kalman   |                               |    llama3.2, qwen2.5)       |
        |           Multi-Object MOT  |                               |  * Biological Summaries     |
        |  Stage 3: Differential      |                               |  * Veterinary Health Alerts |
        |           Kinematics &      |                               +-----------------------------+
        |           Ethogram Engine   |                                              ^
        +-----------------------------+                                              |
                       |                                                             |
                       v                                                             |
        +-----------------------------+                                              |
        |     BehaviorEvent JSON      |----------------------------------------------+
        | (Standard Pydantic Schema)  |
        +-----------------------------+
                       |
                       v
        +-----------------------------+
        |      IPC & REST/WS Server   |
        | (FastAPI / WebSockets :8000)|
        +-----------------------------+
                       |
                       v
        +-----------------------------+
        |   Next.js 14/15 Frontend    |
        |   (Interactive Canvas HUD)  |
        +-----------------------------+
```

---

### Stage 1: Spatial Contour & Keypoint Detection
* **Model**: YOLOv8s / YOLOv11-Pose.
* **Mechanism**: Detects target animals and extracts boundary envelopes and anatomical landmarks (snout, eyes, withers, spine, tail, stifle, paws) in $<8\text{ms}$.
* **Multi-Scale Resolution**: Dynamic inference scaling ($640\times 640$ to $1280\times 1280$) for small or distant subjects in 4K footage.

---

### Stage 2: Persistent Multi-Animal Tracking (MOT)
* **Model**: BoT-SORT with 8-Dimensional Kalman Filter State Estimation:
  $$\mathbf{x} = [x_c, y_c, a, h, v_x, v_y, v_a, v_h]^T$$
* **Spatial Association**: Mahalanobis distance gating combined with Hungarian optimal bipartite matching.
* **Identity Lock**: Enforces persistent identity tags (`DOG-01`, `DOG-02`) across occlusions, camera motion, and crossings.

---

### Stage 3: Differential Kinematics & Ethological Classification
* **Velocity Vectors**:
  $$\vec{v}(t) = \left( \frac{\Delta x}{\Delta t}, \frac{\Delta y}{\Delta t} \right), \quad \text{Speed } \|\vec{v}\| = \sqrt{v_x^2 + v_y^2} \text{ in m/s and km/h}$$
* **Heading Orientation**:
  $$\theta(t) = (\text{atan2}(v_y, v_x) + 360^\circ) \bmod 360^\circ$$
* **Inter-Individual Distance (IID) Matrix**:
  $$\text{IID}(i, j) = \|\mathbf{p}_i(t) - \mathbf{p}_j(t)\|, \quad \text{Approach Rate } \dot{D} = \frac{d(\text{IID})}{dt}$$
* **Canine Ethogram Classifier**: Maps kinematics and aspect ratios to 23 clinical behaviors across 5 categories (`locomotion`, `posture`, `social_behavior`, `aggression`, `maintenance`) using rolling temporal hysteresis majority voting.

---

### Stage 4: Cognitive Multimodal Reasoning & Active Learning Loop
* **Ollama Client**: Integrates local LLMs (`gemma3`, `llama3.2-vision`) to produce natural language veterinary health assessments and anomaly alerts.
* **Active Learning Flywheel**: High-uncertainty interaction frames ($\text{Confidence} < 0.50$) are automatically enqueued into MongoDB time-series collections for human verification and continuous transfer learning.

---

## 3. Web & Edge Application Topology (100% Next.js First)

```
[ Web Browser Client ]
          │
          │ (HTTP POST Video Upload & WS Live Stream)
          ▼
[ Next.js 14/15 App Router (:3000) ]
          │
          │ (Forwarding FormData & WebSocket Proxy)
          ▼
[ AnimalLens Edge Vision Microservice (:8000) ]
          ├── FastAPI High-Performance Asynchronous Server
          ├── Ultralytics YOLOv8 PyTorch Engine
          ├── BoT-SORT Kalman Multi-Animal Tracker
          └── Rolling Ring Video Buffer & Temporal Classifier
```

---

## 4. Architectural Decision Records (ADRs)

| ADR ID | Decision | Strategic Advantage |
| :--- | :--- | :--- |
| **ADR-01** | Decoupled 2-Layer Processing | Guarantees $>60\text{ FPS}$ execution on standard edge CPUs without requiring expensive cloud GPUs. |
| **ADR-02** | Mathematical Kinematics | Ensures zero LLM hallucination for critical measurements (speed, distance, heading). |
| **ADR-03** | Next.js / TypeScript First | Enables frontend and mobile developers to build rich dashboards without touching Python environments. |
| **ADR-04** | MongoDB Uncertainty Queue | Provides a continuous active learning loop for real-world domain adaptation. |
