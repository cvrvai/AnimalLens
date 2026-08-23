# AI SDLC & Project Execution Playbook

This document details the **Software Development Life Cycle (SDLC)** for AnimalLens, combining Applied Machine Learning, Operations Research, and Software Engineering. It includes a week-by-week timeline, milestone deliverables, and an actionable task checklist for the project team.

---

## 1. The AnimalLens AI SDLC Framework

Standard software SDLC (Agile/Scrum) fails for Computer Vision because data collection, temporal annotation, and ML model calibration require specialized scientific and data engineering lifecycles.

AnimalLens follows a **7-Stage ML-SDLC (Machine Learning Software Development Life Cycle)**:

```text
+-----------------------------------------------------------------------------------+
|                           AnimalLens 7-Stage ML-SDLC                              |
+-----------------------------------------------------------------------------------+
  Stage 1: Platform & Interface Architecture (Completed - v0.1.0)
     |     * Modular Layer A/B decoupling, Pydantic schemas, Python SDK, FastAPI, CLI
     v
  Stage 2: Standardized Video Data Acquisition (Ethological Protocol)
     |     * Controlled tank camera rigs, fixed illumination, multi-cohort recording
     v
  Stage 3: Annotation & Inter-Rater Reliability Validation (Cohen's Kappa)
     |     * Spatial bounding boxes, temporal action intervals, kappa >= 0.75 threshold
     v
  Stage 4: Perception Model Training & Anti-Leakage Evaluation (Grouped CV)
     |     * YOLOv8 detection, BoT-SORT tracking, session-isolated validation splits
     v
  Stage 5: Temporal Action Recognition & Latency Optimization
     |     * MS-TCN++ / Kinematic rule models, rolling buffer Pareto latency tuning
     v
  Stage 6: Layer B Reasoning & Multimodal Ollama Integration
     |     * Natural language prompt synthesis, vision routing, ethological advice
     v
  Stage 7: Edge Deployment, Active Learning & Model Distribution (Hugging Face)
           * Model registry packaging, uncertainty triage loop, multi-species expansion
```

---

## 2. 12-Week Project Timeline & Milestones

```text
Weeks:    [W1-W2]      [W3-W4]      [W5-W6]      [W7-W8]      [W9-W10]     [W11-W12]
Phases:   Phase 1/2A   Phase 2B     Phase 2C     Phase 3      Phase 3B     Phase 4/5
          Architecture  Data &       YOLOv8 &     Temporal     Ollama &     HuggingFace
          & Rig Setup   Annotation   BoT-SORT     Realtime     Active Learn & Release
```

| Phase | Duration | Focus Area | Key Deliverables |
| :--- | :---: | :--- | :--- |
| **v0.1 — Phase 1** | Completed | Platform Architecture & Core SDK | SDK, Schemas, CLI (`doctor`, `models`), FastAPI REST/WS, 31 tests. |
| **Phase 2A** | Weeks 1–2 | Tank Hardware & Data Collection | Camera rig setup, 50+ hours raw multi-tank video recorded across daylight/infrared. |
| **Phase 2B** | Weeks 3–4 | Annotation & Reliability | 1,000+ labeled frames (detection) + 200 temporal behavior clips with $\kappa \ge 0.75$. |
| **Phase 2C** | Weeks 5–6 | Redclaw Detector & Tracker | Trained `yolov8n-redclaw.pt` (mAP50 > 0.90) + BoT-SORT (MOTA > 0.85). |
| **Phase 3** | Weeks 7–8 | Temporal Behavior & RTSP | Action classification (mating, aggression, feeding) + RTSP rolling buffer < 50ms latency. |
| **Phase 3B** | Weeks 9–10 | Layer B Ollama & Active Learning | Multimodal LLM prompt tuning + automated uncertainty data export. |
| **Phase 4/5** | Weeks 11–12 | Hugging Face & v0.5 Release | Official `animallens/redclaw-behavior-v1` weights on Hugging Face + Pig adapter prototype. |

---

## 3. Actionable Task Checklist for the Founder / Engineer

### Stage 2: Hardware & Video Data Collection (Weeks 1–2)
- [ ] **Camera Setup**: Mount top-down and 45-degree angle 1080p cameras (30 FPS) over observation tanks.
- [ ] **Illumination Control**: Install diffuse LED lighting to prevent surface water glare and reflection artifacts.
- [ ] **Recording Schedule**: Record baseline behavior across 4 daily observation blocks:
  - 08:00–10:00 (Post-feeding activity)
  - 12:00–14:00 (Midday resting/sheltering)
  - 18:00–20:00 (Dusk social interactions)
  - 22:00–00:00 (Night infrared nocturnal activity)
- [ ] **Cohort Management**: Record at least 3 distinct tanks with known male:female sex ratios and varying stocking densities.

### Stage 3: Dataset Annotation & Verification (Weeks 3–4)
- [ ] **Object Detection Labeling**: Annotate bounding boxes for `carapace` and full body in Label Studio / CVAT.
- [ ] **Temporal Behavior Labeling**: Mark start/end timestamps for key ethological events:
  - `reproduction.mating`
  - `aggression.threat_display` / `aggression.fighting`
  - `feeding.foraging`
  - `resting` / `sheltering`
- [ ] **Inter-Rater Validation**: Double-annotate 20% of clips with a second observer and compute Cohen's $\kappa$ to ensure consistency.
- [ ] **Anti-Leakage Partitioning**: Generate `train.json`, `val.json`, and `test.json` using strict session/tank isolation.

### Stage 4: Model Training & Evaluation (Weeks 5–6)
- [ ] **Detector Training**: Fine-tune YOLOv8 nano/small on Redclaw dataset:
  ```bash
  yolo detect train data=redclaw.yaml model=yolov8n.pt epochs=100 imgsz=640
  ```
- [ ] **Tracker Tuning**: Optimize BoT-SORT track buffer, proximity distance threshold, and minimum track length for water turbidity.
- [ ] **Metric Validation**: Confirm $\text{mAP@50} \ge 0.88$ and multi-object tracking accuracy $\text{MOTA} \ge 0.80$.

### Stage 5: Temporal Recognition & Real-Time Engine (Weeks 7–8)
- [ ] **Kinematic Feature Pipeline**: Integrate velocity profiles, inter-individual distance (IID), and approach angles into the temporal classifier.
- [ ] **RTSP Latency Benchmark**: Benchmark live camera ingestion latency on edge devices (target: $< 60\text{ms}$ frame latency).
- [ ] **Rolling Buffer Tuning**: Set optimal buffer window (15 seconds) and temporal stride (5 FPS).

### Stage 6: Layer B LLM & Active Learning Integration (Weeks 9–10)
- [ ] **Ollama Model Evaluation**: Benchmark Gemma 3 vs. Qwen 2.5 vs. LLaMA 3.2 on biological event explanation speed and accuracy.
- [ ] **Active Learning Triage**: Validate that low-confidence predictions ($\text{conf} < 0.45$) automatically export to `datasets/redclaw/uncertainty_queue/`.

### Stage 7: Hugging Face Distribution & Public Launch (Weeks 11–12)
- [ ] **Package Model Weights**: Create official `manifest.json` and upload weights to Hugging Face Hub (`animallens/redclaw-behavior-v1`).
- [ ] **Test CLI Pull**: Verify `animallens pull redclaw-behavior-v1` downloads and loads weights seamlessly.
- [ ] **Publish v0.5 Release**: Tag GitHub release and publish updated documentation.
