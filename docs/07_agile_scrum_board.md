# AnimalLens Agile Scrum Board & Sprint Delivery Hub

**Project**: AnimalLens — Open Animal Behavior Intelligence Platform  
**Delivery Method**: 2-Week Agile Sprints (MLOps & Applied AI Scrum)  
**Sprint Cycle**: Sprint 1 (Active) to Sprint 6 (Public Release)

---

## Active Sprint 1 — Hardware Rig, Video Collection & MongoDB Ingestion

**Sprint Goal**: Establish multi-angle 1080p camera recording rigs across 3 experimental tanks, record initial 50 hours of raw circadian video (day/night), and deploy MongoDB time-series ingestion pipeline.

---

### Sprint 1 Scrum Execution Board

#### Backlog & In Refinement

| User Story / Task | Assignee | Priority | Points | Target Date | Epic | Acceptance Criteria |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Set up Label Studio / CVAT Annotation Workspace** | Choonvai | `P0 Must` | 3 | Aug 28, 2026 | `Data Eng` | Deploy CVAT/LabelStudio with Redclaw ethogram classes configured. |
| **Draft Pre-Annotation Protocol for Crawling & Grappling** | Ethologist | `P1 Should` | 2 | Aug 29, 2026 | `Research` | Document visual bounding criteria for overlapping crayfishes. |
| **Design MongoDB Aggregation Pipeline for Circadian Budgets** | AI Lead | `P1 Should` | 3 | Aug 30, 2026 | `MongoDB` | Write aggregation query returning 24-hour activity percentages. |

---

#### To Do

| User Story / Task | Assignee | Priority | Points | Target Date | Epic | Acceptance Criteria |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mount Top-Down & 45-Degree 1080p Cameras on Tanks** | Choonvai | `P0 Must` | 5 | Aug 25, 2026 | `Hardware` | 3 tanks equipped with stable top-down and 45-degree angle cameras. |
| **Install Anti-Glare Diffuse Lighting & IR Illuminators** | Choonvai | `P0 Must` | 3 | Aug 26, 2026 | `Hardware` | Zero surface water glare artifacts in daylight & 100% IR visibility at night. |
| **Deploy Local MongoDB Database & Time-Series Collections** | AI Lead | `P0 Must` | 5 | Aug 26, 2026 | `MongoDB` | Local MongoDB instance running with `events` and `sessions` indexes. |

---

#### In Progress

| User Story / Task | Assignee | Priority | Points | Target Date | Epic | Acceptance Criteria |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Execute 50-Hour Raw Video Recording Across 4 Circadian Blocks** | Choonvai | `P0 Must` | 8 | Aug 27, 2026 | `Data Eng` | 50h 1080p video saved with session metadata (temperature, DO, pH, cohort). |
| **Build MongoDB Ingestion Adapter in Python SDK** | AI Lead | `P1 Should` | 5 | Aug 28, 2026 | `MongoDB` | `AnimalLens.analyze_video()` auto-persists `BehaviorEvent` to MongoDB. |

---

#### Testing & QA Validation

| User Story / Task | Assignee | Priority | Points | Target Date | Epic | Acceptance Criteria |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Verify Video Ingestion Frame Rate & Integrity** | QA / AI | `P0 Must` | 2 | Aug 28, 2026 | `QA` | No dropped frames across 10-hour continuous recorded segments. |
| **Validate MongoDB Schema BSON Index Performance** | AI Lead | `P1 Should` | 2 | Aug 29, 2026 | `QA` | Query response time < 15ms on 50,000 synthetic event records. |

---

#### Done (Phase 1 Foundation)

| User Story / Task | Assignee | Priority | Points | Completed | Epic |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Core Architecture & Pydantic Event Schemas** | AI Lead | `P0 Must` | 8 | Aug 23, 2026 | `Platform` |
| **Decoupled Layer A Vision & Layer B Ollama Reasoning** | AI Lead | `P0 Must` | 5 | Aug 23, 2026 | `Platform` |
| **Redclaw Crayfish Taxonomy Adapter (`species/redclaw/`)** | Ethologist | `P0 Must` | 5 | Aug 23, 2026 | `Species` |
| **Operations Research Analytics (Markov & Clark-Evans)** | AI Lead | `P1 Should` | 5 | Aug 23, 2026 | `Analytics` |
| **Python SDK, Typer CLI & FastAPI REST/WebSocket Server** | AI Lead | `P0 Must` | 8 | Aug 23, 2026 | `Platform` |
| **31 Passing Automated Unit Tests Pushed to GitHub** | AI Lead | `P0 Must` | 3 | Aug 23, 2026 | `QA` |

---

## 2. Product Backlog & Upcoming Sprints

### Sprint 2: Annotation, Anti-Leakage & Kappa Reliability (Weeks 3–4)
* **Goal**: Annotate 1,000 detection frames and 200 behavioral clips with inter-rater Cohen's Kappa $\ge 0.75$.
* **User Stories**:
  - `US-201`: Label carapace & full-body bounding boxes for 1,000 frames (`8 pts | Choonvai / Annotator`).
  - `US-202`: Mark temporal boundaries for mating, fighting, foraging, and resting (`5 pts | Annotator`).
  - `US-203`: Compute inter-annotator Cohen's Kappa score on 20% double-annotated subset (`3 pts | Ethologist`).
  - `US-204`: Create session-isolated train/val/test splits to guarantee zero temporal data leakage (`5 pts | AI Lead`).

### Sprint 3: YOLOv8 Perception & BoT-SORT Tracking (Weeks 5–6)
* **Goal**: Train fine-tuned Redclaw detector ($\text{mAP@50} \ge 0.88$) and multi-object tracker ($\text{MOTA} \ge 0.80$).
* **User Stories**:
  - `US-301`: Train YOLOv8n with CIoU loss and multi-scale augmentations (`8 pts | AI Lead`).
  - `US-302`: Tune BoT-SORT / ByteTrack Kalman filters for underwater multi-animal tracking (`5 pts | AI Lead`).
  - `US-303`: Benchmark detection FPS across edge CPU/GPU targets (`3 pts | AI Lead`).

### Sprint 4: Temporal Action Engine & Real-Time RTSP (Weeks 7–8)
* **Goal**: Build temporal action classifier and achieve $< 60\text{ms}$ live RTSP camera latency.
* **User Stories**:
  - `US-401`: Implement kinematic feature extraction (velocity differentials, IID vectors) (`5 pts | AI Lead`).
  - `US-402`: Train temporal action segmentation model with smoothing loss (`8 pts | AI Lead`).
  - `US-403`: Optimize `RollingVideoBuffer` ring buffer for low-latency live streaming (`5 pts | AI Lead`).

### Sprint 5: Ollama Reasoning & Active Learning Loop (Weeks 9–10)
* **Goal**: Connect local LLM reasoning and automated uncertainty data collection in MongoDB.
* **User Stories**:
  - `US-501`: Benchmark Ollama Gemma 3 vs. Qwen 2.5 on ethological explanation generation (`5 pts | AI Lead`).
  - `US-502`: Build automated uncertainty triage routing low-confidence events to MongoDB queue (`5 pts | AI Lead`).

### Sprint 6: Hugging Face Distribution & v0.5 Release (Weeks 11–12)
* **Goal**: Package official model weights on Hugging Face Hub and launch public v0.5 release.
* **User Stories**:
  - `US-601`: Package and upload `animallens/redclaw-behavior-v1` weights on Hugging Face (`5 pts | AI Lead`).
  - `US-602`: Verify `animallens pull redclaw-behavior-v1` one-command download (`3 pts | QA`).
  - `US-603`: Publish v0.5 release notes, documentation, and live demo video (`5 pts | Choonvai`).

---

## 3. Agile Ceremonies & Cadence

* **Sprint Planning**: Every second Monday at 09:00 (Commit to Story Points and Sprint Goal).
* **Daily Async Standup**:
  1. What did I accomplish yesterday?
  2. What will I work on today?
  3. Are there any blockers / hardware / dataset issues?
* **Sprint Review & Demo**: Every second Friday at 16:00 (Live demo of newly passing tests and perception accuracy).
* **Sprint Retrospective**: Every second Friday at 17:00 (Continuous process refinement).
