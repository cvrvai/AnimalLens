# Founder Execution Gantt Chart & Kanban Roadmap

This document contains the visual Gantt timeline, sprint breakdown, and categorized Kanban task boards for engineering AnimalLens from Phase 1 through Phase 5.

---

## 1. Visual 12-Week Gantt Chart

```text
=========================================================================================================
Phase / Task Track              | W1  W2  | W3  W4  | W5  W6  | W7  W8  | W9  W10 | W11 W12 | Status
=========================================================================================================
1. Core Architecture & SDK      | [DONE]  |         |         |         |         |         | COMPLETED
2. Tank Rigs & 50h Video Record | [======]|         |         |         |         |         | IN PROGRESS
3. Annotation & Kappa Validate  |         | [======]|         |         |         |         | READY
4. YOLOv8 Training & BoT-SORT   |         |         | [======]|         |         |         | PLANNED
5. Temporal Engine & RTSP       |         |         |         | [======]|         |         | PLANNED
6. MongoDB Storage & Analytics  |         |         |   [====]|         |         |         | PLANNED
7. Ollama Reasoning Layer B     |         |         |         |         | [======]|         | PLANNED
8. Active Learning Loop         |         |         |         |         | [======]|         | PLANNED
9. Hugging Face Model Deploy    |         |         |         |         |         | [======]| PLANNED
10. v0.5 Public Release Launch  |         |         |         |         |         |   [====]| PLANNED
=========================================================================================================
```

---

## 2. Interactive Kanban Task Board

### Column A: In Progress (Sprint 1: Weeks 1–2)
| Task ID | Task Description | Priority | Category | Assignee |
| :--- | :--- | :---: | :---: | :---: |
| **TSK-01** | Mount top-down & 45-degree 1080p cameras over 3 experimental tanks | `P0` | Hardware | Founder |
| **TSK-02** | Install anti-glare diffuse LED lighting & infrared night illuminators | `P1` | Hardware | Founder |
| **TSK-03** | Record initial 50 hours of raw video across 4 circadian blocks (Day/Night) | `P0` | Data Eng | Founder |
| **TSK-04** | Deploy local MongoDB instance with `events` and `sessions` collections | `P1` | Database | AI Team |

### Column B: Backlog (Sprint 2: Weeks 3–4 — Annotation & Reliability)
| Task ID | Task Description | Priority | Category | Assignee |
| :--- | :--- | :---: | :---: | :---: |
| **TSK-05** | Set up Label Studio / CVAT workspace with Redclaw ethogram classes | `P0` | Data Eng | Founder |
| **TSK-06** | Annotate 1,000 spatial bounding boxes for `carapace` and full body | `P0` | Data Eng | Annotator |
| **TSK-07** | Double-annotate 20% of clips & compute inter-rater Cohen's Kappa ($\ge 0.75$) | `P1` | Research | Ethologist |
| **TSK-08** | Generate anti-leakage grouped train/val/test splits (session/tank isolated) | `P0` | ML / DL | AI Team |

### Column C: Planned (Sprint 3: Weeks 5–6 — Perception & Tracking)
| Task ID | Task Description | Priority | Category | Assignee |
| :--- | :--- | :---: | :---: | :---: |
| **TSK-09** | Fine-tune YOLOv8 nano/small detector with CIoU loss ($\text{mAP@50} \ge 0.88$) | `P0` | ML / DL | AI Team |
| **TSK-10** | Tune BoT-SORT / ByteTrack Kalman filters for turbid water ($\text{MOTA} \ge 0.80$) | `P0` | ML / DL | AI Team |
| **TSK-11** | Integrate MongoDB event logger directly into `AnimalLens.analyze_video()` | `P1` | Database | AI Team |

### Column D: Planned (Sprint 4: Weeks 7–8 — Temporal Action & RTSP)
| Task ID | Task Description | Priority | Category | Assignee |
| :--- | :--- | :---: | :---: | :---: |
| **TSK-12** | Implement kinematic feature extraction (velocity, IID, approach angles) | `P0` | ML / DL | AI Team |
| **TSK-13** | Benchmark edge RTSP ingestion latency (target: $< 60\text{ms}$ frame latency) | `P1` | Platform | AI Team |
| **TSK-14** | Implement MongoDB aggregation queries for real-time transition matrices | `P1` | Database | AI Team |

### Column E: Planned (Sprint 5: Weeks 9–10 — Ollama & Active Learning)
| Task ID | Task Description | Priority | Category | Assignee |
| :--- | :--- | :---: | :---: | :---: |
| **TSK-15** | Benchmark local Ollama models (Gemma 3 vs. Qwen 2.5 vs. LLaMA 3.2) | `P1` | ML / DL | AI Team |
| **TSK-16** | Connect low-confidence predictions to MongoDB `uncertainty_queue` | `P0` | Active Learn | AI Team |

### Column F: Completed (Phase 1)
| Task ID | Task Description | Status |
| :--- | :--- | :---: |
| **TSK-00A** | Standalone repository architecture & Pydantic schemas | `DONE` |
| **TSK-00B** | Decoupled Layer A perception protocols & Layer B Ollama provider | `DONE` |
| **TSK-00C** | Redclaw species taxonomy adapter (`species/redclaw/`) | `DONE` |
| **TSK-00D** | Python SDK, Typer CLI (`doctor`, `models`), FastAPI REST/WS | `DONE` |
| **TSK-00E** | Operations Research metrics (Markov transitions, Clark-Evans $R$) | `DONE` |
| **TSK-00F** | 31 automated unit tests passing & pushed to GitHub | `DONE` |
