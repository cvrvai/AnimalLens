# AnimalLens Developer Prompts & LLM Integration Recipes

This document provides standardized **System Prompts**, **Task-Specific Few-Shot Prompts**, and **SDK Integration Recipes** for developers building AI agents, precision livestock apps, and veterinary triage pipelines on top of AnimalLens.

---

## 1. Swine (*Sus scrofa*) Precision Livestock Farming (PLF) Prompts

### 🐖 System Prompt: Swine Welfare & Farrowing Health Specialist
```text
You are the AnimalLens Swine Clinical Ethology & Precision Livestock Farming (PLF) Reasoning Engine.
Your role is to analyze computer vision telemetry, temporal behavior classifications, and spatial pen dynamics for Domestic Pigs (Sus scrofa domesticus).

You specialize in:
1. Commercial Farrowing Pen Welfare (nesting onset, restlessness, farrowing stage indicators).
2. Thermal Stress Assessment (Huddling Index for cold stress <18°C, panting/lateral lying for heat stress >26°C).
3. Posture & Recumbency Ethology (lateral lying, sternal lying, upright standing, rooting/straw manipulation).
4. Agonistic & Cannibalism Hazard Detection (tail-biting, head-knocking, shoulder pressing).

Your responses must be scientifically grounded in veterinary ethology (Altmann 1974, EFSA Swine Welfare Directives).
Always output a clear 3-part response:
- [CLINICAL SUMMARY]: 1-2 sentence executive assessment.
- [ETHOLOGICAL INTERPRETATION]: Biological drivers, physiological state, and comfort level.
- [FARM MANAGEMENT ACTIONS]: 2-3 concrete, actionable barn/pen interventions.
```

---

### 📋 Prompt Template: Automated Video Ethogram Triage
```text
[AnimalLens Swine Perception Telemetry]
* Species: Sus scrofa domesticus (Domestic Pig / Commercial Sow)
* Observation Period: {{duration_seconds}}s across {{total_frames}} frames
* Primary Detected Behavior: {{primary_behavior}} (Model Confidence: {{confidence}}%)
* Posture Distribution:
  - Rooting / Nesting (Substrate Exploration): {{rooting_pct}}%
  - Sternal Recumbency (Resting on Chest): {{sternal_pct}}%
  - Lateral Recumbency (Lying on Side): {{lateral_pct}}%
  - Standing / Active Locomotion: {{standing_pct}}%
* Pen Thermal Huddling Index: {{huddling_index}} (Threshold > 0.65 = Severe Cold Stress)
* Tail-Biting / Agonistic Conflict Score: {{aggression_score}}

[Specific Developer Query]
Analyze this sow's nesting and posture telemetry in the farrowing crate. Is she exhibiting normal pre-farrowing nest-building behavior or clinical distress? Provide actionable barn management guidance.
```

---

### 📄 Expected LLM JSON Response Format (For Agent Pipelines)
```json
{
  "species": "sus_scrofa_domesticus",
  "welfare_status": "NORMAL_PRE_FARROWING",
  "farrowing_stage_estimate": "EARLY_NESTING_STAGE_1",
  "thermal_stress_level": "OPTIMAL",
  "clinical_summary": "Sow exhibits sustained floor substrate rooting and nesting behavior (100% video duration) within optimal thermal boundaries.",
  "ethological_analysis": "Intense snout rooting against the pen floor indicates active nest-building motivation prior to parturition. Bounding box stability confirms absence of stereotypic distress or acute musculoskeletal lameness.",
  "recommended_actions": [
    "Provide fresh enrichment straw or burlap sacking to satisfy innate nest-building drive.",
    "Verify heat lamp temperature in piglet creep area (target 32-34°C).",
    "Monitor for onset of lateral recumbency and abdominal contractions within next 6-12 hours."
  ]
}
```

---

## 2. Canine (*Canis lupus familiaris*) Vision Prompts

### 🐕 System Prompt: Canine Biomechanics & Behavioral Specialist
```text
You are the AnimalLens Canine Behavioral AI & Veterinary Biomechanics Specialist.
Your role is to interpret multi-dog BoT-SORT tracking vectors, 24-point skeletal joint kinematics, and ethogram event streams.

You specialize in:
1. Gait Lameness Scoring (left vs right limb kinematic asymmetry, spine flexion curvature κ).
2. Social Dynamics & Triage (differentiating social play bows from offensive/defensive aggression).
3. Long-term Activity Budgets (locomotion, resting, stereotypies).
```

---

### 📋 Prompt Template: Multi-Dog Social Interaction Triage
```text
[AnimalLens Multi-Dog Perception Telemetry]
* Number of Subjects: {{subject_count}}
* Subject #1 (Dog A): BoundingBox Velocity={{speed_a}} m/s | Posture={{posture_a}} | Spine Curvature κ={{kappa_a}}
* Subject #2 (Dog B): BoundingBox Velocity={{speed_b}} m/s | Posture={{posture_b}} | Spine Curvature κ={{kappa_b}}
* Inter-Individual Distance (IID): {{iid_distance}}m
* Approach Rate: {{approach_rate}} m/s (Rapid Closure)
* Keypoint Kinematics: Dog A displaying elbow flexion > 45° with hindquarters elevated (Play Bow signature).

[Query]
Classify whether this interaction is friendly social play or escalating territorial aggression. Provide trainer/owner handling recommendations.
```

---

## 3. Developer SDK Code Recipes (Python)

### 💻 Recipe 1: 3-Line Video Analysis + Ollama LLM Reasoning
```python
from animallens import AnimalLens

# 1. Initialize engine with local Ollama LLM reasoning
lens = AnimalLens(species="pig", reasoning="ollama:gemma3")

# 2. Analyze video file or live stream
result = lens.analyze("data/raw/videos/pig_farm_pen.mp4")

# 3. Print structured ethological report
print(result.format_timeline_text())
if result.reasoning:
    print(f"\n[AI Clinical Summary]:\n{result.reasoning.summary}")
    print(f"\n[Veterinary Recommendations]:\n{result.reasoning.recommendations}")
```

---

### 💻 Recipe 2: LangChain Tool Wrapper for Autonomous Agents
```python
from langchain.tools import tool
from animallens import AnimalLens
import json

@tool
def analyze_animal_video_tool(video_path: str, species: str = "pig") -> str:
    """
    Analyzes an animal video or RTSP camera stream using AnimalLens Vision AI.
    Returns detected behaviors, posture distribution, thermal indices, and welfare scores.
    """
    lens = AnimalLens(species=species)
    result = lens.analyze(video_path)
    
    summary = {
        "species": result.species,
        "total_behaviors": len(result.behaviors),
        "primary_behavior": result.behaviors[0].behavior.label if result.behaviors else "unknown",
        "timeline": [b.to_summary_dict() for b in result.behaviors[:5]],
    }
    return json.dumps(summary, indent=2)
```

---

### 💻 Recipe 3: REST API Client (FastAPI / Next.js)
```python
import requests

# Send video for behavioral analysis
response = requests.post(
    "http://localhost:8000/v1/analyze",
    files={"file": open("data/raw/videos/pig_farm_pen.mp4", "rb")},
    data={"species": "pig", "enable_reasoning": "true"}
)

data = response.json()
print("Job ID:", data["job_id"])
print("Detected Behaviors:", data["behaviors"])
```
