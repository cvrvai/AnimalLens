<div align="center">

# 🐾 AnimalLens
### Edge AI Animal Vision & Ethology Platform for Next.js & Modern Web Apps

[![Next.js](https://img.shields.io/badge/Next.js-14%2F15-black?logo=next.js)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?logo=typescript)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-18%2F19-61DAFB?logo=react)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

**AnimalLens** provides frame-accurate animal detection, multi-animal Kalman tracking, kinematic speed extraction, and ethological behavior classification directly into your **Next.js & React** web applications.

Features out-of-the-box **Canine Vision AI** (*Canis lupus familiaris*) detecting gait, speed in km/h, posture, and social interactions at **>60 FPS**.

</div>

---

## ⚡ Next.js 3-Minute Quickstart

Connect real-time animal vision intelligence to your Next.js application in 3 steps:

### 1. Launch the Local Edge AI Microservice

Run the local vision engine via Docker (or run locally):

```bash
docker run -d -p 8000:8000 ghcr.io/cvrvai/animallens:latest
```

*API runs at `http://localhost:8000` with interactive Swagger docs at `http://localhost:8000/docs`.*

---

### 2. Call the Inference API from Next.js (App Router)

Create an API Route in `app/api/analyze/route.ts`:

```typescript
import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const file = formData.get("file") as File;

  if (!file) {
    return NextResponse.json({ error: "No video file provided" }, { status: 400 });
  }

  // Forward to AnimalLens Edge Vision Service
  const forwardData = new FormData();
  forwardData.append("file", file);
  forwardData.append("species", "dog");
  forwardData.append("sample_fps", "10.0");

  const response = await fetch("http://localhost:8000/v1/analyze/video", {
    method: "POST",
    body: forwardData,
  });

  const data = await response.json();
  return NextResponse.json(data);
}
```

---

### 3. Display AI Detections & Behavior Telemetry in React

Use standard React components to render the analysis results:

```tsx
"use client";

import { useState } from "react";

export default function AnimalLensDashboard() {
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState<any>(null);

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/api/analyze", { method: "POST", body: formData });
    const json = await res.json();
    setAnalysis(json);
    setLoading(false);
  }

  return (
    <div className="p-8 max-w-4xl mx-auto bg-slate-950 text-white rounded-2xl">
      <h1 className="text-2xl font-bold mb-4">🐾 AnimalLens Canine Vision AI</h1>
      <input type="file" accept="video/*" onChange={handleUpload} className="mb-6 block" />

      {loading && <p className="text-cyan-400">Processing video at >60 FPS with BoT-SORT...</p>}

      {analysis && (
        <div className="space-y-4">
          <div className="p-4 bg-slate-900 border border-cyan-500/30 rounded-xl">
            <h2 className="text-lg font-semibold text-cyan-400">Identified Species</h2>
            <p className="text-xl">{analysis.species} ({analysis.taxonomy_version})</p>
          </div>

          <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
            <h2 className="text-lg font-semibold mb-2">Behavior Timeline & Kinematics</h2>
            <div className="space-y-2">
              {analysis.timeline?.map((entry: any, i: number) => (
                <div key={i} className="flex justify-between items-center bg-slate-800/50 p-2 rounded">
                  <span className="text-slate-400 font-mono">[{entry.time}s]</span>
                  <span className="font-semibold text-emerald-400">{entry.behavior}</span>
                  <span className="text-xs bg-cyan-950 text-cyan-300 px-2 py-1 rounded">
                    {(entry.confidence * 100).toFixed(0)}% Conf
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
```

---

## 📡 Real-Time Live Camera Streaming (WebSockets)

For live security cameras, USB webcams, or pet monitors, connect directly over WebSockets in Next.js:

```typescript
// useAnimalLensSocket.ts
import { useEffect, useState } from "react";

export function useAnimalLensSocket() {
  const [liveEvent, setLiveEvent] = useState<any>(null);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/v1/events");

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "behavior.detected") {
        setLiveEvent(data.payload);
      }
    };

    return () => ws.close();
  }, []);

  return liveEvent;
}
```

---

## 📄 TypeScript Interface Reference

```typescript
export interface AnimalAnalysisResult {
  schema_version: string;
  species: string;
  duration_seconds: number;
  total_frames_analyzed: number;
  timeline: TimelineEntry[];
  behaviors: BehaviorEvent[];
}

export interface TimelineEntry {
  time: number;
  behavior: string;
  confidence: number;
}

export interface BehaviorEvent {
  event_id: string;
  timestamp: number;
  subjects: SubjectTelemetry[];
  behavior: {
    category: "locomotion" | "posture" | "social_behavior" | "aggression";
    label: string;
    human_readable: string;
    confidence: number;
  };
  temporal: {
    start: number;
    end: number;
    duration: number;
  };
}

export interface SubjectTelemetry {
  track_id: number;
  display_id: string; // e.g. "DOG-01"
  velocity_mps: number; // Speed in m/s
  velocity_kmh: number; // Speed in km/h
  heading_degrees: number; // 0-360 orientation
  bbox: {
    x_min: number;
    y_min: number;
    x_max: number;
    y_max: number;
  };
}
```

---

## 🔬 Supported Canine Behaviors & Kinematics

| Category | Detected Actions | Kinematic Profile |
| :--- | :--- | :--- |
| **Locomotion** | `running_gallop`, `trot`, `walk` | Velocity $> 3.5\text{ m/s}$ (Gallop), $1.2-3.5\text{ m/s}$ (Trot), $0.3-1.2\text{ m/s}$ (Walk) |
| **Posture** | `standing`, `sitting`, `lying_sternal` | Stationary velocity $< 0.3\text{ m/s}$ & aspect ratio posture geometry |
| **Social Behavior** | `play_bow`, `following`, `greeting` | Inter-Individual Distance ($\text{IID} < 0.6\text{m}$) & reciprocal heading |
| **Agonistic** | `aggressive_lunge`, `defensive_retreat` | Rapid approach acceleration & defensive spatial separation |

---

## 🐍 Optional: Python Core Engine & Custom Training

If you are a Machine Learning Engineer or Data Scientist who wants to fine-tune custom weights or run direct Python scripts:

### Install Python SDK:
```bash
pip install git+https://github.com/cvrvai/AnimalLens.git
```

### Python Script:
```python
from animallens import AnimalLens

lens = AnimalLens(species="dog")
result = lens.analyze("dog_video.mp4")
print(result.format_timeline_text())
```

### Start Server via Python CLI:
```bash
animallens serve --port 8000 --device cpu
```

---

## 🤝 Contributing & License

Contributions for new React components, Next.js templates, and species adapters are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

Licensed under the **Apache License, Version 2.0**.
