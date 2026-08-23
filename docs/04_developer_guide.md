# Developer Guide & API Reference

This guide covers installing AnimalLens, using the Python SDK, integrating the REST/WebSocket API, and creating custom species adapters.

---

## 1. Quick Installation

```bash
pip install animallens
```

Run system diagnostics:
```bash
animallens doctor
```

---

## 2. Python SDK

### Basic Usage
```python
from animallens import AnimalLens

# Initialize with species and optional Ollama reasoning
lens = AnimalLens(species="redclaw", reasoning="ollama:gemma3")

# Analyze static image
image_result = lens.analyze_image("tank.jpg")

# Analyze video file and get timeline
video_result = lens.analyze_video("tank.mp4", sample_fps=5.0)
print(video_result.format_timeline_text())

# Access Operations Research and Quantitative Analytics
transition_matrix = video_result.get_transition_matrix()
print("State Transitions:", transition_matrix.to_dict_matrix())

spatial_metrics = video_result.get_spatial_metrics()
print(f"Clark-Evans Dispersion Index: {spatial_metrics.clark_evans_dispersion_index}")
```

### Real-Time Live Streaming
```python
for event in lens.stream("rtsp://camera-ip/live"):
    print(f"[{event.temporal.start:.1f}s] {event.behavior.category}.{event.behavior.label}")
```

---

## 3. REST & WebSocket API

Start local API server:
```bash
animallens serve --host 0.0.0.0 --port 8088
```

Open interactive Swagger documentation: `http://localhost:8088/docs`.

### WebSocket Real-time Behavior Stream
Connect to `ws://localhost:8088/v1/events` to receive real-time JSON payloads:
```json
{
  "type": "behavior.detected",
  "data": {
    "species": "cherax_quadricarinatus",
    "behavior": "mating",
    "confidence": 0.92,
    "event_id": "evt_01928"
  }
}
```

---

## 4. Adding a New Species Adapter

To add a new animal species (e.g. `pig`):

1. Create `animallens/species/pig/` with:
   * `config.yaml`
   * `taxonomy.yaml`
   * `adapter.py`
2. Subclass `SpeciesAdapter` and define species-specific taxonomy rules.
3. Register with `species_registry.register_adapter("pig", PigAdapter())`.
