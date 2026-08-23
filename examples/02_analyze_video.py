"""
AnimalLens Example: Analyze recorded video file to generate a structured timeline.
"""
from animallens import AnimalLens

# 1. Initialize AnimalLens
ai = AnimalLens(species="redclaw", reasoning=None)

# 2. Analyze video file (uses synthetic frames if file does not exist)
result = ai.analyze_video("tank_sample.mp4", sample_fps=5.0, max_duration_seconds=12.0)

# 3. Print human-readable behavior timeline
print("=== Behavior Timeline ===")
print(result.format_timeline_text())

# 4. Print structured JSON event
if result.behaviors:
    sample_event = result.behaviors[0]
    print("\n=== Sample Structured Event ===")
    print(sample_event.model_dump_json(indent=2))
