"""
AnimalLens Example: Real-time live camera or RTSP behavior streaming.
"""
from animallens import AnimalLens

ai = AnimalLens(species="redclaw", reasoning=None)

print("Starting real-time behavior detection stream on RTSP/Camera (press Ctrl+C to stop)...")
count = 0

# Connect to RTSP camera or webcam index (e.g. 0)
for event in ai.stream("rtsp://demo-camera.local/live"):
    count += 1
    print(f"[{event.temporal.start:05.1f}s] Detected: {event.behavior.category}.{event.behavior.label} (Conf: {event.behavior.confidence:.2f})")

    # Stop after 10 events for demonstration
    if count >= 10:
        print("Demo completed 10 events.")
        break
