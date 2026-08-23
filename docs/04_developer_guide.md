# Developer Guide & API Reference

Comprehensive engineering guide for installing AnimalLens, using the Python SDK, training custom vision models, performing 24-point skeletal pose biomechanics, managing ReID identity galleries, and deploying production Docker microservices.

---

## 1. Quick Installation & Setup

```bash
# Install core package with deep learning and tracking dependencies
pip install git+https://github.com/cvrvai/AnimalLens.git
```

Verify hardware acceleration and dependencies:
```bash
animallens doctor
```

---

## 2. Python SDK Reference

### 2.1 Basic Behavioral Analysis
```python
from animallens import AnimalLens

# Initialize engine for canine analysis with optional Ollama Layer B reasoning
lens = AnimalLens(species="dog", reasoning="ollama:gemma3")

# Analyze video file and print formatted ethogram timeline
result = lens.analyze_video("data/raw/videos/canine_pack.mp4", sample_fps=5.0)
print(result.format_timeline_text())

# Access Markov transition matrices
transition_matrix = result.get_transition_matrix()
print("State Transitions:", transition_matrix.to_dict_matrix())
```

### 2.2 Skeletal Pose Estimation & Biomechanics (Phase 9)
```python
from animallens import AnimalLens

lens = AnimalLens(species="dog")
result = lens.analyze_image("dog_pose.jpg")

for event in result.behaviors:
    for subject in event.subjects:
        print(f"Track ID: {subject.track_id}")
        
        # 24-Point Anatomical Keypoints
        for kp in subject.keypoints or []:
            print(f"  Landmark: {kp.name:15s} ({kp.x:.3f}, {kp.y:.3f}) conf: {kp.confidence:.2f}")
            
        # Biomechanical Angles & Clinical Lameness Metrics
        bio = subject.attributes.get("biomechanics", {})
        print(f"  Spine Flexion Angle: {bio.get('spine_flexion_angle_deg')} deg")
        print(f"  Left Elbow Flexion:  {bio.get('left_elbow_angle_deg')} deg")
        print(f"  Gait Asymmetry:      {bio.get('gait_asymmetry_score')}")
        print(f"  Veterinary Rating:   {bio.get('veterinary_gait_classification')}")
```

### 2.3 Deep Metric Re-Identification (Phase 11)
```python
import cv2
from animallens.reid import ReIDGallery, ReIDFeatureExtractor

# Initialize persistent gallery
gallery = ReIDGallery(match_threshold=0.82)

# Register known dogs
img_max = cv2.imread("max_anchor.jpg")
gallery.match_or_create(img_max, track_id=1, default_prefix="Max")

# Identify dog in a new camera feed
new_crop = cv2.imread("query_crop.jpg")
matched_name, similarity = gallery.identify(gallery.extractor.extract(new_crop))
print(f"Identified Subject: {matched_name} (Confidence: {similarity:.2%})")
```

---

## 3. CLI Command Suite

| Command | Purpose | Example |
| :--- | :--- | :--- |
| `animallens analyze` | Run detection, tracking, & pose estimation on local files | `animallens analyze data/video.mp4 --species dog` |
| `animallens train` | 1-click dataset extraction & transfer learning from video | `animallens train --video clip.mp4 --epochs 50 --device 0` |
| `animallens serve` | Launch local FastAPI REST & WebSocket server | `animallens serve --host 0.0.0.0 --port 8000` |
| `animallens models` | Pull & verify pretrained weights from Hugging Face | `animallens models pull canine-yolov8s` |
| `animallens dataset`| Split dataset with leak prevention and Cohen's Kappa | `animallens dataset split --source data/ --val-split 0.2` |
| `animallens doctor` | System diagnostic report on Python, PyTorch, CUDA | `animallens doctor` |

---

## 4. REST & WebSocket API Reference

Launch the microservice server:
```bash
animallens serve --host 0.0.0.0 --port 8000
```
Interactive OpenAPI documentation: [http://localhost:8000/docs](http://localhost:8000/docs).

### Key Endpoints

* `GET /v1/health`: Returns system status, inference backend, and loaded models.
* `POST /v1/analyze/video`: Multipart upload of video file for full behavioral timeline, tracking telemetry, keypoints, and biomechanics.
* `POST /v1/analyze/image`: Multipart upload of single image.
* `GET /v1/reid/gallery`: List all registered animal profiles in the vector gallery.
* `POST /v1/reid/register`: Register a named animal into the persistent ReID gallery.
* `POST /v1/train`: Trigger automated model fine-tuning run.
* `WS /v1/events`: Real-time streaming WebSocket feed for live frontend canvas HUD rendering.

---

## 5. Production Docker Compose Stack (Phase 12)

Start the full microservice cluster (AnimalLens API + MongoDB + Ollama LLM):

```bash
docker-compose up -d --build
```
