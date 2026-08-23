# Deep Learning, Computer Vision & MongoDB Architecture

This document specifies the deep learning models, loss formulations, computer vision pipelines, and MongoDB database architecture for AnimalLens.

---

## 1. Deep Learning & Perception Architecture

```text
+-----------------------------------------------------------------------------------+
|                        Layer A: Computer Vision & ML Engine                       |
+-----------------------------------------------------------------------------------+
  Raw Video Stream (1080p @ 30 FPS / RTSP / MP4)
        |
        v
  [ Ingestion & Frame Sampling ] -> Downsample to 5-15 FPS (640x640 RGB)
        |
        v
  [ YOLOv8 Object Detection ] -> Carapace & Body BBoxes + Confidence scores
        |
        v
  [ BoT-SORT Multi-Object Tracker ] -> ReID Appearance Embeddings + Kalman Filter
        |                               (Persistent Track IDs & Velocity Vectors)
        v
  [ Rolling Video Buffer ] -> 15-second Ring Buffer (Trajectory + Spatial Distance)
        |
        v
  [ Temporal Action Segmentation ] -> MS-TCN++ / ST-GCN / Kinematic Action Engine
        |                              (Mating, Aggression, Foraging, Resting)
        v
  [ BehaviorEvent JSON Generator ] -> Emits Strongly-Typed Pydantic Schemas
```

### Detection Model (YOLOv8)
* **Backbone**: Modified CSPDarknet53 with C2f feature extraction modules.
* **Neck**: Path Aggregation Network (PANet) with multi-scale feature pyramids (P3, P4, P5).
* **Loss Functions**:
  $$\mathcal{L}_{\text{det}} = \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}}$$
  * Complete IoU ($\mathcal{L}_{\text{CIoU}}$) ensures robust bounding box regression under partial occlusion and underwater light distortion.

### Tracking Model (BoT-SORT / ByteTrack)
* **Kinematic State Vector**: $\mathbf{x}_t = [x, y, a, h, \dot{x}, \dot{y}, \dot{a}, \dot{h}]^T$ modeled via constant velocity Kalman filtering.
* **Cost Matrix Formulation**: Combines high-order appearance cosine distance with motion Mahalanobis distance to prevent ID switches during sparring interactions:
  $$C_{ij} = (1 - \alpha) \cdot d_{\text{motion}}(i, j) + \alpha \cdot d_{\text{appearance}}(f_i, f_j)$$

### Temporal Action Segmentation (MS-TCN++ / Kinematic Engine)
* **Input Window**: Sequence of $T=75$ frames ($15\text{s} \times 5\text{FPS}$) containing normalized bounding box centroids, pairwise distance vectors, and velocity magnitude differentials.
* **Loss Function**: Multi-class Cross-Entropy with truncated Mean Squared Error smoothing loss to enforce temporal smoothness:
  $$\mathcal{L}_{\text{temporal}} = \mathcal{L}_{\text{CE}} + \lambda_{\text{smooth}} \frac{1}{T} \sum_{t=1}^{T-1} \min\left(\Delta_{\text{thresh}}, \|\hat{y}_{t+1} - \hat{y}_t\|^2\right)$$

---

## 2. MongoDB Database Architecture

AnimalLens uses **MongoDB** as its primary persistence and analytical database. MongoDB is selected because the hierarchical `BehaviorEvent` JSON schema maps 1-to-1 to native BSON documents, supporting high-throughput streaming ingestion and flexible time-series aggregation.

```text
                             +-----------------------+
                             |    AnimalLens Core    |
                             +-----------------------+
                                         |
               +-------------------------+-------------------------+
               |                         |                         |
               v                         v                         v
     +-------------------+     +-------------------+     +-------------------+
     | events Collection |     | sessions Database |     | uncertainty_queue |
     | (Time-Series)     |     | (Metadata & Tank) |     | (Active Learning) |
     +-------------------+     +-------------------+     +-------------------+
               |                         |                         |
               +-------------------------+-------------------------+
                                         |
                                         v
                         +-------------------------------+
                         | MongoDB Aggregation Pipelines |
                         | * Markov Transition Matrices  |
                         | * 24h Circadian Time Budgets  |
                         | * Territorial Crowding Trends |
                         +-------------------------------+
```

### Collection Schemas

#### 1. `events` Collection (Time-Series Collection)
Stores all discrete behavioral occurrences:
```json
{
  "_id": "66c841a0e1b2c3d4e5f60001",
  "event_id": "evt_01928",
  "timestamp": 1724400000.0,
  "datetime": "2026-08-23T10:00:00.000Z",
  "species": {
    "id": "cherax_quadricarinatus",
    "name": "Redclaw Crayfish",
    "scientific_name": "Cherax quadricarinatus"
  },
  "source": {
    "type": "camera",
    "camera_id": "CAM-TANK-01",
    "session_id": "sess_20260823_0800"
  },
  "subjects": [
    { "track_id": 17, "animal_id": "F-003", "velocity": 0.04, "bbox": [0.12, 0.34, 0.28, 0.52] },
    { "track_id": 23, "animal_id": "M-002", "velocity": 0.05, "bbox": [0.18, 0.38, 0.32, 0.55] }
  ],
  "behavior": {
    "category": "reproduction",
    "label": "mating",
    "confidence": 0.93,
    "is_uncertain": false
  },
  "temporal": {
    "start": 42.1,
    "end": 74.4,
    "duration": 32.3
  },
  "spatial": {
    "inter_individual_distance": 0.08,
    "clark_evans_index": 0.65
  },
  "model": {
    "species_model": "redclaw-behavior-v1",
    "version": "1.0.0"
  }
}
```
**Indexes**:
* `{ "datetime": 1, "species.id": 1, "source.camera_id": 1 }`
* `{ "behavior.category": 1, "behavior.label": 1 }`
* `{ "subjects.track_id": 1 }`

#### 2. `sessions` Collection
Stores metadata for video recording sessions and experimental tanks:
```json
{
  "_id": "sess_20260823_0800",
  "tank_id": "TANK-03",
  "camera_id": "CAM-TANK-01",
  "resolution": "1920x1080",
  "fps": 30.0,
  "start_time": "2026-08-23T08:00:00.000Z",
  "end_time": "2026-08-23T10:00:00.000Z",
  "cohort": {
    "cohort_id": "COHORT-RC-2026A",
    "total_count": 12,
    "male_count": 4,
    "female_count": 8,
    "stocking_density_per_sqm": 6.0
  },
  "water_parameters": {
    "temperature_celsius": 27.5,
    "dissolved_oxygen_ppm": 6.8,
    "ph": 7.6
  }
}
```

#### 3. `uncertainty_queue` Collection (Active Learning)
Captures low-confidence predictions ($\text{conf} < 0.45$) or `unknown` behaviors for human biologist review:
```json
{
  "_id": "unc_0001",
  "event_ref_id": "evt_01945",
  "keyframe_image_uri": "s3://animallens-datasets/uncertainty/20260823/frame_4210.jpg",
  "video_clip_uri": "s3://animallens-datasets/uncertainty/20260823/clip_4200_4250.mp4",
  "model_prediction": {
    "category": "social_interaction",
    "label": "unknown",
    "confidence": 0.41
  },
  "verified_by_human": false,
  "human_verified_label": null,
  "notes": "Possible pre-copulatory grooming or aggressive claw grappling"
}
```

---

## 3. MongoDB Aggregation Pipelines

### Aggregation 1: Real-Time Behavior State Transition Matrix
Compute empirical transition counts $N_{ij}$ over any time window directly in MongoDB:
```javascript
db.events.aggregate([
  { $match: { "source.session_id": "sess_20260823_0800" } },
  { $sort: { "temporal.start": 1 } },
  {
    $group: {
      _id: "$source.session_id",
      sequence: { $push: "$behavior.label" }
    }
  }
]);
```

### Aggregation 2: 24-Hour Circadian Time Budget
Compute hourly activity distribution to evaluate feeding schedules and nocturnal activity:
```javascript
db.events.aggregate([
  {
    $match: {
      "datetime": {
        $gte: ISODate("2026-08-23T00:00:00.000Z"),
        $lt: ISODate("2026-08-24T00:00:00.000Z")
      }
    }
  },
  {
    $group: {
      _id: {
        hour: { $hour: "$datetime" },
        category: "$behavior.category"
      },
      total_duration: { $sum: "$temporal.duration" },
      event_count: { $sum: 1 }
    }
  },
  { $sort: { "_id.hour": 1 } }
]);
```
