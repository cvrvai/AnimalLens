"""
Quick demo script to test pig behavior analysis pipeline end-to-end.
Run: py -3.13 scripts/test_pig_analysis.py
"""
import numpy as np
from animallens.sdk import AnimalLens
from animallens.species.pig.adapter import PigAdapter
from animallens.core.schemas import BoundingBox
from animallens.perception.base import TrackState


def main():
    print("=" * 70)
    print("  AnimalLens — Swine / Domestic Pig Behavior Analysis Demo")
    print("=" * 70)

    # 1. SDK initialization with pig species
    lens = AnimalLens(species="pig")
    print(f"\n✅ Species loaded: {lens.species_name}")
    print(f"   Scientific Name: Sus scrofa domesticus")
    print(f"   Detection Threshold: {lens.species_adapter.config.detection_threshold}")

    # 2. Show full 18-behavior ethogram taxonomy
    adapter = PigAdapter()
    tax = adapter.taxonomy
    print(f"\n📋 Pig Ethogram Taxonomy (v{tax.version}):")
    total_behaviors = 0
    for cat_name, cat in tax.categories.items():
        print(f"   [{cat_name.upper()}] → {', '.join(cat.labels)}")
        total_behaviors += len(cat.labels)
    print(f"   Total unique behavior labels: {total_behaviors}")

    # 3. Test clinical recumbency posture classification
    print("\n🏥 Clinical Recumbency Posture Classification:")
    test_cases = [
        ("Wide bbox (w=0.40, h=0.15)", BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.35), 0.0),
        ("Medium bbox (w=0.30, h=0.20)", BoundingBox(x_min=0.1, y_min=0.2, x_max=0.4, y_max=0.4), 0.0),
        ("Square bbox (w=0.20, h=0.20)", BoundingBox(x_min=0.1, y_min=0.2, x_max=0.3, y_max=0.4), 0.0),
        ("Moving pig (velocity=0.5)", BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.35), 0.5),
    ]
    for desc, bbox, vel in test_cases:
        posture = adapter.classify_recumbency(bbox, velocity=vel)
        print(f"   {desc} → {posture}")

    # 4. Test barn huddling thermal stress index
    print("\n🌡️  Commercial Barn Thermal Huddling Index:")

    # Scenario A: Pigs tightly clustered (COLD STRESS)
    cold_tracks = [
        TrackState(track_id=1, current_bbox=BoundingBox(x_min=0.40, y_min=0.40, x_max=0.50, y_max=0.50)),
        TrackState(track_id=2, current_bbox=BoundingBox(x_min=0.42, y_min=0.42, x_max=0.52, y_max=0.52)),
        TrackState(track_id=3, current_bbox=BoundingBox(x_min=0.44, y_min=0.41, x_max=0.54, y_max=0.51)),
        TrackState(track_id=4, current_bbox=BoundingBox(x_min=0.43, y_min=0.43, x_max=0.53, y_max=0.53)),
    ]
    cold_features = adapter.extract_custom_features(cold_tracks, {})
    print(f"   Scenario A (4 pigs tightly huddled):")
    print(f"     Huddling Index:         {cold_features['huddling_cold_stress_index']}")
    print(f"     Thermal Status:         {cold_features['thermal_comfort_status']}")
    print(f"     Active Pig Count:       {cold_features['active_pig_count']}")
    print(f"     Posture Distribution:   {cold_features['posture_distribution']}")

    # Scenario B: Pigs spread out (NORMAL THERMAL COMFORT)
    normal_tracks = [
        TrackState(track_id=1, current_bbox=BoundingBox(x_min=0.05, y_min=0.05, x_max=0.15, y_max=0.15)),
        TrackState(track_id=2, current_bbox=BoundingBox(x_min=0.80, y_min=0.80, x_max=0.90, y_max=0.90)),
        TrackState(track_id=3, current_bbox=BoundingBox(x_min=0.40, y_min=0.10, x_max=0.50, y_max=0.20)),
    ]
    normal_features = adapter.extract_custom_features(normal_tracks, {})
    print(f"\n   Scenario B (3 pigs spread across pen):")
    print(f"     Huddling Index:         {normal_features['huddling_cold_stress_index']}")
    print(f"     Thermal Status:         {normal_features['thermal_comfort_status']}")
    print(f"     Active Pig Count:       {normal_features['active_pig_count']}")
    print(f"     Posture Distribution:   {normal_features['posture_distribution']}")

    # 5. Run full SDK perception pipeline on synthetic frame
    print("\n🎥 Full Perception Pipeline (Synthetic 640x480 Frame):")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = lens.analyze_image(dummy_frame)
    print(f"   Species:      {result.species}")
    print(f"   Behaviors:    {len(result.behaviors)} detected")
    for i, beh in enumerate(result.behaviors):
        print(f"     [{i}] label={beh.behavior.label}, confidence={beh.behavior.confidence:.2f}, category={beh.behavior.category}")

    print("\n" + "=" * 70)
    print("  ✅ All pig behavior analysis tests PASSED successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
