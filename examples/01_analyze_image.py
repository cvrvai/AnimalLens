"""
AnimalLens Example: Analyze a single image.
"""
from PIL import Image
from animallens import AnimalLens

# 1. Initialize AnimalLens with desired species
ai = AnimalLens(species="redclaw", reasoning=None)

# 2. Create or load an image
image = Image.new("RGB", (800, 600), color=(30, 45, 60))

# 3. Analyze image
result = ai.analyze_image(image)

# 4. Inspect structured results
print("Species:", result.species)
print(f"Detected {len(result.behaviors)} behavior event(s):")
for event in result.behaviors:
    print(f"  * Category: {event.behavior.category} | Label: {event.behavior.label} (Conf: {event.behavior.confidence:.2f})")
    print(f"    Tracked Subjects: {len(event.subjects)}")
