"""
AnimalLens Example: Layer A Vision AI + Layer B Ollama Reasoning.
"""
from PIL import Image
from animallens import AnimalLens

# 1. Connect any Ollama model (e.g. gemma3, qwen2.5, llama3.2)
ai = AnimalLens(
    species="redclaw",
    reasoning="ollama:gemma3"
)

# 2. Analyze image or video
image = Image.new("RGB", (640, 480), color=(25, 40, 55))
result = ai.analyze_image(image)

# 3. Print Layer A Vision results
print(f"Layer A Vision Detected: {result.events_count} event(s)")
if result.behaviors:
    evt = result.behaviors[0]
    print(f"Behavior: {evt.behavior.category}.{evt.behavior.label} (Conf: {evt.behavior.confidence:.2f})")

# 4. Print Layer B LLM Reasoning
if result.reasoning:
    print("\n--- Layer B LLM Reasoning ---")
    print("Model:", result.reasoning.model)
    print("Summary:", result.reasoning.summary)
    if result.reasoning.explanation:
        print("Ethological Explanation:", result.reasoning.explanation)
    if result.reasoning.recommendations:
        print("Recommendations:")
        for r in result.reasoning.recommendations:
            print(f"  * {r}")
