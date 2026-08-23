"""
Unit tests for reasoning providers and Ollama integration.
"""
import pytest
from animallens.core.schemas import (
    BehaviorEvent,
    BehaviorInfo,
    ModelInfo,
    SourceInfo,
    SourceType,
    SpeciesInfo,
    TemporalInfo,
)
from animallens.reasoning.base import NoOpReasoningProvider
from animallens.reasoning.factory import get_reasoning_provider
from animallens.reasoning.ollama import OllamaReasoningProvider


@pytest.mark.asyncio
async def test_noop_reasoning_provider():
    provider = get_reasoning_provider(None)
    assert isinstance(provider, NoOpReasoningProvider)
    assert provider.is_enabled is False

    event = BehaviorEvent(
        species=SpeciesInfo(id="cherax_quadricarinatus", name="Redclaw Crayfish"),
        source=SourceInfo(type=SourceType.IMAGE),
        behavior=BehaviorInfo(category="reproduction", label="mating", confidence=0.95),
        temporal=TemporalInfo(start=0.0, end=1.0, duration=1.0),
        model=ModelInfo(species_model="redclaw-behavior-v1"),
    )

    out = await provider.explain_event(event)
    assert out.provider == "none"
    assert "mating" in out.summary


def test_reasoning_factory():
    p1 = get_reasoning_provider("none")
    assert isinstance(p1, NoOpReasoningProvider)

    p2 = get_reasoning_provider("ollama:gemma3")
    assert isinstance(p2, OllamaReasoningProvider)
    assert p2.model_name == "gemma3"

    p3 = get_reasoning_provider("qwen2.5:7b")
    assert isinstance(p3, OllamaReasoningProvider)
    assert p3.model_name == "qwen2.5:7b"


def test_ollama_parser():
    provider = OllamaReasoningProvider(model_name="gemma3")
    raw_llm_output = (
        "SUMMARY: Observed mating interaction between two mature crayfish.\n"
        "EXPLANATION: In Cherax quadricarinatus, copulation involves the male rolling the female into a supine position and depositing a spermatophore.\n"
        "RECOMMENDATIONS:\n"
        "- Maintain optimal water temperature (26-28C)\n"
        "- Provide shelter tiles to reduce post-mating aggression"
    )

    parsed = provider._parse_llm_response(raw_llm_output, default_summary="Fallback")
    assert "Observed mating interaction" in parsed.summary
    assert "copulation involves" in parsed.explanation
    assert len(parsed.recommendations) == 2
    assert "Maintain optimal water temperature" in parsed.recommendations[0]
