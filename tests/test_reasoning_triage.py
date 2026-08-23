"""
Unit tests for Layer B Multimodal Prompt Synthesizer and Active Learning Uncertainty Triage.
"""
import mongomock
import pytest
from animallens.core.schemas import (
    BehaviorEvent,
    BehaviorInfo,
    BoundingBox,
    ModelInfo,
    SourceInfo,
    SourceType,
    SpeciesInfo,
    SubjectInfo,
    TemporalInfo,
)
from animallens.reasoning.prompts import PromptSynthesizer
from animallens.reasoning.triage import UncertaintyTriageEngine
from animallens.storage.config import MongoConfig
from animallens.storage.mongodb import MongoDBStorage


def _sample_event(confidence: float = 0.90, label: str = "foraging", category: str = "feeding") -> BehaviorEvent:
    return BehaviorEvent(
        event_id="evt_triage_test",
        species=SpeciesInfo(id="cherax_quadricarinatus", name="Redclaw Crayfish"),
        source=SourceInfo(type=SourceType.VIDEO, camera_id="CAM-01"),
        subjects=[
            SubjectInfo(track_id=1, bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2)),
            SubjectInfo(track_id=2, bbox=BoundingBox(x_min=0.3, y_min=0.3, x_max=0.4, y_max=0.4)),
        ],
        behavior=BehaviorInfo(category=category, label=label, confidence=confidence),
        temporal=TemporalInfo(start=0.0, end=3.0, duration=3.0),
        model=ModelInfo(species_model="redclaw-behavior-v1"),
    )


def test_prompt_synthesizer_injection():
    """Verify prompt synthesis injects species, behavioral kinematics, and structured queries."""
    event = _sample_event(confidence=0.88, label="fighting", category="aggression")
    kinematics = {
        "mean_speed": 0.14,
        "polarization_index": 0.85,
        "pairwise": [{"track_id_1": 1, "track_id_2": 2, "distance": 0.12, "approach_rate": -0.09, "is_in_contact": True}],
    }

    prompt = PromptSynthesizer.build_event_prompt(event, kinematics_data=kinematics)
    assert "Redclaw Crayfish" in prompt
    assert "aggression.fighting" in prompt
    assert "88.0%" in prompt
    assert "Mean Group Speed: 0.14" in prompt
    assert "Distance=0.12" in prompt


def test_prompt_synthesizer_parse_response():
    """Verify parsing natural language LLM output into structured ReasoningOutput."""
    raw_llm = (
        "Crayfish observed engaging in agonistic territorial defense.\n\n"
        "Aggressive chela displays indicate resource competition over benthic shelter.\n\n"
        "Recommendations:\n"
        "- Increase PVC pipe shelter density in the rearing tank.\n"
        "- Monitor dissolved oxygen levels (>5.0 mg/L)."
    )

    reasoning = PromptSynthesizer.parse_llm_response(raw_llm, provider="ollama:gemma3", model="gemma3")
    assert reasoning.provider == "ollama:gemma3"
    assert "agonistic territorial defense" in reasoning.summary
    assert len(reasoning.recommendations) >= 1
    assert "shelter density" in reasoning.recommendations[0]


def test_uncertainty_triage_low_confidence():
    """Verify triage catches low confidence predictions."""
    storage = MongoDBStorage(client=mongomock.MongoClient(), config=MongoConfig(db_name="test_triage"))
    triage = UncertaintyTriageEngine(min_confidence_threshold=0.55, storage=storage)

    event = _sample_event(confidence=0.42)
    result = triage.evaluate(event, save_to_storage=True)

    assert result.is_uncertain is True
    assert event.behavior.is_uncertain is True
    assert "Low confidence" in result.reason

    # Verify document in MongoDB active learning queue
    docs = storage.get_uncertainty_queue()
    assert len(docs) == 1
    assert docs[0]["event_ref_id"] == "evt_triage_test"


def test_uncertainty_triage_kinematic_anomaly():
    """Verify triage catches physical / kinematic contradictions."""
    storage = MongoDBStorage(client=mongomock.MongoClient(), config=MongoConfig(db_name="test_triage_kin"))
    triage = UncertaintyTriageEngine(storage=storage)

    # Labeled resting but speed is abnormally fast
    event = _sample_event(confidence=0.85, label="resting", category="resting")
    kin = {"mean_speed": 0.22}

    result = triage.evaluate(event, kinematics=kin, save_to_storage=True)
    assert result.is_uncertain is True
    assert "Kinematic anomaly" in result.reason
    assert result.priority == 3
