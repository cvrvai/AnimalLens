"""
Prompt Synthesis Engine for Layer B Ollama / Local Multimodal LLM Reasoning.
Synthesizes ethological biology context, kinematic trajectories, and multi-agent interaction states.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from animallens.core.schemas import BehaviorEvent, ReasoningOutput


class PromptSynthesizer:
    """
    Constructs high-signal, biology-grounded prompts for LLM reasoning over vision perception outputs.
    """

    @staticmethod
    def build_event_prompt(
        event: BehaviorEvent,
        kinematics_data: Optional[Dict[str, Any]] = None,
        custom_question: Optional[str] = None,
    ) -> str:
        species_info = event.species
        behavior_info = event.behavior
        temporal_info = event.temporal
        subjects = event.subjects

        # Extract kinematic details if present in event or argument
        kinematics = kinematics_data or event.metadata.get("kinematics", {})
        tracks_kin = kinematics.get("tracks", {})
        pairwise_kin = kinematics.get("pairwise", [])
        mean_speed = kinematics.get("mean_speed", "N/A")
        polarization = kinematics.get("polarization_index", "N/A")

        pairwise_desc = []
        for p in pairwise_kin:
            p_dist = p.get("distance", "N/A")
            p_rate = p.get("approach_rate", "N/A")
            p_contact = "Contact" if p.get("is_in_contact") else "No Contact"
            pairwise_desc.append(f"    - Animal {p.get('track_id_1')} vs Animal {p.get('track_id_2')}: Distance={p_dist}, ApproachRate={p_rate}, {p_contact}")

        pairwise_str = "\n".join(pairwise_desc) if pairwise_desc else "    - No pairwise interactions."

        prompt = f"""You are the AnimalLens Biological Reasoning Engine specialized in Quantitative Ethology, Animal Behavior, and Aquaculture Ecology.

[Observation Context]
* Species: {species_info.name} ({species_info.scientific_name or 'Cherax quadricarinatus'})
* Detected Behavior: {behavior_info.category}.{behavior_info.label} (Model Confidence: {behavior_info.confidence * 100:.1f}%)
* Temporal Extent: {temporal_info.start:.2f}s to {temporal_info.end:.2f}s (Duration: {temporal_info.duration:.2f}s)
* Number of Subjects: {len(subjects)}

[Kinematic Differential Metrics]
* Mean Group Speed: {mean_speed}
* Group Polarization Index: {polarization}
* Pairwise Proximity & Dynamics:
{pairwise_str}

[Instructions]
Provide an expert ethological interpretation of this observed behavior event.
Structure your response as follows:
1. Executive Summary: 1-2 sentence concise summary.
2. Biological Explanation: Ethological function and biological significance of this behavior.
3. Recommendations: 2-3 actionable aquaculture / environmental recommendations (e.g. stocking density, water quality, shelter provision).
"""
        if custom_question:
            prompt += f"\n[User Specific Inquiry]\nPlease specifically address this question: {custom_question}\n"

        return prompt

    @staticmethod
    def parse_llm_response(
        raw_text: str,
        provider: str = "ollama:gemma3",
        model: str = "gemma3",
    ) -> ReasoningOutput:
        """Parse natural language LLM response into structured ReasoningOutput."""
        lines = [line.strip() for line in raw_text.strip().split("\n") if line.strip()]
        summary = lines[0] if lines else "Behavior event analyzed."

        recommendations = []
        for line in lines:
            if line.startswith(("-", "*", "1.", "2.", "3.", "•")):
                cleaned = line.lstrip("-*• 0123456789.").strip()
                if len(cleaned) > 10 and any(keyword in cleaned.lower() for keyword in ["tank", "water", "density", "shelter", "feed", "separate", "monitor"]):
                    recommendations.append(cleaned)

        if not recommendations:
            recommendations = ["Maintain optimal water quality parameters (DO > 5 mg/L, Temp 24-28 C).", "Monitor tank social hierarchy for excessive aggression."]

        return ReasoningOutput(
            provider=provider,
            model=model,
            summary=summary,
            explanation=raw_text,
            recommendations=recommendations[:3],
            raw_response=raw_text,
        )
