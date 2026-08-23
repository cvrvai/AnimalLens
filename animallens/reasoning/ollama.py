"""
Ollama reasoning provider for AnimalLens.
Connects to any local/remote Ollama instance for event interpretation, Q&A, and biological reasoning.
"""
from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any, Dict, List, Optional
import httpx
from animallens.core.config import settings
from animallens.core.exceptions import ReasoningError
from animallens.core.schemas import BehaviorEvent, ReasoningOutput
from animallens.reasoning.base import BaseReasoningProvider

logger = logging.getLogger(__name__)

# Known vision-capable model families in Ollama
KNOWN_VISION_FAMILIES = {
    "gemma3",
    "llava",
    "bakllava",
    "llama3.2-vision",
    "moondream",
    "minicpm-v",
    "qwen2-vl",
    "qwen2.5-vl",
}


class OllamaClient:
    """Lightweight HTTP client for the Ollama REST API."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout

    async def list_models(self) -> List[Dict[str, Any]]:
        """Fetch list of installed models from Ollama."""
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.get(url)
                if res.status_code == 200:
                    data = res.json()
                    models = []
                    for item in data.get("models", []):
                        name = item.get("name", "")
                        details = item.get("details", {})
                        size_gb = round(item.get("size", 0) / (1024 ** 3), 2)
                        family = details.get("family", "")
                        is_vision = any(vf in name.lower() or vf in family.lower() for vf in KNOWN_VISION_FAMILIES)
                        models.append({
                            "name": name,
                            "size_gb": size_gb,
                            "family": family,
                            "parameter_size": details.get("parameter_size", "unknown"),
                            "quantization_level": details.get("quantization_level", "unknown"),
                            "is_vision": is_vision,
                        })
                    return models
                return []
        except Exception as e:
            logger.debug(f"Failed to connect to Ollama at {self.base_url}: {e}")
            return []

    async def generate(
        self,
        model: str,
        prompt: str,
        images: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Call Ollama /api/generate endpoint."""
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if images:
            payload["images"] = images

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(url, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    return data.get("response", "")
                else:
                    raise ReasoningError(
                        f"Ollama API returned HTTP {res.status_code}: {res.text}"
                    )
        except httpx.ConnectError:
            raise ReasoningError(
                f"Cannot connect to Ollama at '{self.base_url}'. Is the Ollama service running? "
                f"Start it with `ollama serve` or check your OLLAMA_BASE_URL setting."
            )
        except Exception as e:
            if isinstance(e, ReasoningError):
                raise
            raise ReasoningError(f"Ollama reasoning error: {e}") from e


class OllamaReasoningProvider(BaseReasoningProvider):
    """
    Ollama Reasoning Provider.
    Sends Layer A structured behavior events to any selected Ollama model.
    """

    def __init__(
        self,
        model_name: str = "gemma3",
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        # Strip potential "ollama:" prefix
        clean_model = model_name
        if clean_model.lower().startswith("ollama:"):
            clean_model = clean_model[7:]
        if not clean_model:
            clean_model = "gemma3"

        super().__init__(model_name=clean_model)
        self.client = OllamaClient(base_url=base_url, timeout=timeout)
        self._vision_capable: Optional[bool] = None

    @property
    def provider_name(self) -> str:
        return f"ollama:{self.model_name}"

    @property
    def is_vision_capable(self) -> bool:
        if self._vision_capable is not None:
            return self._vision_capable
        # Check by known family heuristics
        return any(vf in self.model_name.lower() for vf in KNOWN_VISION_FAMILIES)

    def _convert_frame_to_base64(self, frame: Any) -> Optional[str]:
        """Convert a PIL Image, numpy array or bytes to base64 string for Ollama vision."""
        try:
            from PIL import Image
            if isinstance(frame, bytes):
                return base64.b64encode(frame).decode("utf-8")
            if isinstance(frame, Image.Image):
                buf = io.BytesIO()
                frame.save(buf, format="JPEG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")
            # If numpy array
            if hasattr(frame, "shape"):
                img = Image.fromarray(frame)
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            logger.warning(f"Could not convert frame to base64: {e}")
        return None

    def _build_system_prompt(self, species_name: str) -> str:
        return (
            f"You are AnimalLens Reasoning Engine, an expert ethologist and animal behavior scientist specializing in {species_name}. "
            "You analyze structured computer vision behavior events detected by AnimalLens Layer A. "
            "Provide objective, biologically grounded interpretation, ethological context, and practical recommendations. "
            "Format your output clearly with:\n"
            "SUMMARY: <brief 1-2 sentence overview>\n"
            "EXPLANATION: <biological context, ethogram significance, trigger factors>\n"
            "RECOMMENDATIONS:\n"
            "- <action item 1>\n"
            "- <action item 2>"
        )

    def _parse_llm_response(self, text: str, default_summary: str) -> ReasoningOutput:
        """Parse structured fields from LLM response or fallback cleanly."""
        summary = default_summary
        explanation = ""
        recommendations: List[str] = []

        lines = text.strip().split("\n")
        current_section = None

        for line in lines:
            line_str = line.strip()
            if line_str.upper().startswith("SUMMARY:"):
                summary = line_str[8:].strip()
                current_section = "summary"
            elif line_str.upper().startswith("EXPLANATION:"):
                explanation = line_str[12:].strip()
                current_section = "explanation"
            elif line_str.upper().startswith("RECOMMENDATIONS:"):
                current_section = "recommendations"
            else:
                if current_section == "explanation" and line_str:
                    explanation += (" " if explanation else "") + line_str
                elif current_section == "recommendations" and line_str.startswith(("-", "*", "1.", "2.", "3.")):
                    cleaned = line_str.lstrip("-*0123456789. ")
                    if cleaned:
                        recommendations.append(cleaned)

        if not explanation:
            explanation = text

        return ReasoningOutput(
            provider=self.provider_name,
            model=self.model_name,
            summary=summary,
            explanation=explanation.strip(),
            recommendations=recommendations,
            raw_response=text,
        )

    async def explain_event(
        self,
        event: BehaviorEvent,
        frames: Optional[List[Any]] = None,
    ) -> ReasoningOutput:
        """Query Ollama with the behavior event context."""
        species_name = event.species.name
        system_prompt = self._build_system_prompt(species_name)

        event_payload = {
            "species": event.species.name,
            "scientific_name": event.species.scientific_name,
            "behavior_category": event.behavior.category,
            "behavior_label": event.behavior.label,
            "confidence": round(event.behavior.confidence, 3),
            "subjects_involved": len(event.subjects),
            "duration_seconds": event.temporal.duration,
            "is_uncertain": event.behavior.is_uncertain,
        }

        prompt = (
            f"Analyze this detected animal behavior event:\n"
            f"```json\n{json.dumps(event_payload, indent=2)}\n```\n\n"
            "Explain the ethological significance of this behavior, whether it indicates distress, reproduction, or normal maintenance, "
            "and suggest relevant management actions."
        )

        encoded_images = []
        if self.is_vision_capable and frames:
            for frame in frames[:3]:  # Max 3 frames
                b64 = self._convert_frame_to_base64(frame)
                if b64:
                    encoded_images.append(b64)

        try:
            response_text = await self.client.generate(
                model=self.model_name,
                prompt=prompt,
                images=encoded_images if encoded_images else None,
                system_prompt=system_prompt,
            )
            default_summary = (
                f"Observed {event.behavior.category} ({event.behavior.label}) with {event.behavior.confidence:.1%} confidence."
            )
            return self._parse_llm_response(response_text, default_summary)
        except ReasoningError as e:
            # Provide graceful degradation with error detail in explanation
            return ReasoningOutput(
                provider=self.provider_name,
                model=self.model_name,
                summary=f"Observed {event.behavior.category} ({event.behavior.label}) - LLM reasoning offline.",
                explanation=f"Reasoning model '{self.model_name}' could not be reached: {e}",
                recommendations=[],
                raw_response=None,
            )

    async def summarize_events(
        self,
        events: List[BehaviorEvent],
        context: Optional[str] = None,
    ) -> ReasoningOutput:
        """Summarize a sequence of events across a video or recording session."""
        if not events:
            return ReasoningOutput(
                provider=self.provider_name,
                model=self.model_name,
                summary="No behavior events recorded.",
                explanation=None,
                recommendations=[],
            )

        species_name = events[0].species.name
        system_prompt = self._build_system_prompt(species_name)

        event_summaries = [
            {
                "time": f"{e.temporal.start:.1f}s-{e.temporal.end:.1f}s",
                "behavior": f"{e.behavior.category}.{e.behavior.label}",
                "confidence": round(e.behavior.confidence, 2),
                "subjects": len(e.subjects),
            }
            for e in events
        ]

        prompt = (
            f"Here is the behavior timeline of {len(events)} events for {species_name}:\n"
            f"```json\n{json.dumps(event_summaries, indent=2)}\n```\n"
            f"Context: {context or 'Standard tank observation session'}\n\n"
            "Provide an executive ethological summary of the animal activity, social dynamics, and overall health status."
        )

        try:
            response_text = await self.client.generate(
                model=self.model_name,
                prompt=prompt,
                system_prompt=system_prompt,
            )
            default_summary = f"Aggregated {len(events)} events for {species_name}."
            return self._parse_llm_response(response_text, default_summary)
        except ReasoningError as e:
            return ReasoningOutput(
                provider=self.provider_name,
                model=self.model_name,
                summary=f"Processed {len(events)} events for {species_name}.",
                explanation=f"Ollama reasoning offline: {e}",
                recommendations=[],
            )

    async def ask_question(
        self,
        question: str,
        events: List[BehaviorEvent],
    ) -> str:
        """Answer a conversational question about the observed behavior session."""
        species_name = events[0].species.name if events else "animal"
        system_prompt = self._build_system_prompt(species_name)

        event_data = [
            f"[{e.temporal.start:.1f}s] {e.behavior.category}.{e.behavior.label} (conf: {e.behavior.confidence:.2f})"
            for e in events[:50]
        ]
        context_str = "\n".join(event_data)

        prompt = (
            f"Observed Events Log:\n{context_str}\n\n"
            f"User Question: {question}\n\n"
            "Answer the question directly based on the ethological data observed."
        )

        try:
            return await self.client.generate(
                model=self.model_name,
                prompt=prompt,
                system_prompt=system_prompt,
            )
        except ReasoningError as e:
            return f"Unable to query reasoning provider '{self.model_name}': {e}"
