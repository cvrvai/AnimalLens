"""
Base abstraction for Layer B Reasoning Providers (Ollama, local/remote LLMs).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional
from animallens.core.schemas import BehaviorEvent, ReasoningOutput


class BaseReasoningProvider(ABC):
    """
    Abstract interface for reasoning providers.
    Reasoning models provide biological interpretation, recommendations, and summaries
    based on Layer A structured events.
    """

    def __init__(self, model_name: str = "none") -> None:
        self.model_name = model_name

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier (e.g. ollama, openai, local)."""
        pass

    @property
    def is_enabled(self) -> bool:
        """Whether this provider actively performs inference."""
        return True

    @property
    def is_vision_capable(self) -> bool:
        """Whether this model accepts image frames as input."""
        return False

    @abstractmethod
    async def explain_event(
        self,
        event: BehaviorEvent,
        frames: Optional[List[Any]] = None,
    ) -> ReasoningOutput:
        """Generate biological reasoning and advice for a single BehaviorEvent."""
        pass

    @abstractmethod
    async def summarize_events(
        self,
        events: List[BehaviorEvent],
        context: Optional[str] = None,
    ) -> ReasoningOutput:
        """Generate an aggregated summary for a list of BehaviorEvents."""
        pass

    @abstractmethod
    async def ask_question(
        self,
        question: str,
        events: List[BehaviorEvent],
    ) -> str:
        """Answer a conversational question grounded in observed behavior events."""
        pass


class NoOpReasoningProvider(BaseReasoningProvider):
    """Null provider when reasoning is disabled (reasoning=None)."""

    def __init__(self) -> None:
        super().__init__(model_name="none")

    @property
    def provider_name(self) -> str:
        return "none"

    @property
    def is_enabled(self) -> bool:
        return False

    async def explain_event(
        self,
        event: BehaviorEvent,
        frames: Optional[List[Any]] = None,
    ) -> ReasoningOutput:
        return ReasoningOutput(
            provider="none",
            model="none",
            summary=f"Observed {event.behavior.category} ({event.behavior.label}) with confidence {event.behavior.confidence:.2f}.",
            explanation=None,
            recommendations=[],
        )

    async def summarize_events(
        self,
        events: List[BehaviorEvent],
        context: Optional[str] = None,
    ) -> ReasoningOutput:
        count = len(events)
        return ReasoningOutput(
            provider="none",
            model="none",
            summary=f"Processed {count} behavior event(s).",
            explanation=None,
            recommendations=[],
        )

    async def ask_question(
        self,
        question: str,
        events: List[BehaviorEvent],
    ) -> str:
        return f"Reasoning provider is disabled. Observed {len(events)} event(s)."
