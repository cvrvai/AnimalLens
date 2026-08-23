"""
Reasoning provider module for AnimalLens (Layer B).
"""
from animallens.reasoning.base import BaseReasoningProvider, NoOpReasoningProvider
from animallens.reasoning.ollama import OllamaClient, OllamaReasoningProvider
from animallens.reasoning.factory import get_reasoning_provider

__all__ = [
    "BaseReasoningProvider",
    "NoOpReasoningProvider",
    "OllamaClient",
    "OllamaReasoningProvider",
    "get_reasoning_provider",
]
