"""
Factory function to construct reasoning providers.
"""
from __future__ import annotations

from typing import Optional, Union
from animallens.reasoning.base import BaseReasoningProvider, NoOpReasoningProvider
from animallens.reasoning.ollama import OllamaReasoningProvider


def get_reasoning_provider(
    reasoning: Optional[Union[str, BaseReasoningProvider]] = None,
    base_url: Optional[str] = None,
) -> BaseReasoningProvider:
    """
    Construct or return a reasoning provider instance based on specification.

    Examples:
        get_reasoning_provider(None) -> NoOpReasoningProvider
        get_reasoning_provider("ollama:gemma3") -> OllamaReasoningProvider(model_name="gemma3")
        get_reasoning_provider("ollama:qwen3") -> OllamaReasoningProvider(model_name="qwen3")
        get_reasoning_provider("llama3.2:3b") -> OllamaReasoningProvider(model_name="llama3.2:3b")
    """
    if reasoning is None or reasoning == "none" or reasoning is False:
        return NoOpReasoningProvider()

    if isinstance(reasoning, BaseReasoningProvider):
        return reasoning

    if isinstance(reasoning, str):
        spec = reasoning.strip().lower()
        if spec in ("", "none", "null", "false"):
            return NoOpReasoningProvider()

        if spec.startswith("ollama:"):
            model_name = spec.split(":", 1)[1]
            return OllamaReasoningProvider(model_name=model_name, base_url=base_url)
        elif ":" in spec:
            # e.g. "gemma3:12b" or "qwen:8b"
            return OllamaReasoningProvider(model_name=reasoning, base_url=base_url)
        else:
            # Default single name to Ollama model
            return OllamaReasoningProvider(model_name=reasoning, base_url=base_url)

    return NoOpReasoningProvider()
