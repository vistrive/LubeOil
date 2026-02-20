"""LLM provider router - factory for selecting and managing providers."""

import structlog

from lobp.core.config import settings
from lobp.llm.base import LLMProvider
from lobp.llm.openai_provider import OpenAIProvider
from lobp.llm.anthropic_provider import AnthropicProvider
from lobp.llm.ollama_provider import OllamaProvider
from lobp.llm.sarvam_provider import SarvamProvider

logger = structlog.get_logger()

# Singleton cache
_providers: dict[str, LLMProvider] = {}


def _init_providers() -> None:
    """Initialize all configured providers."""
    global _providers

    _providers["openai"] = OpenAIProvider(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )
    _providers["anthropic"] = AnthropicProvider(
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
    )
    _providers["ollama"] = OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )
    _providers["sarvam"] = SarvamProvider(
        api_key=settings.sarvam_api_key,
        base_url=settings.sarvam_base_url,
        model=settings.sarvam_model,
    )


def get_llm_provider(name: str | None = None) -> LLMProvider:
    """Get an LLM provider by name. Falls back to default."""
    if not _providers:
        _init_providers()

    provider_name = name or settings.default_llm_provider
    provider = _providers.get(provider_name)
    if not provider:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. "
            f"Available: {list(_providers.keys())}"
        )
    return provider


def get_all_providers() -> dict[str, dict]:
    """Get status of all configured providers."""
    if not _providers:
        _init_providers()

    return {
        name: {
            "provider": name,
            "available": p.is_available(),
            "model": getattr(p, "model", "unknown"),
        }
        for name, p in _providers.items()
    }
