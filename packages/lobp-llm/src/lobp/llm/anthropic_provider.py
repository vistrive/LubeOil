"""Anthropic / Claude LLM provider."""

import httpx
import structlog

from lobp.llm.base import LLMProvider, LLMResponse

logger = structlog.get_logger()


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider (Sonnet, Opus, Haiku)."""

    provider_name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        body: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            body["system"] = system_prompt

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.base_url}/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()

            content = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    content += block["text"]

            return LLMResponse(
                content=content,
                provider=self.provider_name,
                model=self.model,
                usage=data.get("usage", {}),
            )
        except Exception as e:
            logger.error("Anthropic API error", error=str(e))
            return LLMResponse(
                content="", provider=self.provider_name,
                model=self.model, error=str(e),
            )
