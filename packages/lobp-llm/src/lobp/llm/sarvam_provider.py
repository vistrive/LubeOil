"""Sarvam.ai LLM provider for Indian language and multilingual support."""

import httpx
import structlog

from lobp.llm.base import LLMProvider, LLMResponse

logger = structlog.get_logger()


class SarvamProvider(LLMProvider):
    """Sarvam.ai provider for multilingual and Indic language models."""

    provider_name = "sarvam"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.sarvam.ai",
        model: str = "sarvam-2b-v0.5",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            content = ""
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")

            return LLMResponse(
                content=content,
                provider=self.provider_name,
                model=self.model,
                usage=data.get("usage", {}),
            )
        except Exception as e:
            logger.error("Sarvam API error", error=str(e))
            return LLMResponse(
                content="", provider=self.provider_name,
                model=self.model, error=str(e),
            )
