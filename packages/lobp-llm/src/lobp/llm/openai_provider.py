"""OpenAI / ChatGPT LLM provider."""

import httpx
import structlog

from lobp.llm.base import LLMProvider, LLMResponse

logger = structlog.get_logger()


class OpenAIProvider(LLMProvider):
    """OpenAI ChatGPT provider (GPT-4o, GPT-4, etc.)."""

    provider_name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"

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
                    f"{self.base_url}/chat/completions",
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

            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                provider=self.provider_name,
                model=self.model,
                usage=data.get("usage", {}),
            )
        except Exception as e:
            logger.error("OpenAI API error", error=str(e))
            return LLMResponse(
                content="", provider=self.provider_name,
                model=self.model, error=str(e),
            )
