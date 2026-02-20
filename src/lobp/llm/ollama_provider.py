"""Ollama / Llama LLM provider for local inference."""

import httpx
import structlog

from lobp.llm.base import LLMProvider, LLMResponse

logger = structlog.get_logger()


class OllamaProvider(LLMProvider):
    """Ollama provider for local Llama models."""

    provider_name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def is_available(self) -> bool:
        try:
            import httpx as _httpx
            resp = _httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            return LLMResponse(
                content=data.get("response", ""),
                provider=self.provider_name,
                model=self.model,
                usage={
                    "eval_count": data.get("eval_count", 0),
                    "eval_duration": data.get("eval_duration", 0),
                },
            )
        except Exception as e:
            logger.error("Ollama API error", error=str(e))
            return LLMResponse(
                content="", provider=self.provider_name,
                model=self.model, error=str(e),
            )
