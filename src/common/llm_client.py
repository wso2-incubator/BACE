from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from the model."""
        pass


# ----------------------------
# OpenAI Client Implementation
# ----------------------------
class OpenAIClient(LLMClient):
    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model)
        from openai import OpenAI

        self.client = OpenAI(**kwargs)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI response content is None")
        return str(content)


# ----------------------------
# OpenAI Codex Client Implementation (using Response API)
# ----------------------------
class OpenAICodexClient(LLMClient):
    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model)
        from openai import OpenAI

        self.client = OpenAI(**kwargs)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            **kwargs,
        )
        # Extract content from response API format
        if hasattr(response, "output") and response.output:
            content = (
                response.output[0].content[0].text
                if response.output[0].content
                else None
            )
        else:
            content = None

        if content is None:
            raise ValueError("OpenAI Codex response content is None")
        return str(content)


# ----------------------------
# Ollama Client Implementation
# ----------------------------
class OllamaClient(LLMClient):
    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model)
        import ollama

        self.ollama = ollama

    def generate(self, prompt: str, **kwargs: Any) -> str:
        response = self.ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        content = response["message"]["content"]
        if not isinstance(content, str):
            raise ValueError(f"Expected string content, got {type(content)}")
        return content


# ----------------------------
# Factory Function (Optional)
# ----------------------------
def create_llm_client(provider: str, model: str, **kwargs: Any) -> LLMClient:
    provider = provider.lower()
    if provider == "openai":
        return OpenAIClient(model, **kwargs)
    elif provider == "openai-codex":
        return OpenAICodexClient(model, **kwargs)
    elif provider == "ollama":
        return OllamaClient(model, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
