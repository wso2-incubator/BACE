import ollama
from typing import Any, Dict, Optional


class LLMClient:
    def __init__(self, provider: str, model: str, **kwargs):
        self.provider = provider.lower()
        self._client = self._init_client(**kwargs)
        self.model = model

    def _init_client(self, **kwargs) -> Any:
        if self.provider == "openai":
            from openai import OpenAI
            return OpenAI()
        elif self.provider == "ollama":
            return None  # ollama uses module-level functions
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(self, prompt: str, **kwargs) -> str:
        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content
        elif self.provider == "ollama":
            response = ollama.chat(model=self.model, messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])
            return response['message']['content']
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
