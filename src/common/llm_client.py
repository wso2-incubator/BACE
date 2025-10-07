from typing import Any

import ollama


class LLMClient:
    def __init__(self, provider: str, model: str, **kwargs: Any) -> None:
        self.provider = provider.lower()
        self._client = self._init_client(**kwargs)
        self.model = model

    def _init_client(self, **kwargs: Any) -> Any:
        if self.provider == "openai":
            from openai import OpenAI

            return OpenAI()
        elif self.provider == "ollama":
            return None  # ollama uses module-level functions
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("OpenAI response content is None")
            # Ensure we return a string type
            return str(content)
        elif self.provider == "ollama":
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            content = response["message"]["content"]
            if not isinstance(content, str):
                raise ValueError(f"Expected string content, got {type(content)}")
            return content
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
