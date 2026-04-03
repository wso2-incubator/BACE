"""OpenAI LLM client implementations."""

from typing import TYPE_CHECKING, Any, Optional, Union

from loguru import logger

from .base import LLMClient

if TYPE_CHECKING:
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.responses.response import Response
    from openai.types.responses.response_input_param import ResponseInputParam


class OpenAIChatClient(LLMClient):
    """OpenAI Client using the Chat Completions API."""

    def __init__(
        self,
        model: str,
        max_output_tokens: Optional[int] = None,
        enable_token_limit: bool = True,
        workers: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, max_output_tokens, enable_token_limit, workers=workers)
        from openai import OpenAI

        self.client = OpenAI(**kwargs)
        logger.debug(f"Initialized OpenAIChatClient with model: {model}")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        logger.debug(f"OpenAIChatClient generating with model: {self.model}")
        logger.trace(f"Prompt (first 200 chars): {prompt[:200]}...")

        response: ChatCompletion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        content = response.choices[0].message.content
        if content is None:
            logger.error("OpenAI response content is None")
            raise ValueError("OpenAI response content is None")

        result = str(content)
        logger.debug(f"Generated {len(result)} characters")
        logger.trace(f"Response (first 200 chars): {result[:200]}...")

        # Track usage
        usage = getattr(response, "usage", None)
        if usage:
            self._add_input_tokens(getattr(usage, "prompt_tokens", 0))
            self._add_output_tokens(getattr(usage, "completion_tokens", 0))
        else:
            self._add_input_tokens(self._estimate_tokens(prompt))
            self._add_output_tokens(self._estimate_tokens(result))

        return result


class OpenAIClient(LLMClient):
    """OpenAI Client using the Response API."""

    def __init__(
        self,
        model: str,
        max_output_tokens: Optional[int] = None,
        enable_token_limit: bool = True,
        reasoning_effort: str | None = "minimal",
        workers: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, max_output_tokens, enable_token_limit, workers=workers)
        from openai import OpenAI

        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(**kwargs)
        logger.debug(
            f"Initialized OpenAIClient with model: {model}, "
            f"reasoning_effort: {reasoning_effort}"
        )

    def _add_openai_tokens(self, response: "Response") -> None:
        """Add input and output tokens from OpenAI Response API response.

        Args:
            response: The OpenAI Response API response object.
        """
        if hasattr(response, "usage") and response.usage is not None:
            # Input tokens
            if hasattr(response.usage, "input_tokens"):
                self._add_input_tokens(response.usage.input_tokens)

            # Output tokens
            if hasattr(response.usage, "output_tokens"):
                self._add_output_tokens(response.usage.output_tokens)
        else:
            logger.warning(
                "Response object does not have usage information; "
                "using approximate character-based estimation instead."
            )

            # Note: We don't have the original prompt here easily,
            # but usually this method is called after generation.
            self._add_output_tokens(self._estimate_tokens(response.output_text))

    def generate(self, prompt: Union[str, "ResponseInputParam"], **kwargs: Any) -> str:
        # Allow overriding reasoning effort for this specific call
        reasoning_effort = kwargs.pop("reasoning_effort", self.reasoning_effort)

        logger.debug(
            f"OpenAIClient generating with model: {self.model}, "
            f"reasoning_effort: {reasoning_effort}"
        )
        logger.trace(f"Prompt (first 200 chars): {prompt[:200]}...")

        response: "Response" = self.client.responses.create(
            model=self.model,
            input=prompt,
            reasoning={"effort": reasoning_effort},
            **kwargs,
        )

        content = response.output_text if hasattr(response, "output_text") else None

        if content is None:
            logger.error("OpenAI Response API response content is None")
            raise ValueError("OpenAI Codex response content is None")

        result = str(content)
        logger.debug(f"Generated {len(result)} characters")
        logger.trace(f"Response (first 200 chars): {result[:200]}...")

        self._add_openai_tokens(response)
        return result

    def set_reasoning_effort(self, reasoning_effort: str) -> None:
        """Set the reasoning effort level for future generations.

        Args:
            reasoning_effort: The reasoning effort level to use.
                            Common values: 'minimal', 'low', 'medium', 'high'
        """
        old_effort = self.reasoning_effort
        self.reasoning_effort = reasoning_effort
        logger.info(f"Updated reasoning effort: {old_effort} → {reasoning_effort}")

    def get_reasoning_effort(self) -> str | None:
        """Get the current reasoning effort level.

        Returns:
            The current reasoning effort level.
        """
        return self.reasoning_effort
