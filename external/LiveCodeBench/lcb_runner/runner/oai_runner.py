import os
from time import sleep

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    pass

from lcb_runner.lm_styles import LMStyle
from lcb_runner.runner.base_runner import BaseRunner


class OpenAIRunner(BaseRunner):
    client = OpenAI(
        api_key=os.getenv("OPENAI_KEY"),
    )

    def __init__(self, args, model) -> None:
        super().__init__(args, model)

        if model.model_style == LMStyle.OpenAIReasonPreview:
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "max_completion_tokens": 25000,
            }
        elif model.model_style == LMStyle.OpenAIReason:
            assert "__" in args.model, (
                f"Model {args.model} is not a valid OpenAI Reasoning model as we require reasoning effort in model name."
            )
            model, reasoning_effort = args.model.split("__")
            self.client_kwargs: dict[str | str] = {
                "model": model,
                "reasoning_effort": reasoning_effort,
            }
        elif model.model_name.startswith("gpt-5"):
            # GPT-5 models use the responses API with different parameters
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "reasoning": {"effort": "minimal"},
                "text": {"verbosity": "low"},
                "timeout": args.openai_timeout,
            }
        else:
            # Standard OpenAI chat models
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": args.n,
                "timeout": args.openai_timeout,
                # "stop": args.stop, --> stop is only used for base models currently
            }

    def _run_single(self, prompt) -> list[str]:
        assert isinstance(prompt, list)

        try:
            if self.model.model_name.startswith("gpt-5"):
                # Use responses API for GPT-5
                response = self.client.responses.create(
                    input=prompt,
                    **self.client_kwargs,
                )
                return [response.output_text]
            else:
                # Use standard chat completions API
                response = self.client.chat.completions.create(
                    messages=prompt,
                    **self.client_kwargs,
                )
                return [c.message.content for c in response.choices]
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError,
        ) as e:
            print("Exception: ", repr(e))
            print("Sleeping for 30 seconds...")
            print("Consider reducing the number of parallel processes.")
            sleep(30)
            return self._run_single(prompt)
        except Exception as e:
            print(f"Failed to run the model for {prompt}!")
            print("Exception: ", repr(e))
            raise e
