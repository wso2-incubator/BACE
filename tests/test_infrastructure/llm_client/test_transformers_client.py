import sys
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Union

import pytest

from infrastructure.llm_client.exceptions import LLMInputFormatError
from infrastructure.llm_client.transformers import TransformersClient


def _make_fake_transformers() -> SimpleNamespace:
    """Return a fake `transformers` module with a `pipeline` factory."""

    def pipeline(
        task: str, model: Optional[str] = None
    ) -> Callable[[Union[str, List[Dict[str, str]]], Any], List[Dict[str, str]]]:
        # pipeline returns a callable that accepts (messages, **kwargs)
        def pipe(
            messages: Union[str, List[Dict[str, str]]], *args: Any, **kwargs: Any
        ) -> List[Dict[str, str]]:
            # Return a predictable generated_text based on input for assertions
            if isinstance(messages, list) and messages:
                content = messages[0].get("content", str(messages[0]))
            else:
                content = str(messages)
            return [{"generated_text": [{"role": "assistant", "content": f"fake-gen: {content}"}]}]

        return pipe

    mod = SimpleNamespace()
    mod.pipeline = pipeline
    return mod


def test_generate_with_str(monkeypatch: pytest.MonkeyPatch) -> None:
    fake: SimpleNamespace = _make_fake_transformers()
    monkeypatch.setitem(sys.modules, "transformers", fake)

    client: TransformersClient = TransformersClient(
        "didula-wso2/exp_23_emb_grpo_checkpoint_220_16bit_vllm"
    )
    out: str = client.generate("hello world")
    assert isinstance(out, str)
    assert out == "fake-gen: hello world"


def test_generate_with_list(monkeypatch: pytest.MonkeyPatch) -> None:
    fake: SimpleNamespace = _make_fake_transformers()
    monkeypatch.setitem(sys.modules, "transformers", fake)

    client: TransformersClient = TransformersClient(
        "didula-wso2/exp_23_emb_grpo_checkpoint_220_16bit_vllm"
    )
    messages: List[Dict[str, str]] = [{"role": "user", "content": "list prompt"}]
    out: str = client.generate(messages)
    assert out == "fake-gen: list prompt"


def test_generate_bad_type_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fake: SimpleNamespace = _make_fake_transformers()
    monkeypatch.setitem(sys.modules, "transformers", fake)

    client: TransformersClient = TransformersClient(
        "didula-wso2/exp_23_emb_grpo_checkpoint_220_16bit_vllm"
    )
    with pytest.raises(LLMInputFormatError):
        client.generate(123)  # type: ignore[arg-type]
