# src/coevolution/operators/agent_coder_llm_operator.py

from dataclasses import dataclass
from typing import Any, Dict, List, Set

from loguru import logger

from coevolution.core.interfaces import (
    BaseOperatorInput,
    InitialInput,
    IOperator,
    OperatorOutput,
    OperatorResult,
)
from coevolution.core.interfaces.language import (
    LanguageParsingError,
    LanguageTransformationError,
)
from coevolution.utils.prompts import (
    AGENT_CODER_PROGRAMMER_EDIT,
    AGENT_CODER_PROGRAMMER_INIT,
)

from .base_llm_operator import BaseLLMOperator, UnsupportedOperatorInput, llm_retry


@dataclass(frozen=True)
class AgentCoderEditInput(BaseOperatorInput):
    """
    Input for the stateful AgentCoder edit loop.
    No history needed here; the operator holds it.
    """

    current_snippet: str
    failing_tests_with_trace: list[tuple[str, str]]  # List of (test_case, error_trace)
    starter_code: str


class AgentCoderLLMOperator(BaseLLMOperator, IOperator):
    """
    Stateful AgentCoder Operator.

    Acts as a persistent developer session.
    - Turn 0: Initialization (generate_initial_snippets)
    - Turn N: Iterative Repair (apply -> edit)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_history: List[Dict[str, str]] = []

    def reset_session(self) -> None:
        """Wipes memory. Call this when switching problems."""
        logger.info("AgentCoder: Memory wiped. Starting fresh session.")
        self._conversation_history.clear()

    def supported_operations(self) -> Set[str]:
        return {"edit"}

    def _contains_starter_code(self, code: str, starter_code: str) -> bool:
        return self.language_adapter.contains_starter_code(code, starter_code)

    # -------------------------------------------------------------------------
    # TURN 0: INITIALIZATION
    # -------------------------------------------------------------------------
    @llm_retry((ValueError, LanguageParsingError, LanguageTransformationError))
    def generate_initial_snippets(self, input_dto: InitialInput) -> OperatorOutput:
        """
        Generates the first solution AND seeds the conversation history.
        """
        if input_dto.population_size != 1:
            raise ValueError(
                "AgentCoder only supports population_size=1 (Linear Mode)."
            )

        # 1. Clear previous session to be safe
        self.reset_session()

        # 2. Construct Prompt (Same as CodeLLMOperator)
        prompt = AGENT_CODER_PROGRAMMER_INIT.format(
            question_content=input_dto.question_content,
            starter_code=input_dto.starter_code,
        )

        # 3. Add to History
        self._conversation_history.append({"role": "user", "content": prompt})

        logger.debug("AgentCoder: Generating initial solution...")

        # 4. Generate
        response = self._generate(self._conversation_history)

        # 5. Update History
        self._conversation_history.append({"role": "assistant", "content": response})

        # 6. Extract & Return, Psuedocode will be in the first block, so take the last
        code = self.language_adapter.extract_code_blocks(response)[-1]

        # Validation
        if not self._contains_starter_code(code, input_dto.starter_code):
            raise ValueError("Generated code missing starter code.")

        results = [OperatorResult(snippet=code, metadata={"operation": "initial"})]
        return OperatorOutput(results=results)

    # -------------------------------------------------------------------------
    # TURN N: ITERATIVE REPAIR (EDIT)
    # -------------------------------------------------------------------------
    @llm_retry((ValueError, LanguageParsingError, LanguageTransformationError))
    def _handle_edit(self, input_dto: AgentCoderEditInput) -> OperatorOutput:
        # Safety: Ensure we have a session. If not, we can't edit "nothing".
        if not self._conversation_history:
            raise ValueError(
                "No existing session found. Cannot perform edit without initialization."
            )

        # 1. Construct Feedback (Same prompt as CodeLLMOperator)
        # We treat the feedback as the next User Message in the chat.
        feedback = "\n\n".join(
            f"Test Case:\n{test}\n\nError Trace:\n{trace}"
            for test, trace in input_dto.failing_tests_with_trace
        )
        prompt = AGENT_CODER_PROGRAMMER_EDIT.format(
            question_content=input_dto.question_content,
            starter_code=input_dto.starter_code,
            feedback=feedback,
        )

        # 2. Add to History
        self._conversation_history.append({"role": "user", "content": prompt})

        logger.debug(
            f"AgentCoder: Repairing... (History Depth: {len(self._conversation_history)})"
        )

        # 3. Generate with Context
        response = self._generate(self._conversation_history)

        # 4. Update History
        self._conversation_history.append({"role": "assistant", "content": response})

        # 5. Extract
        edited_code = self._extract_code_block(response)

        # 6. Validation
        if not self._contains_starter_code(edited_code, input_dto.starter_code):
            logger.warning("AgentCoder response missing starter code.")
            raise ValueError("Edit result does not contain starter code structure.")

        return OperatorOutput(
            results=[
                OperatorResult(snippet=edited_code, metadata={"operation": "edit"})
            ]
        )

    def apply(self, input_dto: BaseOperatorInput) -> OperatorOutput:
        match input_dto:
            case AgentCoderEditInput():
                return self._handle_edit(input_dto)
            case _:
                raise UnsupportedOperatorInput(
                    type(input_dto), getattr(input_dto, "operation", None)
                )


__all__ = ["AgentCoderEditInput", "AgentCoderLLMOperator"]
