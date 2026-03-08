"""AgentCoderEditOperator — stateful single-agent loop edit operator."""

from __future__ import annotations

from loguru import logger

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import (
    OPERATION_EDIT,
    CoevolutionContext,
)
from coevolution.core.interfaces.language import (
    ICodeParser,
    LanguageParsingError,
    LanguageTransformationError,
)
from coevolution.core.interfaces.probability import IProbabilityAssigner
from coevolution.core.interfaces.selection import IParentSelectionStrategy

from coevolution.strategies.llm_base import (
    BaseLLMOperator,
    ILanguageModel,
    LLMGenerationError,
    llm_retry,
)


class AgentCoderEditOperator(BaseLLMOperator[CodeIndividual]):
    """Stateful edit operator for the AgentCoder single-agent loop.

    Maintains conversation history across calls within one problem.
    Call reset_session() between problems.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        parent_selector: IParentSelectionStrategy[CodeIndividual],
        prob_assigner: IProbabilityAssigner,
    ) -> None:
        super().__init__(llm, parser, language_name, parent_selector, prob_assigner)
        self._conversation_history: list[dict[str, str]] = []

    def reset_session(self) -> None:
        """Wipe conversation memory. Call between problems."""
        logger.info("AgentCoderEditOperator: session reset")
        self._conversation_history.clear()

    def operation_name(self) -> str:
        return OPERATION_EDIT

    @llm_retry((ValueError, LanguageParsingError, LanguageTransformationError, LLMGenerationError))
    def execute(self, context: CoevolutionContext) -> list[CodeIndividual]:
        code_pop = context.code_population
        problem = context.problem

        if len(code_pop) != 1:
            raise ValueError(
                f"AgentCoderEditOperator requires exactly 1 individual, got {len(code_pop)}"
            )
        code_parent = code_pop[0]

        unittest_exec = context.interactions["unittest"].execution_results
        if code_parent.id not in unittest_exec:
            logger.warning(f"No execution results for {code_parent.id}")
            return [code_parent]

        test_results = unittest_exec[code_parent.id]
        unittest_pop = context.test_populations["unittest"]

        failing = [
            (t.snippet, test_results[t.id].error_log or "")
            for t in unittest_pop
            if test_results[t.id].status == "failed"
        ]

        if not failing:
            logger.info("AgentCoderEditOperator: no failing tests — returning parent unchanged")
            return [code_parent]

        if not self._conversation_history:
            raise ValueError(
                "AgentCoderEditOperator: no conversation history. "
                "Run AgentCoderInitializer first, or call reset_session()."
            )

        feedback = "\n\n".join(
            f"Test Case:\n{test}\n\nError Trace:\n{trace}" for test, trace in failing
        )
        prompt = self.prompt_manager.render_prompt(
            "operators/agent_coder/edit.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
            feedback=feedback,
        )
        self._conversation_history.append({"role": "user", "content": prompt})

        logger.debug(f"AgentCoder: repairing (history depth: {len(self._conversation_history)})")
        response = self._generate(self._conversation_history)
        self._conversation_history.append({"role": "assistant", "content": response})

        edited_code = self._extract_code_block(response)
        if not self.parser.contains_starter_code(edited_code, problem.starter_code):
            raise ValueError("AgentCoder edit result does not contain starter code")

        probability = self.prob_assigner.assign_probability(
            OPERATION_EDIT, [code_parent.probability]
        )
        return [
            CodeIndividual(
                snippet=edited_code,
                probability=probability,
                creation_op=OPERATION_EDIT,
                generation_born=code_pop.generation + 1,
                parents={
                    "code": [code_parent.id],
                    "test": [
                        t.id
                        for t in unittest_pop
                        if test_results[t.id].status == "failed"
                    ],
                },
            )
        ]


__all__ = ["AgentCoderEditOperator"]
