"""CodeEditOperator — guides code repair using failing tests."""

from __future__ import annotations

from typing import Protocol

from loguru import logger

from coevolution.core.individual import CodeIndividual, TestIndividual
from coevolution.core.interfaces import (
    OPERATION_EDIT,
    CoevolutionContext,
    LanguageParsingError,
    LanguageTransformationError,
)
from coevolution.core.interfaces.language import ILanguage
from coevolution.core.interfaces.probability import IProbabilityAssigner
from coevolution.core.interfaces.selection import IParentSelectionStrategy
from coevolution.strategies.llm_base import (
    BaseLLMOperator,
    ILanguageModel,
    LLMGenerationError,
    llm_retry,
)

from ._helpers import _CodeLLMHelpers

type TestPopulationType = str


class IFailingTestSelector(Protocol):
    """Protocol for selecting failing tests for code individuals."""

    @staticmethod
    def select_k_failing_tests(
        coevolution_context: CoevolutionContext,
        code_individual: CodeIndividual,
        k: int = 10,
    ) -> list[tuple[TestIndividual, TestPopulationType]]: ...


class CodeEditOperator(_CodeLLMHelpers, BaseLLMOperator[CodeIndividual]):
    """Edit: select parent + failing tests → targeted LLM fix → new CodeIndividual."""

    def __init__(
        self,
        llm: ILanguageModel,
        language_adapter: ILanguage,
        parent_selector: IParentSelectionStrategy[CodeIndividual],
        prob_assigner: IProbabilityAssigner,
        failing_test_selector: IFailingTestSelector,
        k_failing_tests: int = 10,
    ) -> None:
        super().__init__(llm, language_adapter, parent_selector, prob_assigner)
        self.failing_test_selector = failing_test_selector
        self.k_failing_tests = k_failing_tests

    def operation_name(self) -> str:
        return OPERATION_EDIT

    @llm_retry(
        (
            ValueError,
            LanguageParsingError,
            LanguageTransformationError,
            LLMGenerationError,
        )
    )
    def execute(self, context: CoevolutionContext) -> list[CodeIndividual]:
        code_pop = context.code_population
        problem = context.problem

        parents = self.parent_selector.select_parents(code_pop, 1, context)
        if not parents:
            logger.warning("CodeEditOperator: no parents available")
            return []
        parent = parents[0]

        failing = self.failing_test_selector.select_k_failing_tests(
            context, parent, k=self.k_failing_tests
        )
        if not failing:
            logger.debug("CodeEditOperator: no failing tests for this parent, skipping")
            return []

        feedback_parts = []
        for idx, (test_ind, _pop_type) in enumerate(failing, start=1):
            exec_result = context.interactions["unittest"].execution_results
            trace = "No trace available"
            if parent.id in exec_result and test_ind.id in exec_result[parent.id]:
                trace = exec_result[parent.id][test_ind.id].error_log or trace
            feedback_parts.append(
                f"Failing Test #{idx}:\n{test_ind.snippet}\n\nError Trace:\n{trace}"
            )
        feedback = "\n\n" + "=" * 80 + "\n\n".join(feedback_parts)

        prompt = self.prompt_manager.render_prompt(
            "operators/code/edit_multiple.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
            individual=parent.snippet,
            feedback=feedback,
        )
        response = self._generate(prompt)
        edited_code = self._extract_code_block(response)
        edited_code = self._validated_code(edited_code, problem.starter_code, "edit")

        probability = self.prob_assigner.assign_probability(
            OPERATION_EDIT, [parent.probability]
        )
        return [
            CodeIndividual(
                snippet=edited_code,
                probability=probability,
                creation_op=OPERATION_EDIT,
                generation_born=code_pop.generation + 1,
                parents={
                    "code": [parent.id],
                    "test": [t.id for t, _ in failing],
                },
                metadata={"num_failing_tests": len(failing)},
            )
        ]


__all__ = ["CodeEditOperator", "IFailingTestSelector"]
__all__ = ["CodeEditOperator", "IFailingTestSelector"]
