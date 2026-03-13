"""UnittestMutationOperator — mutates a single test parent via LLM."""

from __future__ import annotations

from loguru import logger

from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import (
    OPERATION_MUTATION,
    CoevolutionContext,
)
from coevolution.core.interfaces.language import (
    LanguageParsingError,
    LanguageTransformationError,
)

from coevolution.strategies.llm_base import (
    BaseLLMOperator,
    LLMGenerationError,
    llm_retry,
)
from ._helpers import _TestLLMHelpers


class UnittestMutationOperator(_TestLLMHelpers, BaseLLMOperator[TestIndividual]):
    """Mutation: select one parent test → LLM rephrase → new TestIndividual."""

    def operation_name(self) -> str:
        return OPERATION_MUTATION

    @llm_retry(
        (
            ValueError,
            LanguageParsingError,
            LanguageTransformationError,
            LLMGenerationError,
        )
    )
    def execute(self, context: CoevolutionContext) -> list[TestIndividual]:
        test_pop = context.test_populations["unittest"]
        parents = self.parent_selector.select_parents(test_pop, 1, context)
        if not parents:
            logger.warning("UnittestMutationOperator: no parents available")
            return []
        parent = parents[0]

        prompt = self.prompt_manager.render_prompt(
            "operators/unittest/mutate.j2",
            question_content=context.problem.question_content,
            individual=parent.snippet,
        )
        response = self._generate(prompt)
        extracted = self._extract_code_block(response)
        clean_block = self.parser.remove_main_block(extracted)
        mutated = self._extract_first_test_function(clean_block)

        probability = self.prob_assigner.assign_probability(
            OPERATION_MUTATION, [parent.probability]
        )
        return [
            TestIndividual(
                snippet=mutated,
                probability=probability,
                creation_op=OPERATION_MUTATION,
                generation_born=test_pop.generation + 1,
                parents={"code": [], "test": [parent.id]},
            )
        ]


__all__ = ["UnittestMutationOperator"]
