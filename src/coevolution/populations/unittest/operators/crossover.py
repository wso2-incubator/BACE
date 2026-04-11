"""UnittestCrossoverOperator — combines two test parents via LLM."""

from __future__ import annotations

from loguru import logger

from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import (
    OPERATION_CROSSOVER,
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


class UnittestCrossoverOperator(_TestLLMHelpers, BaseLLMOperator[TestIndividual]):
    """Crossover: combine two parent tests → new TestIndividual."""

    def operation_name(self) -> str:
        return OPERATION_CROSSOVER

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
        parents = self.parent_selector.select_parents(test_pop, 2, context)
        if len(parents) < 2:
            logger.warning("UnittestCrossoverOperator: need 2 parents, got fewer")
            return []
        p1, p2 = parents[0], parents[1]

        prompt = self.prompt_manager.render_prompt(
            "operators/unittest/crossover.j2",
            question_content=context.problem.question_content,
            parent1=p1.snippet,
            parent2=p2.snippet,
        )
        response = self._generate(prompt)
        extracted = self._extract_code_block(response)
        clean_block = self.parser.remove_main_block(extracted)
        child = self._extract_first_test_function(clean_block)

        probability = self.prob_assigner.assign_probability(
            OPERATION_CROSSOVER, [p1.probability, p2.probability]
        )
        return [
            TestIndividual(
                snippet=child,
                probability=probability,
                creation_op=OPERATION_CROSSOVER,
                generation_born=test_pop.generation + 1,
                parents={"code": [], "test": [p1.id, p2.id]},
                explanation=self.parser.get_docstring(child),
            )
        ]


__all__ = ["UnittestCrossoverOperator"]
