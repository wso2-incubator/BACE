"""CodeCrossoverOperator — combines two code parents via LLM."""

from __future__ import annotations

from loguru import logger

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import (
    OPERATION_CROSSOVER,
    CoevolutionContext,
)
from infrastructure.code_preprocessing.exceptions import (
    CodeParsingError,
    CodeTransformationError,
)

from coevolution.strategies.llm_base import (
    BaseLLMOperator,
    LLMGenerationError,
    llm_retry,
)
from ._helpers import _CodeLLMHelpers


class CodeCrossoverOperator(_CodeLLMHelpers, BaseLLMOperator[CodeIndividual]):
    """Crossover: select two parents → LLM combine → new CodeIndividual."""

    def operation_name(self) -> str:
        return OPERATION_CROSSOVER

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError, LLMGenerationError))
    def execute(self, context: CoevolutionContext) -> list[CodeIndividual]:
        code_pop = context.code_population
        problem = context.problem

        parents = self.parent_selector.select_parents(code_pop, 2, context)
        if len(parents) < 2:
            logger.warning("CodeCrossoverOperator: need 2 parents, got fewer")
            return []
        p1, p2 = parents[0], parents[1]

        prompt = self.prompt_manager.render_prompt(
            "operators/code/crossover.j2",
            question_content=problem.question_content,
            parent1=p1.snippet,
            parent2=p2.snippet,
            starter_code=problem.starter_code,
        )
        response = self._generate(prompt)
        child_code = self._extract_code_block(response)
        child_code = self._validated_code(child_code, problem.starter_code, "crossover")

        probability = self.prob_assigner.assign_probability(
            OPERATION_CROSSOVER, [p1.probability, p2.probability]
        )
        return [
            CodeIndividual(
                snippet=child_code,
                probability=probability,
                creation_op=OPERATION_CROSSOVER,
                generation_born=code_pop.generation + 1,
                parents={"code": [p1.id, p2.id], "test": []},
            )
        ]


__all__ = ["CodeCrossoverOperator"]
