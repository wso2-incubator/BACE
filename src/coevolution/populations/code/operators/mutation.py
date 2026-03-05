"""CodeMutationOperator — mutates a single code parent via LLM."""

from __future__ import annotations

from loguru import logger

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import (
    OPERATION_MUTATION,
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


class CodeMutationOperator(_CodeLLMHelpers, BaseLLMOperator[CodeIndividual]):
    """Mutation: select one parent → LLM rephrase → new CodeIndividual."""

    def operation_name(self) -> str:
        return OPERATION_MUTATION

    @llm_retry((ValueError, CodeParsingError, CodeTransformationError, LLMGenerationError))
    def execute(self, context: CoevolutionContext) -> list[CodeIndividual]:
        code_pop = context.code_population
        problem = context.problem

        parents = self.parent_selector.select_parents(code_pop, 1, context)
        if not parents:
            logger.warning("CodeMutationOperator: no parents available")
            return []
        parent = parents[0]

        prompt = self.prompt_manager.render_prompt(
            "operators/code/mutate.j2",
            question_content=problem.question_content,
            individual=parent.snippet,
            starter_code=problem.starter_code,
        )
        response = self._generate(prompt)
        mutated_code = self._extract_code_block(response)
        mutated_code = self._validated_code(mutated_code, problem.starter_code, "mutation")

        probability = self.prob_assigner.assign_probability(
            OPERATION_MUTATION, [parent.probability]
        )
        return [
            CodeIndividual(
                snippet=mutated_code,
                probability=probability,
                creation_op=OPERATION_MUTATION,
                generation_born=code_pop.generation + 1,
                parents={"code": [parent.id], "test": []},
            )
        ]


__all__ = ["CodeMutationOperator"]
