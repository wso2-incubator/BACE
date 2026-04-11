"""UnittestEditOperator — improves test discriminating power using code context."""

from __future__ import annotations

import random

from loguru import logger

from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import (
    OPERATION_EDIT,
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


class UnittestEditOperator(_TestLLMHelpers, BaseLLMOperator[TestIndividual]):
    """Edit: improve a test's discriminating power using passing/failing code context.

    Three edit modes (auto-selected by what interaction data is available):
    - discriminating  : has passing AND failing code → edit to discriminate harder
    - all-failing     : no passing code → edit to be more lenient/correct
    - all-passing     : no failing code → edit to break one of the passing ones
    """

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
    def execute(self, context: CoevolutionContext) -> list[TestIndividual]:
        test_pop = context.test_populations["unittest"]
        code_pop = context.code_population
        interactions = context.interactions["unittest"]

        # 1. Select parent test
        parents = self.parent_selector.select_parents(test_pop, 1, context)
        if not parents:
            return []
        parent = parents[0]

        # 2. Look up which code individuals pass/fail this test
        parent_test_idx = test_pop.get_index_of_individual(parent)
        if parent_test_idx == -1:
            logger.error(f"Parent test ID {parent.id} not found in test pop index")
            return []

        if interactions.observation_matrix.size == 0:
            logger.debug("Interaction matrix empty — skipping edit")
            return []

        test_col = interactions.observation_matrix[:, parent_test_idx]
        passing_indices = random.sample(
            [i for i, r in enumerate(test_col) if r == 1],
            min(2, sum(1 for r in test_col if r == 1)),
        )
        failing_indices = random.sample(
            [i for i, r in enumerate(test_col) if r == 0],
            min(2, sum(1 for r in test_col if r == 0)),
        )

        if not passing_indices and not failing_indices:
            logger.debug(f"No code interaction data for test {parent.id}")
            return []

        passing_inds = [code_pop[i] for i in passing_indices]
        failing_inds = [code_pop[i] for i in failing_indices]

        error_traces = [
            interactions.execution_results[fi.id][parent.id].error_log
            or "No trace available"
            for fi in failing_inds
        ]

        # 3. Choose the right prompt
        if passing_inds and failing_inds:
            edit_type = "discriminating"
            prompt = self.prompt_manager.render_prompt(
                "operators/unittest/edit_discriminating.j2",
                question_content=context.problem.question_content,
                current_test_snippet=parent.snippet,
                passing_code_snippet=passing_inds[0].snippet,
                failing_code_snippet=failing_inds[0].snippet,
                failing_code_trace=error_traces[0],
            )
        elif not passing_inds:
            edit_type = "all-failing"
            if len(failing_inds) < 2:
                logger.debug("Not enough failing inds for all-failing edit")
                return []
            prompt = self.prompt_manager.render_prompt(
                "operators/unittest/edit_all_failing.j2",
                question_content=context.problem.question_content,
                current_test_snippet=parent.snippet,
                failing_code_snippet_P=failing_inds[0].snippet,
                failing_code_trace_P=error_traces[0],
                failing_code_snippet_Q=failing_inds[1].snippet,
                failing_code_trace_Q=error_traces[1],
            )
        else:
            edit_type = "all-passing"
            if len(passing_inds) < 2:
                logger.debug("Not enough passing inds for all-passing edit")
                return []
            prompt = self.prompt_manager.render_prompt(
                "operators/unittest/edit_all_passing.j2",
                question_content=context.problem.question_content,
                current_test_snippet=parent.snippet,
                passing_code_snippet_P=passing_inds[0].snippet,
                passing_code_snippet_Q=passing_inds[1].snippet,
            )

        logger.debug(f"UnittestEditOperator: using '{edit_type}' edit mode")
        response = self._generate(prompt)
        extracted = self._extract_code_block(response)
        clean_block = self.parser.remove_main_block(extracted)
        edited = self._extract_first_test_function(clean_block)

        probability = self.prob_assigner.assign_probability(
            OPERATION_EDIT, [parent.probability]
        )
        return [
            TestIndividual(
                snippet=edited,
                probability=probability,
                creation_op=OPERATION_EDIT,
                generation_born=test_pop.generation + 1,
                parents={
                    "code": [ind.id for ind in passing_inds + failing_inds],
                    "test": [parent.id],
                },
                explanation=self.parser.get_docstring(edited),
                metadata={"edit_type": edit_type},
            )
        ]


__all__ = ["UnittestEditOperator"]
