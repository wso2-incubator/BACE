"""UnittestInitializer — creates Gen-0 test individuals via LLM."""

from __future__ import annotations

from loguru import logger

from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import (
    OPERATION_INITIAL,
    PopulationConfig,
    Problem,
)
from coevolution.core.interfaces.language import (
    ICodeParser,
    LanguageParsingError,
    LanguageTransformationError,
)

from coevolution.strategies.llm_base import (
    BaseLLMInitializer,
    ILanguageModel,
    LLMGenerationError,
    LLMSyntaxError,
    llm_retry,
)
from ._helpers import _TestLLMHelpers


class UnittestInitializer(_TestLLMHelpers, BaseLLMInitializer[TestIndividual]):
    """Creates Gen-0 test individuals via LLM.

    Asks for `population_size` tests in one shot and recovers
    gracefully if the count doesn't match.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        pop_config: PopulationConfig,
        llm_workers: int = 1,
    ) -> None:
        super().__init__(llm, parser, language_name, pop_config)
        self.llm_workers = llm_workers

    def initialize(self, problem: Problem) -> list[TestIndividual]:
        target = self.pop_config.initial_population_size
        test_functions = self._generate_test_functions(problem, target)

        individuals: list[TestIndividual] = []
        for fn in test_functions:
            individuals.append(
                TestIndividual(
                    snippet=fn,
                    probability=self.pop_config.initial_prior,
                    creation_op=OPERATION_INITIAL,
                    generation_born=0,
                )
            )
        logger.debug(f"UnittestInitializer: created {len(individuals)} individuals")
        return individuals

    @llm_retry(
        (
            ValueError,
            LanguageParsingError,
            LanguageTransformationError,
            LLMGenerationError,
            LLMSyntaxError,
        )
    )
    def _generate_test_functions(self, problem: Problem, target: int) -> list[str]:
        prompt = self.prompt_manager.render_prompt(
            "operators/unittest/initial.j2",
            population_size=target,
            question_content=problem.question_content,
            starter_code=problem.starter_code,
        )
        response = self._generate(prompt)

        code_blocks = self.parser.extract_code_blocks(response)
        test_functions: list[str] = []
        for block in code_blocks:
            clean_block = self.parser.remove_main_block(block)
            test_functions.extend(self._extract_test_functions(clean_block))

        # Trim over-generation
        if len(test_functions) > target:
            test_functions = test_functions[:target]

        # Top-up under-generation with an additional LLM call
        if len(test_functions) < target:
            additional = target - len(test_functions)
            logger.info(f"UnittestInitializer: topping up {additional} more tests")
            extra_prompt = self.prompt_manager.render_prompt(
                "operators/unittest/initial.j2",
                population_size=additional,
                question_content=problem.question_content,
                starter_code=problem.starter_code,
            )
            extra_response = self._generate(extra_prompt)
            for block in self.parser.extract_code_blocks(extra_response):
                clean_block = self.parser.remove_main_block(block)
                test_functions.extend(self._extract_test_functions(clean_block))
                if len(test_functions) >= target:
                    break
            test_functions = test_functions[:target]

        return test_functions


__all__ = ["UnittestInitializer"]
