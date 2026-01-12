"""
Agent Coder breeding strategy implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

from loguru import logger

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import (
    OPERATION_EDIT,
    CoevolutionContext,
    InitialInput,
    IProbabilityAssigner,
    OperatorRatesConfig,
    PopulationConfig,
    Problem,
)
from coevolution.core.interfaces.types import OPERATION_INITIAL

from ..operators.agent_coder_llm_operator import (
    AgentCoderEditInput,
    AgentCoderLLMOperator,
)
from .base_breeding import BaseBreedingStrategy

type TestPopulationType = str  # e.g., "unit", "differential", "public"


@dataclass
class FailingTest:
    test_id: str
    test_case: str
    error_trace: str


class AgentCoderBreedingStrategy(BaseBreedingStrategy[CodeIndividual]):
    """
    Concrete AgentCoderBreedingStrategy using a CodeLLMOperator.
    """

    def __init__(
        self,
        operator: AgentCoderLLMOperator,
        op_rates_config: OperatorRatesConfig,
        pop_config: PopulationConfig,
        probability_assigner: IProbabilityAssigner,
        init_pop_batch_size: int = 2,
        max_workers: int = 1,
    ) -> None:
        """
        Initialize the CodeBreedingStrategy.
        Args:
            operator: The CodeLLMOperator to use for breeding operations.
            op_rates_config: Configuration for operation rates.
            pop_config: Population configuration parameters.
            probability_assigner: Strategy for assigning probabilities to offspring.
            parent_selector: Strategy for selecting parent individuals.
            failing_test_selector: Strategy for selecting failing tests for edit operations.
            init_pop_batch_size: Number of individuals to generate per batch during initialization.
            max_workers: Maximum number of parallel workers for initialization.
        """
        self.operator = operator
        self.op_rates_config = op_rates_config
        self.pop_config = pop_config
        self.probability_assigner = probability_assigner
        self.max_workers = max_workers

        if pop_config.initial_population_size != 1:
            raise ValueError(
                f"AgentCoder breeding strategy only supports initial_population_size=1, but got {pop_config.initial_population_size}"
            )

        if pop_config.max_population_size != 1:
            raise ValueError(
                f"AgentCoder breeding strategy only supports max_population_size=1, but got {pop_config.max_population_size}"
            )

        for operation, rate in self.op_rates_config.operation_rates.items():
            if operation != "edit":
                raise ValueError(
                    f"AgentCoder breeding strategy only supports 'edit' operation, "
                    f"but got '{operation}'"
                )

            if rate < 1:
                raise ValueError(f"Operation rate for 'edit' must be 1, but got {rate}")

        # validate operations in rates config are supported by the operator
        for op in self.op_rates_config.operation_rates.keys():
            if op not in self.operator.supported_operations():
                raise ValueError(
                    f"Operation '{op}' in rates config is not supported by the operator."
                )

        logger.info("Initialized AgentCoderBreedingStrategy.")

    def initialize_individuals(
        self, problem: Problem
    ) -> tuple[list[CodeIndividual], str | None]:
        """Create initial individuals using parallel execution.

        Submits multiple batch generation tasks to a ThreadPoolExecutor.
        """
        individual: CodeIndividual
        input_dto = InitialInput(
            operation=OPERATION_INITIAL,
            question_content=problem.question_content,
            starter_code=problem.starter_code,
            population_size=1,  # AgentCoder only supports pop=1
        )

        output = self.operator.generate_initial_snippets(input_dto)
        individual = CodeIndividual(
            snippet=output[0].results[0].snippet,
            probability=self.pop_config.initial_prior,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={"code": [], "test": []},
        )

        logger.info("Initialized 1 individual using AgentCoderBreedingStrategy.")
        return [individual], None

    @override
    def breed(
        self, coevolution_context: CoevolutionContext, num_offsprings: int
    ) -> list[CodeIndividual]:
        """Breed new individuals using the AgentCoder operator."""
        if num_offsprings > 1:
            raise ValueError(
                "AgentCoder breeding strategy only supports breeding one offspring at a time."
            )

        code_pop = coevolution_context.code_population
        if len(code_pop) != 1:
            raise ValueError(
                "AgentCoder breeding strategy requires exactly one individual in the code population."
            )

        code_parent = code_pop[0]

        unittest_exec_results = coevolution_context.interactions[
            "unittest"
        ].execution_results
        if code_parent.id not in unittest_exec_results:
            raise ValueError(
                f"No execution results found for code individual ID {code_parent.id} in unittest interaction."
            )
        test_results = unittest_exec_results[code_parent.id].test_results

        unittest_pop = coevolution_context.test_populations["unittest"]

        # Extract failing tests
        failing_tests: list[FailingTest] = []
        for test_ind in unittest_pop:
            if test_results[test_ind.id].status == "failed":
                failing_tests.append(
                    FailingTest(
                        test_id=test_ind.id,
                        test_case=test_ind.snippet,
                        error_trace=test_results[test_ind.id].details or "",
                    )
                )

        if not failing_tests:
            logger.info(
                f"No failing tests found for code individual ID {code_parent.id}. Skipping edit."
            )
            return []

        # Prepare edit input
        edit_input = AgentCoderEditInput(
            operation=OPERATION_EDIT,
            question_content=coevolution_context.problem.question_content,
            current_snippet=code_parent.snippet,
            failing_tests_with_trace=[
                (ft.test_case, ft.error_trace) for ft in failing_tests
            ],
            starter_code=coevolution_context.problem.starter_code,
        )

        # Apply edit operation
        output = self.operator.apply(edit_input)
        edited_snippet = output.results[0].snippet
        new_individual = CodeIndividual(
            snippet=edited_snippet,
            probability=self.probability_assigner.assign_probability(
                parent_probs=[code_parent.probability],
                operation=OPERATION_EDIT,
                initial_prior=self.pop_config.initial_prior,
            ),
            creation_op=OPERATION_EDIT,
            generation_born=coevolution_context.code_population.generation + 1,
            parents={
                "code": [code_parent.id],
                "test": [failing_test.test_id for failing_test in failing_tests],
            },
        )

        logger.info(
            f"Bred new individual from parent ID {code_parent.id} using AgentCoder edit."
        )
        return [new_individual]


__all__ = ["AgentCoderBreedingStrategy"]
