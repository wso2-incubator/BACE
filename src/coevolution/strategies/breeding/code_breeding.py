"""
This module defines the `CodeBreedingStrategy` class, a concrete implementation of a breeding strategy
for code individuals in a coevolutionary framework. It leverages a language model operator to perform
genetic operations such as mutation, crossover, and edit, facilitating the evolution of code solutions
in response to a given problem and associated test cases.

Classes and Protocols:
----------------------
- IFailingTestSelector (Protocol): Defines the interface for selecting failing tests for a given code individual.
- CodeBreedingStrategy: Implements the main breeding strategy for code individuals, supporting mutation,
    crossover, and edit operations using a provided operator and configuration.

Key Methods:
------------
- __init__: Initializes the strategy with the operator, configuration, probability assigner, parent selector,
    and failing test selector.
- initialize_individuals: Generates the initial population of code individuals using the operator.
- _breed_via_mutation: Produces offspring via mutation of a selected parent.
- _breed_via_crossover: Produces offspring via crossover between two selected parents.
- _breed_via_edit: Produces offspring by editing a parent individual in response to a failing test case.

Usage:
------
This strategy is intended to be used within a coevolutionary algorithm where code and test populations
are evolved together. The strategy relies on external components for parent selection, probability assignment,
and failing test selection, allowing for flexible integration with different evolutionary setups.

Raises:
-------
- ValueError: If the operator rates configuration includes operations not supported by the provided operator.
- RuntimeError: If the operator fails to generate initial code snippets.

Exports:
--------
- CodeBreedingStrategy
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol

from loguru import logger

from coevolution.core.individual import CodeIndividual, TestIndividual
from coevolution.core.interfaces import (
    OPERATION_CROSSOVER,
    OPERATION_EDIT,
    OPERATION_INITIAL,
    OPERATION_MUTATION,
    CoevolutionContext,
    InitialInput,
    IParentSelectionStrategy,
    IProbabilityAssigner,
    OperatorRatesConfig,
    PopulationConfig,
    Problem,
)

from ..operators.code_llm_operator import (
    CodeCrossoverInput,
    CodeEditInput,
    CodeLLMOperator,
    CodeMutationInput,
    PlanGenerationInput,
    PlanToCodeInput,
)
from .base_breeding import BaseBreedingStrategy

type TestPopulationType = str  # e.g., "unit", "differential", "public"


class IFailingTestSelector(Protocol):
    """Protocol for selecting failing tests for code individuals."""

    @staticmethod
    def select_k_failing_tests(
        coevolution_context: CoevolutionContext,
        code_individual: CodeIndividual,
        k: int = 10,
    ) -> list[tuple[TestIndividual, TestPopulationType]]:
        """Select up to k failing tests for the given code individual.

        Args:
            coevolution_context: Current coevolution context with populations and interactions.
            code_individual: The code individual for which to select failing tests.
            k: Maximum number of failing tests to select.

        Returns:
            A list of tuples (selected_test_individual, test_population_type).
            Empty list if no failing tests are found.
        """
        ...


class CodeBreedingStrategy(BaseBreedingStrategy[CodeIndividual]):
    """
    Concrete CodeBreedingStrategy using a CodeLLMOperator.
    """

    def __init__(
        self,
        operator: CodeLLMOperator,
        op_rates_config: OperatorRatesConfig,
        pop_config: PopulationConfig,
        probability_assigner: IProbabilityAssigner,
        parent_selector: IParentSelectionStrategy[CodeIndividual],
        failing_test_selector: IFailingTestSelector,
        init_pop_batch_size: int = 2,
        llm_workers: int = 1,
        k_failing_tests: int = 10,
        planning_enabled: bool = False,
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
            llm_workers: Maximum number of parallel workers for initialization.
            k_failing_tests: Maximum number of failing tests to select for edit operations (default: 10).
            planning_enabled: When True, initialization generates a plan for every individual
                before generating code from that plan (two-phase parallel init). Individuals
                will have their plan stored in metadata["plan"]. Default: False.
        """
        self.operator = operator
        self.op_rates_config = op_rates_config
        self.pop_config = pop_config
        self.probability_assigner = probability_assigner
        self.parent_selector = parent_selector
        self.failing_test_selector = failing_test_selector
        self.llm_workers = llm_workers
        self.init_pop_batch_size = init_pop_batch_size
        self.k_failing_tests = k_failing_tests
        self.planning_enabled = planning_enabled

        if pop_config.initial_population_size < init_pop_batch_size:
            self.init_pop_batch_size = pop_config.initial_population_size
            logger.info(
                f"Adjusted init_pop_batch_size to {self.init_pop_batch_size} "
                f"to not exceed initial_population_size."
            )

        # validate operations in rates config are supported by the operator
        for op in self.op_rates_config.operation_rates.keys():
            if op not in self.operator.supported_operations():
                raise ValueError(
                    f"Operation '{op}' in rates config is not supported by the operator."
                )

        self._strategies = {
            OPERATION_MUTATION: self._breed_via_mutation,
            OPERATION_CROSSOVER: self._breed_via_crossover,
            OPERATION_EDIT: self._breed_via_edit,
        }

    def initialize_individuals(self, problem: Problem) -> list[CodeIndividual]:
        """Create initial individuals using parallel execution.

        When ``planning_enabled`` is False (default), submits multiple batch
        generation tasks to a ThreadPoolExecutor via ``generate_initial_snippets``.

        When ``planning_enabled`` is True, runs two sequential parallel phases:
          Phase A — generate one plan per individual (all in parallel).
          Phase B — generate one code snippet per plan (all in parallel).
        Each resulting CodeIndividual carries ``metadata[\"plan\"]``.
        """
        initial_pop_size = self.pop_config.initial_population_size

        if self.planning_enabled:
            return self._initialize_individuals_with_planning(problem, initial_pop_size)
        else:
            return self._initialize_individuals_standard(problem, initial_pop_size)

    # ------------------------------------------------------------------
    # Private initialization helpers
    # ------------------------------------------------------------------

    def _initialize_individuals_standard(
        self, problem: Problem, initial_pop_size: int
    ) -> list[CodeIndividual]:
        """Original batched initialisation without planning."""
        individuals: list[CodeIndividual] = []

        # Calculate number of batches needed
        # equivalent to math.ceil(initial_pop_size / pop_batch_size)
        num_batches = (
            initial_pop_size + self.init_pop_batch_size - 1
        ) // self.init_pop_batch_size

        # Prepare per-batch DTOs so each batch can use a sampled rephrasing
        input_dtos = []
        for _ in range(num_batches):
            prompt_content = self.select_problem_rephrasing(problem)
            input_dtos.append(
                InitialInput(
                    operation=OPERATION_INITIAL,
                    question_content=prompt_content,
                    starter_code=problem.starter_code,
                    population_size=self.init_pop_batch_size,
                )
            )

        logger.info(
            f"Initializing population of size {initial_pop_size} in {num_batches} parallel batches."
        )

        with ThreadPoolExecutor(max_workers=self.llm_workers) as executor:
            # Submit all batch tasks at once
            future_to_batch = {
                executor.submit(self.operator.generate_initial_snippets, dto): i
                for i, dto in enumerate(input_dtos)
            }

            for future in as_completed(future_to_batch):
                try:
                    initial_outputs = future.result()

                    if not initial_outputs or not initial_outputs.results:
                        logger.warning("A batch task returned no results.")
                        continue

                    for operator_result in initial_outputs.results:
                        individual = CodeIndividual(
                            snippet=operator_result.snippet,
                            probability=self.pop_config.initial_prior,
                            creation_op=OPERATION_INITIAL,
                            generation_born=0,
                        )
                        individuals.append(individual)

                    logger.debug(
                        f"Population progress: {len(individuals)}/{initial_pop_size}"
                    )

                except Exception as e:
                    logger.error(f"Error in initialization batch: {e}")

        # Validation: Ensure we generated at least some individuals
        if not individuals:
            logger.error(
                "CodeBreedingStrategy.initialize_individuals: No initial snippets generated"
            )
            raise RuntimeError("Failed to generate initial code snippets")

        # Trim excess individuals if any
        if len(individuals) > initial_pop_size:
            individuals = individuals[:initial_pop_size]

        logger.debug(f"Created {len(individuals)} code individuals (Gen 0)")
        return individuals

    def _initialize_individuals_with_planning(
        self, problem: Problem, initial_pop_size: int
    ) -> list[CodeIndividual]:
        """Two-phase parallel initialisation with planning.

        Phase A: Generate ``initial_pop_size`` plans in parallel.
        Phase B: Generate one code snippet per plan in parallel.
        Each CodeIndividual carries ``metadata[\"plan\"]``.
        """
        logger.info(
            f"[Planning] Initializing population of size {initial_pop_size}: "
            "Phase A (plans) then Phase B (code from plans)."
        )

        # ---- Phase A: generate plans -----------------------------------------
        plan_dtos = [
            PlanGenerationInput(
                operation=OPERATION_INITIAL,
                question_content=self.select_problem_rephrasing(problem),
                starter_code=problem.starter_code,
            )
            for _ in range(initial_pop_size)
        ]

        plans: list[str] = []
        with ThreadPoolExecutor(max_workers=self.llm_workers) as executor:
            futures = [
                executor.submit(self.operator.generate_plan, dto) for dto in plan_dtos
            ]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    plans.append(result.snippet)
                except Exception as e:
                    logger.error(f"[Planning] Plan generation failed: {e}")

        if not plans:
            logger.error(
                "[Planning] CodeBreedingStrategy: No plans generated during Phase A."
            )
            raise RuntimeError("Failed to generate plans for initial population.")

        logger.info(f"[Planning] Phase A complete: {len(plans)} plans generated.")

        # ---- Phase B: generate code from plans --------------------------------
        code_dtos = [
            PlanToCodeInput(
                operation=OPERATION_INITIAL,
                question_content=self.select_problem_rephrasing(problem),
                plan=plan,
                starter_code=problem.starter_code,
            )
            for plan in plans
        ]

        individuals: list[CodeIndividual] = []
        with ThreadPoolExecutor(max_workers=self.llm_workers) as executor:
            futures = [
                executor.submit(self.operator.generate_code_from_plan, dto)
                for dto in code_dtos
            ]
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    individual = CodeIndividual(
                        snippet=result.snippet,
                        probability=self.pop_config.initial_prior,
                        creation_op=OPERATION_INITIAL,
                        generation_born=0,
                        metadata=result.metadata,  # contains {"plan": <plan text>}
                    )
                    individuals.append(individual)
                    logger.debug(
                        f"[Planning] Code progress: {len(individuals)}/{len(plans)}"
                    )
                except Exception as e:
                    logger.error(f"[Planning] Code generation from plan failed: {e}")

        if not individuals:
            logger.error(
                "[Planning] CodeBreedingStrategy: No individuals created during Phase B."
            )
            raise RuntimeError("Failed to generate code individuals from plans.")

        # Trim to requested population size
        if len(individuals) > initial_pop_size:
            individuals = individuals[:initial_pop_size]

        logger.info(
            f"[Planning] Phase B complete: {len(individuals)} individuals created (Gen 0)."
        )
        return individuals

    # ------------------------------------------------------------------
    # Planning helper for use by plan-level operator breed methods
    # ------------------------------------------------------------------

    def _get_or_generate_plan(
        self, context: CoevolutionContext, individual: CodeIndividual
    ) -> str:
        """Return the plan associated with *individual*, generating one if absent.

        Checks ``individual.metadata.get("plan")`` first.  If the individual has
        no plan (e.g. it was created without planning enabled, or was born via a
        standard operator), a fresh plan is generated via the operator using the
        current problem context.

        The plan is returned as a plain string and is **not** written back to the
        individual (individuals are immutable post-construction).  Callers should
        store the returned plan in the offspring's metadata.

        Args:
            context: Current coevolution context (supplies problem + rephrasing).
            individual: The code individual whose plan is needed.

        Returns:
            Plan text string.
        """
        existing_plan: str | None = individual.metadata.get("plan")
        if existing_plan:
            return existing_plan

        logger.debug(
            f"No plan found on individual '{individual.id}'; generating one on demand."
        )
        plan_dto = PlanGenerationInput(
            operation=OPERATION_INITIAL,
            question_content=self.select_problem_rephrasing(context.problem),
            starter_code=context.problem.starter_code,
        )
        result = self.operator.generate_plan(plan_dto)
        return result.snippet

    def _breed_via_mutation(self, context: CoevolutionContext) -> list[CodeIndividual]:
        """Breed new individuals via mutation."""
        # 1. Selection (Context is local!)
        parents: list[CodeIndividual] = self.parent_selector.select_parents(
            context.code_population, count=1, coevolution_context=context
        )
        if not parents:
            return []
        parent: CodeIndividual = parents[0]

        # 2. Execution
        dto = CodeMutationInput(
            operation=OPERATION_MUTATION,
            question_content=self.select_problem_rephrasing(context.problem),
            parent_snippet=parent.snippet,
            starter_code=context.problem.starter_code,
        )

        try:
            output = self.operator.apply(dto)
        except Exception as e:
            logger.warning(f"Mutation operator failed: {e}")
            return []

        # 3. Construction
        offspring = []
        for res in output.results:
            prob = self.probability_assigner.assign_probability(
                operation=OPERATION_MUTATION,
                parent_probs=[parent.probability],
                initial_prior=self.pop_config.initial_prior,
            )

            offspring.append(
                CodeIndividual(
                    snippet=res.snippet,
                    probability=prob,
                    creation_op=OPERATION_MUTATION,
                    generation_born=context.code_population.generation + 1,
                    parents={"code": [parent.id], "test": []},
                    metadata=res.metadata,
                )
            )

        return offspring

    def _breed_via_crossover(self, context: CoevolutionContext) -> list[CodeIndividual]:
        # Select two parents
        parents: list[CodeIndividual] = self.parent_selector.select_parents(
            context.code_population, count=2, coevolution_context=context
        )
        if len(parents) < 2:
            logger.warning("Crossover requires two parents, but less were selected.")
            return []

        parent1, parent2 = parents[0], parents[1]

        # Prepare DTO for crossover
        dto = CodeCrossoverInput(
            operation=OPERATION_CROSSOVER,
            question_content=self.select_problem_rephrasing(context.problem),
            parent1_snippet=parent1.snippet,
            parent2_snippet=parent2.snippet,
            starter_code=context.problem.starter_code,
        )

        try:
            output = self.operator.apply(dto)
        except Exception as e:
            logger.warning(f"Crossover operator failed: {e}")
            return []

        # Construct offspring individuals
        offspring = []
        for res in output.results:
            prob = self.probability_assigner.assign_probability(
                operation=OPERATION_CROSSOVER,
                parent_probs=[parent1.probability, parent2.probability],
                initial_prior=self.pop_config.initial_prior,
            )

            offspring.append(
                CodeIndividual(
                    snippet=res.snippet,
                    probability=prob,
                    creation_op=OPERATION_CROSSOVER,
                    generation_born=context.code_population.generation + 1,
                    parents={"code": [parent1.id, parent2.id], "test": []},
                    metadata=res.metadata,
                )
            )

        return offspring

    def _breed_via_edit(
        self, coevolution_context: CoevolutionContext
    ) -> list[CodeIndividual]:
        """Breed new individuals via edit using failing tests."""
        # 1. Select Parent
        parents: list[CodeIndividual] = self.parent_selector.select_parents(
            coevolution_context.code_population,
            count=1,
            coevolution_context=coevolution_context,
        )
        if not parents:
            return []
        parent: CodeIndividual = parents[0]

        # 2. Select K Failing Tests
        failing_test_selections = self.failing_test_selector.select_k_failing_tests(
            coevolution_context, parent, k=self.k_failing_tests
        )
        if not failing_test_selections:
            logger.warning(
                f"No failing tests found for code individual '{parent.id}' during edit."
            )
            return []

        # 3. Collect test cases with error traces
        failing_tests_with_trace: list[tuple[str, str]] = []
        test_ids: list[str] = []

        for failing_test_ind, test_population_type in failing_test_selections:
            error_trace = (
                coevolution_context.interactions[test_population_type]
                .execution_results[parent.id][failing_test_ind.id]
                .error_log
            )

            if not error_trace:
                logger.warning(
                    f"No error trace found for failing test '{failing_test_ind.id}' "
                    f"and code individual '{parent.id}'."
                )
                error_trace = "No error trace available."

            failing_tests_with_trace.append((failing_test_ind.snippet, error_trace))
            test_ids.append(failing_test_ind.id)

        # 4. Execution
        dto = CodeEditInput(
            operation=OPERATION_EDIT,
            question_content=self.select_problem_rephrasing(
                coevolution_context.problem
            ),
            parent_snippet=parent.snippet,
            failing_tests_with_trace=failing_tests_with_trace,
            starter_code=coevolution_context.problem.starter_code,
        )

        try:
            output = self.operator.apply(dto)
        except Exception as e:
            logger.warning(f"Edit operator failed: {e}")
            return []

        # 5. Construction
        offspring = []
        for res in output.results:
            prob = self.probability_assigner.assign_probability(
                operation=OPERATION_EDIT,
                parent_probs=[parent.probability],
                initial_prior=self.pop_config.initial_prior,
            )

            offspring.append(
                CodeIndividual(
                    snippet=res.snippet,
                    probability=prob,
                    creation_op=OPERATION_EDIT,
                    generation_born=coevolution_context.code_population.generation + 1,
                    parents={"code": [parent.id], "test": test_ids},
                    metadata={
                        **res.metadata,
                        "num_failing_tests_used": len(failing_tests_with_trace),
                    },
                )
            )

        return offspring

    # `select_problem_rephrasing` is provided by BaseBreedingStrategy


__all__ = ["CodeBreedingStrategy"]
