"""
Concrete DifferentialBreedingStrategy.

Implements the breeding logic for differential testing.
Adheres to the BaseBreedingStrategy architecture.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict, cast

from loguru import logger

from ..core.individual import CodeIndividual, TestIndividual
from ..core.interfaces import (
    OPERATION_CROSSOVER,
    OPERATION_INITIAL,
    CoevolutionContext,
    InitialInput,
    IParentSelectionStrategy,
    IProbabilityAssigner,
    OperatorRatesConfig,
    PopulationConfig,
    Problem,
)
from ..operators.differential_llm_operator import (
    OPERATION_DISCOVERY,
    DifferentialCrossoverInput,
    DifferentialGenScriptInput,
    DifferentialInputOutput,
    DifferentialLLMOperator,
)
from .base_breeding import BaseBreedingStrategy


@dataclass(frozen=True)
class FunctionallyEquivGroup:
    """Represents a cluster of code individuals that behave identically on current tests."""

    code_individuals: list[CodeIndividual]  # <--- CHANGED: Direct Objects
    passing_test_individuals: dict[str, list[TestIndividual]]


class IFunctionallyEquivalentCodeSelector(Protocol):
    """Protocol for finding groups of code that behave identically."""

    def select_functionally_equivalent_codes(
        self,
        coevolution_context: CoevolutionContext,
    ) -> list[FunctionallyEquivGroup]: ...


class DivergenceResult(TypedDict):
    """Represents a single input case where two code snippets behaved differently."""

    input_data: dict[str, Any]
    output_a: Any
    output_b: Any


class IDivergenceFinder(Protocol):
    """Protocol for the execution sandbox."""

    def find_divergence(
        self,
        code_a_snippet: str,
        code_b_snippet: str,
        input_generator_script: str,
        limit: int = 10,
    ) -> list[DivergenceResult]:
        """
        Executes the generator script and the two code snippets.
        Returns a list of all found divergences (input, output_a, output_b).
        Returns empty list if no divergence is found.
        """
        ...


class DifferentialBreedingStrategy(BaseBreedingStrategy[TestIndividual]):
    """
    Concrete DifferentialBreedingStrategy using DifferentialLLMOperator + DivergenceFinder.
    """

    def __init__(
        self,
        operator: DifferentialLLMOperator,
        divergence_finder: IDivergenceFinder,
        op_rates_config: OperatorRatesConfig,
        pop_config: PopulationConfig,
        probability_assigner: IProbabilityAssigner,
        parent_selector: IParentSelectionStrategy[TestIndividual],
        functionally_equivalent_code_selector: IFunctionallyEquivalentCodeSelector,
        max_workers: int = 1,
    ) -> None:
        super().__init__(op_rates_config, max_workers)

        self.operator = operator
        self.divergence_finder = divergence_finder
        self.pop_config = pop_config
        self.probability_assigner = probability_assigner
        self.parent_selector = parent_selector
        self.func_eq_code_selector = functionally_equivalent_code_selector

        # Validate operations
        for op in self.op_rates_config.operation_rates.keys():
            if (
                op not in self.operator.supported_operations()
                and op != OPERATION_DISCOVERY
            ):
                # OPERATION_DISCOVERY is handled via inputs mapping to supported operations
                pass

        # Register Handlers
        self._strategies = {
            OPERATION_DISCOVERY: self._breed_via_discovery,
            OPERATION_CROSSOVER: self._breed_via_crossover,
        }

    def initialize_individuals(
        self, problem: Problem
    ) -> tuple[list[TestIndividual], str | None]:
        """
        Sets up the test suite scaffold.
        """
        _, test_class_block = self.operator.generate_initial_snippets(
            InitialInput(
                operation=OPERATION_INITIAL,
                question_content=problem.question_content,
                starter_code=problem.starter_code,
                population_size=0,
            )
        )

        if test_class_block is None:
            logger.error("Failed to setup initial differential test class block.")
            return [], None

        return [], test_class_block

    # --- Handlers ---

    def _breed_via_discovery(self, context: CoevolutionContext) -> list[TestIndividual]:
        """
        Handler for OPERATION_DISCOVERY.
        """
        groups = self.func_eq_code_selector.select_functionally_equivalent_codes(
            context
        )
        valid_groups = [g for g in groups if len(g.code_individuals) >= 2]

        if not valid_groups:
            return []

        group = random.choice(valid_groups)
        code_a, code_b = random.sample(group.code_individuals, 2)

        # Context building
        diff_tests = group.passing_test_individuals.get("differential", [])
        passing_io_pairs: list[DifferentialInputOutput] = []
        for t in diff_tests:
            pairs = t.metadata.get("io_pairs", [])
            if pairs:
                passing_io_pairs.append(pairs[0])

        dto = DifferentialGenScriptInput(
            operation=OPERATION_DISCOVERY,
            question_content=context.problem.question_content,
            equivalent_code_snippet_1=code_a.snippet,
            equivalent_code_snippet_2=code_b.snippet,
            passing_differential_test_io_pairs=passing_io_pairs,
            num_inputs_to_generate=100,
        )

        try:
            op_output = self.operator.apply(dto)
            script = op_output.results[0].snippet
        except Exception as e:
            logger.warning(f"Differential discovery operator failed: {e}")
            return []

        # Run Divergence Finder (Returns a LIST now)
        divergences = self.divergence_finder.find_divergence(
            code_a.snippet, code_b.snippet, script
        )

        if not divergences:
            return []

        # Create Aggregated Individuals (All pairs in one test)
        return self._create_divergence_tests(context, code_a, code_b, divergences)

    def _breed_via_crossover(self, context: CoevolutionContext) -> list[TestIndividual]:
        # ... (Crossover logic remains identical, it handles lists of IO pairs naturally) ...
        parents = self.parent_selector.select_parents(
            context.test_populations["differential"],
            count=2,
            coevolution_context=context,
        )
        if len(parents) < 2:
            return []

        p1, p2 = parents[0], parents[1]
        p1_io = p1.metadata.get("io_pairs", [])
        p2_io = p2.metadata.get("io_pairs", [])

        if not p1_io or not p2_io:
            return []

        dto = DifferentialCrossoverInput(
            operation=OPERATION_CROSSOVER,
            question_content=context.problem.question_content,
            starter_code=context.problem.starter_code,
            differential_parent_1_io_pairs=cast(list[DifferentialInputOutput], p1_io),
            differential_parent_1_id=p1.id,
            differential_parent_2_io_pairs=cast(list[DifferentialInputOutput], p2_io),
            differential_parent_2_id=p2.id,
        )

        try:
            output = self.operator.apply(dto)
        except Exception:
            return []

        offspring = []
        for res in output.results:
            prob = self.probability_assigner.assign_probability(
                operation=OPERATION_CROSSOVER,
                parent_probs=[p1.probability, p2.probability],
                initial_prior=self.pop_config.initial_prior,
            )

            offspring.append(
                TestIndividual(
                    snippet=res.snippet,
                    probability=prob,
                    creation_op=OPERATION_CROSSOVER,
                    generation_born=context.test_populations["differential"].generation
                    + 1,
                    parents={"code": [], "test": [p1.id, p2.id]},
                    metadata={"io_pairs": res.metadata["io_pairs"]},
                )
            )

        return offspring

    def _create_divergence_tests(
        self,
        context: CoevolutionContext,
        code_a: CodeIndividual,
        code_b: CodeIndividual,
        divergences: list[DivergenceResult],
    ) -> list[TestIndividual]:
        """
        Creates two competing TestIndividuals.
        - Individual A assumes Code A's outputs are correct for ALL divergences.
        - Individual B assumes Code B's outputs are correct for ALL divergences.
        """
        results = []

        # Scenario 1: Code A is the "Winner" (Source of Truth)
        # We collect all inputs and map them to A's outputs
        io_pairs_a: list[DifferentialInputOutput] = [
            {"inputdata": div["input_data"], "output": div["output_a"]}
            for div in divergences
        ]

        # Scenario 2: Code B is the "Winner"
        io_pairs_b: list[DifferentialInputOutput] = [
            {"inputdata": div["input_data"], "output": div["output_b"]}
            for div in divergences
        ]

        # Define Hypotheses: (Winner, Loser, IO_Pairs, Outputs_For_Metadata)
        scenarios = [
            (code_a, code_b, io_pairs_a, [d["output_a"] for d in divergences]),
            (code_b, code_a, io_pairs_b, [d["output_b"] for d in divergences]),
        ]

        for winner, loser, io_pairs, winner_outputs in scenarios:
            # Generate the Python Test Method (Contains assertions for ALL pairs)
            snippet = self.operator.get_test_method_from_io(
                context.problem.starter_code, io_pairs, [winner.id, loser.id]
            )

            ind = TestIndividual(
                snippet=snippet,
                probability=self.pop_config.initial_prior,
                creation_op=OPERATION_DISCOVERY,
                generation_born=context.code_population.generation + 1,
                parents={"code": [winner.id, loser.id], "test": []},
                metadata={
                    "io_pairs": io_pairs,
                    # We store lists in metadata now to track the batch
                    "divergence_outputs": {
                        winner.id: winner_outputs,
                        # The loser's outputs are the complement set
                        loser.id: [
                            d["output_b"] if winner == code_a else d["output_a"]
                            for d in divergences
                        ],
                    },
                },
            )
            results.append(ind)

        return results


__all__ = ["DifferentialBreedingStrategy"]
