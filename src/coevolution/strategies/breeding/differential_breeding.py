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

from coevolution.core.individual import CodeIndividual, TestIndividual
from coevolution.core.interfaces import (
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


class DifferentialResult(TypedDict):
    """Represents a single input case where two code snippets behaved differently."""

    input_data: dict[str, Any]
    output_a: Any
    output_b: Any


class IDifferentialFinder(Protocol):
    """Protocol for the execution sandbox."""

    def find_differential(
        self,
        code_a_snippet: str,
        code_b_snippet: str,
        input_generator_script: str,
        limit: int = 10,
    ) -> list[DifferentialResult]:
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
        differential_finder: IDifferentialFinder,
        op_rates_config: OperatorRatesConfig,
        pop_config: PopulationConfig,
        probability_assigner: IProbabilityAssigner,
        parent_selector: IParentSelectionStrategy[TestIndividual],
        functionally_equivalent_code_selector: IFunctionallyEquivalentCodeSelector,
        max_workers: int = 1,
    ) -> None:
        super().__init__(op_rates_config, max_workers)

        self.operator = operator
        self.differential_finder = differential_finder
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
                raise ValueError(
                    f"DifferentialBreedingStrategy: Operation '{op}' not supported by the operator."
                )

        # Register Handlers
        self._strategies = {
            OPERATION_DISCOVERY: self._breed_via_discovery,
        }

        # Cache of pairs that have already been explored for divergence (whether found or not).
        # Stores tuples of (min_id, max_id) to ensure order independence.
        self._explored_pairs_cache: set[tuple[str, str]] = set()
        logger.info("DifferentialBreedingStrategy initialized.")

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
            logger.debug(
                "No valid functionally equivalent code groups found for differential discovery."
            )
            return []

        group = random.choice(valid_groups)
        code_a, code_b = random.sample(group.code_individuals, 2)

        # Optimization: Skip if we've already checked this pair (successfully or not)
        if self._is_pair_explored(code_a, code_b):
            logger.debug(f"Pair ({code_a.id}, {code_b.id}) already explored. Skipping.")
            return []

        logger.debug(
            f"Selected Code Individuals {code_a.id} and {code_b.id} for differential discovery."
        )

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
            num_inputs_to_generate=100,  # TODO: make configurable
        )

        try:
            op_output = self.operator.apply(dto)
            script = op_output.results[0].snippet
        except Exception as e:
            logger.warning(f"Differential discovery operator failed: {e}")
            return []

        # Run Differential Finder (Returns a LIST now)
        divergences = self.differential_finder.find_differential(
            code_a.snippet, code_b.snippet, script
        )

        # Mark as explored regardless of outcome to prevent re-running
        self._mark_pair_explored(code_a, code_b)

        # No Divergences Found
        if not divergences:
            logger.debug(f"No divergences found between {code_a.id} and {code_b.id}.")
            return []

        # TODO: Probability assignment should be more sophisticated
        parent_probs = []
        for test_type, test_inds in group.passing_test_individuals.items():
            if test_type == "differential":
                for ind in test_inds:
                    parent_probs.append(ind.probability)

        if parent_probs:
            prob = self.probability_assigner.assign_probability(
                operation=OPERATION_DISCOVERY,
                parent_probs=parent_probs,
                initial_prior=self.pop_config.initial_prior,
            )
        else:
            prob = self.pop_config.initial_prior

        # Create TestIndividual per Input Divergence for BOTH scenarios
        return self._create_divergence_tests(context, code_a, code_b, divergences, prob)

    def _create_divergence_tests(
        self,
        context: CoevolutionContext,
        code_a: CodeIndividual,
        code_b: CodeIndividual,
        divergences: list[DifferentialResult],
        probability: float,
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
            for i, io_pair in enumerate(io_pairs):
                snippet = self.operator.get_test_method_from_io(
                    context.problem.starter_code, [io_pair], [winner.id, loser.id], i
                )
                ind = TestIndividual(
                    snippet=snippet,
                    probability=probability,
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

    def _mark_pair_explored(
        self, code_a: CodeIndividual, code_b: CodeIndividual
    ) -> None:
        """
        Cache the fact that this pair has been processed to prevent redundant checks.
        Uses a sorted tuple key to handle symmetry (A,B) == (B,A).
        """
        # Create a consistent key regardless of order
        pair_key = tuple(sorted((code_a.id, code_b.id)))
        self._explored_pairs_cache.add(cast(tuple[str, str], pair_key))

        logger.debug(f"Marked pair ({code_a.id}, {code_b.id}) as explored.")

    def _is_pair_explored(self, code_a: CodeIndividual, code_b: CodeIndividual) -> bool:
        """
        Returns True if we have already run differential discovery on this pair
        (regardless of whether we found divergences or not).
        """
        pair_key = tuple(sorted((code_a.id, code_b.id)))
        return pair_key in self._explored_pairs_cache


__all__ = ["DifferentialBreedingStrategy"]
