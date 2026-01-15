"""
Concrete DifferentialBreedingStrategy.

Implements the breeding logic for differential testing.
Adheres to the BaseBreedingStrategy architecture.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol, cast, override

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


@dataclass(frozen=True)
class DifferentialResult:
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
        llm_workers: int = 1,
        divergence_limit: int = 5,
    ) -> None:
        super().__init__(op_rates_config, llm_workers)

        self.operator = operator
        self.differential_finder = differential_finder
        self.pop_config = pop_config
        self.probability_assigner = probability_assigner
        self.parent_selector = parent_selector
        self.func_eq_code_selector = functionally_equivalent_code_selector
        self.divergence_limit = divergence_limit  # Max divergences to find per pair

        # Validate operations
        for op in self.op_rates_config.operation_rates.keys():
            if (
                op not in self.operator.supported_operations()
                and op != OPERATION_DISCOVERY
            ):
                raise ValueError(
                    f"DifferentialBreedingStrategy: Operation '{op}' not supported by the operator."
                )

        # Handlers not needed as all handled in breed()
        # Only OPERATION_DISCOVERY is relevant here.

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

    @override
    def breed(
        self, coevolution_context: CoevolutionContext, num_offsprings: int
    ) -> list[TestIndividual]:
        """
        Deterministic breeding implementation with Round-Robin Group Scheduling.

        Improvements:
        - Prevents "Diversity Starvation" by interleaving pairs from different functional groups.
        - Prioritizes best pairs WITHIN each group, but ensures all groups get a turn.
        """
        offspring_list: list[TestIndividual] = []

        # --- Step 1: Gather Candidates By Group ---
        logger.debug("Scanning for functionally equivalent code pairs...")
        groups = self.func_eq_code_selector.select_functionally_equivalent_codes(
            coevolution_context
        )

        logger.info(
            f"Identified {len(groups)} functionally equivalent code groups for differential breeding."
        )

        # Store candidates as a list of lists: [ [GroupA_Pair1, GroupA_Pair2], [GroupB_Pair1] ]
        candidates_by_group: list[
            list[tuple[CodeIndividual, CodeIndividual, FunctionallyEquivGroup]]
        ] = []

        # Stats for logging
        theoretical_max_pairs = 0
        total_unexplored = 0

        for group in groups:
            n = len(group.code_individuals)
            if n < 2:
                continue

            # nCr Analysis
            nCr = (n * (n - 1)) // 2
            theoretical_max_pairs += nCr

            # Generate pairs for THIS group
            group_pairs = []
            sorted_inds = sorted(group.code_individuals, key=lambda x: x.id)

            for i in range(len(sorted_inds)):
                for j in range(i + 1, len(sorted_inds)):
                    code_a = sorted_inds[i]
                    code_b = sorted_inds[j]

                    if not self._is_pair_explored(code_a, code_b):
                        group_pairs.append((code_a, code_b, group))

            if group_pairs:
                # Local Elitism: Sort pairs within this group by quality
                group_pairs.sort(
                    key=lambda p: p[0].probability + p[1].probability, reverse=True
                )
                candidates_by_group.append(group_pairs)
                total_unexplored += len(group_pairs)

        # Log Capacity
        potential_offspring = total_unexplored * self.divergence_limit * 2
        logger.info(
            f"Differential Capacity Analysis: "
            f"Candidate Groups={len(candidates_by_group)}, "
            f"Total Pairs (nCr)={theoretical_max_pairs}, "
            f"Unexplored={total_unexplored}, "
            f"Max Potential Offspring={potential_offspring} "
            f"(Requested: {num_offsprings})"
        )

        if total_unexplored == 0:
            logger.warning(
                "No new unexplored pairs available for differential testing."
            )
            return []

        # --- Step 2: Interleave (Round-Robin Flattening) ---
        # We transform [ [A1, A2, A3], [B1, B2] ] -> [A1, B1, A2, B2, A3]
        final_candidate_queue = []

        # We loop until all groups are exhausted
        while candidates_by_group:
            # Iterate through a copy so we can remove empty groups safely
            for group_list in list(candidates_by_group):
                if group_list:
                    # Take the best remaining pair from this group
                    final_candidate_queue.append(group_list.pop(0))

                # If group is now empty, remove it from rotation
                if not group_list:
                    candidates_by_group.remove(group_list)

        logger.debug(
            f"Scheduled {len(final_candidate_queue)} pairs via Round-Robin selection."
        )

        # --- Step 3: Batch Execution (Standard) ---
        # Now we just consume the interleaved queue
        total_candidates = len(final_candidate_queue)
        candidate_idx = 0

        if self.llm_workers <= 1:
            while (
                len(offspring_list) < num_offsprings
                and candidate_idx < total_candidates
            ):
                code_a, code_b, group = final_candidate_queue[candidate_idx]
                candidate_idx += 1
                self._mark_pair_explored(code_a, code_b)
                new_inds = self._process_pair(
                    coevolution_context, code_a, code_b, group
                )
                offspring_list.extend(new_inds)
            return offspring_list[:num_offsprings]

        # Multi-threaded logic (same as before, just using final_candidate_queue)
        with ThreadPoolExecutor(max_workers=self.llm_workers) as executor:
            while (
                len(offspring_list) < num_offsprings
                and candidate_idx < total_candidates
            ):
                remaining_needed = num_offsprings - len(offspring_list)
                batch_size = min(remaining_needed + 2, self.llm_workers * 2)

                current_batch_futures = []
                for _ in range(batch_size):
                    if candidate_idx >= total_candidates:
                        break
                    cA, cB, grp = final_candidate_queue[candidate_idx]
                    candidate_idx += 1
                    self._mark_pair_explored(cA, cB)

                    current_batch_futures.append(
                        executor.submit(
                            self._process_pair, coevolution_context, cA, cB, grp
                        )
                    )

                if not current_batch_futures:
                    break

                for future in as_completed(current_batch_futures):
                    try:
                        res = future.result()
                        offspring_list.extend(res)
                        if len(offspring_list) >= num_offsprings:
                            for f in current_batch_futures:
                                f.cancel()
                            break
                    except Exception as e:
                        logger.error(f"Worker failed: {e}")

        return offspring_list[:num_offsprings]

    # --------------------------------------------------------------------------
    # WORKER LOGIC
    # --------------------------------------------------------------------------

    def _process_pair(
        self,
        context: CoevolutionContext,
        code_a: CodeIndividual,
        code_b: CodeIndividual,
        group: FunctionallyEquivGroup,
    ) -> list[TestIndividual]:
        """
        The actual work unit.
        """
        logger.debug(f"Processing code pair ({code_a.id}, {code_b.id}) for divergence.")
        try:
            # 1. Prepare Context (Same as before)
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
                num_inputs_to_generate=100,  # This is inputs generated by script
            )

            # 2. Run LLM Operator
            op_output = self.operator.apply(dto)
            script = op_output.results[0].snippet

            # 3. Run Differential Finder
            divergences = self.differential_finder.find_differential(
                code_a.snippet,
                code_b.snippet,
                script,
                limit=self.divergence_limit,
            )

            if not divergences:
                logger.debug(
                    f"No divergences found for code pair ({code_a.id}, {code_b.id})."
                )
                return []

            # 4. Assign Probability & Create Individuals (Same as before)
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

            return self._create_divergence_tests(
                context, code_a, code_b, divergences, prob
            )

        except Exception as e:
            logger.error(f"Error processing pair {code_a.id} vs {code_b.id}: {e}")
            return []

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
            {"inputdata": div.input_data, "output": div.output_a} for div in divergences
        ]

        # Scenario 2: Code B is the "Winner"
        io_pairs_b: list[DifferentialInputOutput] = [
            {"inputdata": div.input_data, "output": div.output_b} for div in divergences
        ]

        # Define Hypotheses: (Winner, Loser, IO_Pairs, Outputs_For_Metadata)
        scenarios = [
            (code_a, code_b, io_pairs_a, [d.output_a for d in divergences]),
            (code_b, code_a, io_pairs_b, [d.output_b for d in divergences]),
        ]

        for winner, loser, io_pairs, winner_outputs in scenarios:
            for i, io_pair in enumerate(io_pairs):
                # One test per divergence input
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
                                d.output_b if winner == code_a else d.output_a
                                for d in divergences
                            ],
                        },
                    },
                )
                results.append(ind)

        logger.debug(
            f"Created {len(results)} divergence tests for codes ({code_a.id}, {code_b.id})"
        )
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
