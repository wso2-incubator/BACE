"""
Concrete DifferentialBreedingStrategy.

Implements the breeding logic for differential testing.
Adheres to the BaseBreedingStrategy architecture.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from random import sample
from typing import Any, Optional, Protocol, cast, override

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


# --- Pipeline DTOs ---


@dataclass
class DiscoveryTask:
    """Represents a pair identified for potential divergence."""

    code_a: CodeIndividual
    code_b: CodeIndividual
    group: FunctionallyEquivGroup


@dataclass
class DiscoveryContext:
    """Represents a task with its generated script, ready for execution."""

    task: DiscoveryTask
    generator_script: str


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
        max_pairs_per_group: int = 5,
        num_passing_tests_to_sample: int = 5,
    ) -> None:
        super().__init__(op_rates_config, llm_workers)

        self.operator = operator
        self.differential_finder = differential_finder
        self.pop_config = pop_config
        self.probability_assigner = probability_assigner
        self.parent_selector = parent_selector
        self.func_eq_code_selector = functionally_equivalent_code_selector
        self.divergence_limit = divergence_limit  # Max divergences to find per pair
        self.max_pairs_per_group = (
            max_pairs_per_group  # Max pairs to try per functional group
        )
        self.num_passing_tests_to_sample = num_passing_tests_to_sample

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

    def initialize_individuals(self, problem: Problem) -> list[TestIndividual]:
        """
        Initialize differential testing with empty population.

        Differential tests start empty and are generated dynamically during evolution.
        """
        # Generate empty initial output to ensure operator is ready
        self.operator.generate_initial_snippets(
            InitialInput(
                operation=OPERATION_INITIAL,
                question_content=problem.question_content,
                starter_code=problem.starter_code,
                population_size=0,
            )
        )

        logger.debug("Differential test population starts empty (Gen 0)")
        return []

    @override
    def breed(
        self, coevolution_context: CoevolutionContext, num_offsprings: int
    ) -> list[TestIndividual]:
        """
        Staged Pipeline Implementation:

        Phase 1: Select candidate pairs (CPU - cheap)
        Phase 2: Generate input scripts via LLM (I/O - parallel ThreadPoolExecutor)
        Phase 3: Execute scripts and find divergences (CPU - DifferentialFinder handles parallelism)

        This architecture decouples LLM workers from CPU workers, eliminating the
        complex budget calculations and enabling independent scaling.
        """
        offspring_list: list[TestIndividual] = []

        # --- PHASE 1: SELECT CANDIDATES ---
        candidates = self._select_candidates(coevolution_context)

        if not candidates:
            logger.warning(
                "No new unexplored pairs available for differential testing."
            )
            return []

        # Calculate how many scripts to generate
        # Generate more than needed to account for pairs that may not find divergences
        target_script_count = min(len(candidates), num_offsprings * 2)
        selected_candidates = candidates[:target_script_count]

        logger.info(
            f"Phase 1 Complete: Selected {len(selected_candidates)} candidate pairs "
            f"from {len(candidates)} available (targeting {num_offsprings} offspring)"
        )

        # --- PHASE 2: BATCH GENERATE SCRIPTS (I/O - Threaded) ---
        logger.info(
            f"Phase 2: Generating input scripts for {len(selected_candidates)} pairs "
            f"using {self.llm_workers} LLM workers..."
        )
        discovery_contexts = self._batch_generate_scripts(
            coevolution_context, selected_candidates
        )

        logger.info(
            f"Phase 2 Complete: Successfully generated {len(discovery_contexts)} scripts"
        )

        if not discovery_contexts:
            logger.warning(
                "No scripts generated; cannot proceed with divergence finding."
            )
            return []

        # --- PHASE 3: BATCH FIND DIVERGENCES (CPU - Sequential with internal parallelism) ---
        logger.info(
            f"Phase 3: Executing {len(discovery_contexts)} scripts to find divergences..."
        )
        offspring_list = self._batch_find_divergences(
            coevolution_context, discovery_contexts, num_offsprings
        )

        logger.info(
            f"Phase 3 Complete: Generated {len(offspring_list)} test offspring "
            f"from {len(discovery_contexts)} executed scripts"
        )

        return offspring_list[:num_offsprings]

    # --------------------------------------------------------------------------
    # PHASE 1: CANDIDATE SELECTION
    # --------------------------------------------------------------------------

    def _select_candidates(self, context: CoevolutionContext) -> list[DiscoveryTask]:
        """
        Select candidate pairs using Round-Robin Group Scheduling.

        Returns a flat, interleaved list of DiscoveryTask objects ordered by:
        - Round-robin across groups (prevents diversity starvation)
        - Local elitism within each group (best pairs first)
        """
        logger.debug("Scanning for functionally equivalent code pairs...")
        groups = self.func_eq_code_selector.select_functionally_equivalent_codes(
            context
        )

        logger.info(
            f"Identified {len(groups)} functionally equivalent code groups for differential breeding."
        )

        # Store candidates grouped by their source group
        candidates_by_group: list[list[DiscoveryTask]] = []

        # Stats for logging
        theoretical_max_pairs = 0
        total_unexplored = 0
        total_sampled = 0

        for group_idx, group in enumerate(groups):
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
                        group_pairs.append(
                            DiscoveryTask(code_a=code_a, code_b=code_b, group=group)
                        )

            if group_pairs:
                # Local Elitism: Sort pairs within this group by quality
                group_pairs.sort(
                    key=lambda t: t.code_a.probability + t.code_b.probability,
                    reverse=True,
                )

                # Limit pairs per group to avoid excessive attempts on equivalent code
                original_count = len(group_pairs)
                group_pairs = group_pairs[: self.max_pairs_per_group]
                sampled_count = len(group_pairs)

                if original_count > sampled_count:
                    logger.debug(
                        f"Group {group_idx}: Limited pairs from {original_count} → {sampled_count} "
                        f"(max_pairs_per_group={self.max_pairs_per_group})"
                    )

                candidates_by_group.append(group_pairs)
                total_unexplored += original_count
                total_sampled += sampled_count

        # Log capacity analysis
        potential_offspring = total_sampled * self.divergence_limit * 2
        logger.info(
            f"Candidate Analysis: Groups={len(candidates_by_group)}, "
            f"Total Pairs (nCr)={theoretical_max_pairs}, "
            f"Unexplored={total_unexplored}, "
            f"Sampled={total_sampled}, "
            f"Max Potential Offspring={potential_offspring}"
        )

        # Round-Robin Interleaving
        # Transform [ [A1, A2, A3], [B1, B2] ] -> [A1, B1, A2, B2, A3]
        interleaved_candidates = []

        while candidates_by_group:
            for group_list in list(candidates_by_group):
                if group_list:
                    interleaved_candidates.append(group_list.pop(0))

                # Remove empty groups
                if not group_list:
                    candidates_by_group.remove(group_list)

        logger.debug(
            f"Scheduled {len(interleaved_candidates)} pairs via Round-Robin selection."
        )

        return interleaved_candidates

    # --------------------------------------------------------------------------
    # PHASE 2: SCRIPT GENERATION (I/O BOUND - PARALLEL)
    # --------------------------------------------------------------------------

    def _generate_single_script(
        self, context: CoevolutionContext, task: DiscoveryTask
    ) -> Optional[DiscoveryContext]:
        """
        Generate an input generator script for a single code pair using LLM.

        This is the unit of work for Phase 2 parallelism (I/O-bound LLM calls).
        """
        try:
            # Prepare passing test context
            all_test_snippets = [
                t.snippet
                for tests in task.group.passing_test_individuals.values()
                for t in tests
            ]
            passing_test_cases = sample(
                all_test_snippets,
                min(len(all_test_snippets), self.num_passing_tests_to_sample),
            )

            # Create LLM input
            dto = DifferentialGenScriptInput(
                operation=OPERATION_DISCOVERY,
                question_content=context.problem.question_content,
                equivalent_code_snippet_1=task.code_a.snippet,
                equivalent_code_snippet_2=task.code_b.snippet,
                passing_test_cases=passing_test_cases,
                num_inputs_to_generate=100,
            )

            # Call LLM operator
            op_output = self.operator.apply(dto)
            script = op_output.results[0].snippet

            logger.debug(
                f"Generated script for pair ({task.code_a.id}, {task.code_b.id})"
            )

            return DiscoveryContext(task=task, generator_script=script)

        except Exception as e:
            logger.error(
                f"Script generation failed for {task.code_a.id} vs {task.code_b.id}: {e}"
            )
            return None

    def _batch_generate_scripts(
        self, context: CoevolutionContext, tasks: list[DiscoveryTask]
    ) -> list[DiscoveryContext]:
        """
        Generate input scripts for multiple pairs in parallel using ThreadPoolExecutor.

        Uses self.llm_workers threads to maximize LLM API throughput.
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.llm_workers) as executor:
            future_to_task = {
                executor.submit(self._generate_single_script, context, task): task
                for task in tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    discovery_ctx = future.result()
                    if discovery_ctx:
                        # Mark pair as explored (script successfully generated)
                        self._mark_pair_explored(task.code_a, task.code_b)
                        results.append(discovery_ctx)
                except Exception as e:
                    logger.error(f"Script generation task failed: {e}")

        return results

    # --------------------------------------------------------------------------
    # PHASE 3: DIVERGENCE FINDING (CPU BOUND - SEQUENTIAL WITH INTERNAL PARALLELISM)
    # --------------------------------------------------------------------------

    def _batch_find_divergences(
        self,
        context: CoevolutionContext,
        discovery_contexts: list[DiscoveryContext],
        limit: int,
    ) -> list[TestIndividual]:
        """
        Execute generator scripts and find divergences sequentially.

        The DifferentialFinder internally uses multiprocessing for CPU-bound work,
        so we iterate sequentially here to avoid nested parallelism.

        Stops early when offspring limit is reached.
        """
        offspring: list[TestIndividual] = []

        for ctx in discovery_contexts:
            if len(offspring) >= limit:
                logger.debug(f"Offspring limit {limit} reached; stopping early.")
                break

            task = ctx.task

            try:
                # DifferentialFinder handles CPU parallelism internally
                divergences = self.differential_finder.find_differential(
                    task.code_a.snippet,
                    task.code_b.snippet,
                    ctx.generator_script,
                    limit=self.divergence_limit,
                )

                if not divergences:
                    logger.debug(
                        f"No divergences found for pair ({task.code_a.id}, {task.code_b.id})"
                    )
                    continue

                # Calculate probability for new tests
                parent_probs = []
                for test_type, test_inds in task.group.passing_test_individuals.items():
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

                # Create test individuals
                new_tests = self._create_divergence_tests(
                    context, task.code_a, task.code_b, divergences, prob
                )
                offspring.extend(new_tests)

                logger.debug(
                    f"Found {len(divergences)} divergences → {len(new_tests)} tests "
                    f"for pair ({task.code_a.id}, {task.code_b.id})"
                )

            except Exception as e:
                logger.error(
                    f"Divergence finding failed for {task.code_a.id} vs {task.code_b.id}: {e}"
                )

        return offspring[:limit]

    # --------------------------------------------------------------------------
    # HELPER METHODS
    # --------------------------------------------------------------------------

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
