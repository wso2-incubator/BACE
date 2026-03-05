"""Concrete Differential Test Operator.

DifferentialDiscoveryOperator is a self-sufficient IOperator that runs
the full 3-phase discovery pipeline in a single execute() call:

  Phase 1 (CPU)   — select unexplored functionally-equivalent code pairs
  Phase 2 (I/O)   — generate an input-generator script per pair via LLM
  Phase 3 (CPU)   — run scripts through the sandbox, collect divergences,
                    and convert them to TestIndividuals

One call to execute() may return 0..N TestIndividuals depending on how many
divergences are found.  The Breeder accumulates across calls until it reaches
num_offsprings; because execute() typically returns several tests at once the
Breeder will often satisfy its quota in fewer calls than for code/unittest.

The explored-pairs cache lives on the operator instance between generations.
The Breeder/Orchestrator should call reset_explored_pairs() between problems.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from random import sample
from typing import Any, Optional, Protocol, cast

from loguru import logger

from coevolution.core.individual import CodeIndividual, TestIndividual
from coevolution.core.interfaces import (
    CoevolutionContext,
    Problem,
)
from coevolution.core.interfaces.language import ILanguage
from coevolution.core.interfaces.probability import IProbabilityAssigner
from coevolution.core.interfaces.selection import IParentSelectionStrategy

from .base_llm_service import (
    BaseEvolutionaryOperator,
    BaseLLMInitializer,
    ILanguageModel,
)
from .differential_llm_operator import (
    DifferentialGenScriptInput,
    DifferentialInputOutput,
    DifferentialLLMOperator,
)

OPERATION_DISCOVERY: str = "discovery"


# ---------------------------------------------------------------------------
# Shared data types (moved here from differential_breeding.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FunctionallyEquivGroup:
    """A cluster of code individuals that behave identically on current tests."""
    code_individuals: list[CodeIndividual]
    passing_test_individuals: dict[str, list[TestIndividual]]


class IFunctionallyEquivalentCodeSelector(Protocol):
    """Protocol for finding groups of code that behave identically."""
    def select_functionally_equivalent_codes(
        self,
        coevolution_context: CoevolutionContext,
    ) -> list[FunctionallyEquivGroup]: ...


@dataclass(frozen=True)
class DifferentialResult:
    """A single input where two code snippets produced different output."""
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
    ) -> list[DifferentialResult]: ...


@dataclass
class _DiscoveryTask:
    code_a: CodeIndividual
    code_b: CodeIndividual
    group: FunctionallyEquivGroup


@dataclass
class _DiscoveryContext:
    task: _DiscoveryTask
    generator_script: str


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------


class DifferentialDiscoveryOperator(BaseEvolutionaryOperator[TestIndividual]):
    """Self-sufficient operator for differential test discovery.

    On each execute() call runs the full 3-phase pipeline and returns
    all discovered TestIndividuals. The Breeder accumulates across calls.

    The 'parent_selector' argument is accepted for interface uniformity but
    is not used — pair selection is handled via a functional-equivalence
    selector instead.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        language_adapter: ILanguage,
        parent_selector: IParentSelectionStrategy[TestIndividual],
        prob_assigner: IProbabilityAssigner,
        llm_service: DifferentialLLMOperator,
        differential_finder: IDifferentialFinder,
        func_eq_selector: IFunctionallyEquivalentCodeSelector,
        divergence_limit: int = 5,
        max_pairs_per_group: int = 5,
        num_passing_tests_to_sample: int = 5,
        llm_workers: int = 4,
    ) -> None:
        super().__init__(llm, language_adapter, parent_selector, prob_assigner)
        self.llm_service = llm_service
        self.differential_finder = differential_finder
        self.func_eq_selector = func_eq_selector
        self.divergence_limit = divergence_limit
        self.max_pairs_per_group = max_pairs_per_group
        self.num_passing_tests_to_sample = num_passing_tests_to_sample
        self.llm_workers = llm_workers
        self._explored_pairs_cache: set[tuple[str, str]] = set()

    def operation_name(self) -> str:
        return OPERATION_DISCOVERY

    def reset_explored_pairs(self) -> None:
        """Clear the explored-pairs cache. Call between problems."""
        self._explored_pairs_cache.clear()
        logger.info("DifferentialDiscoveryOperator: explored-pairs cache reset")

    def execute(self, context: CoevolutionContext) -> list[TestIndividual]:
        """Run the full 3-phase differential discovery pipeline.

        Returns 0..N TestIndividuals per call. The Breeder accumulates
        across calls until its num_offsprings target is reached.
        """
        # Phase 1: candidate pair selection
        candidates = self._select_candidates(context)
        if not candidates:
            logger.warning("No unexplored pairs found for differential discovery")
            return []

        # Process up to max_pairs_per_group candidates per call
        selected = candidates[: self.max_pairs_per_group]
        logger.info(f"Phase 1: selected {len(selected)} pairs for this execution cycle")

        # Phase 2: generate scripts via LLM
        discovery_ctxs = self._batch_generate_scripts(context, selected)
        logger.info(f"Phase 2: generated {len(discovery_ctxs)} scripts")
        if not discovery_ctxs:
            return []

        # Phase 3: run divergence finding
        # Output naturally bounded by max_pairs_per_group × divergence_limit × 2 scenarios
        offspring = self._batch_find_divergences(context, discovery_ctxs)
        logger.info(f"Phase 3: produced {len(offspring)} test individuals")

        return offspring

    # ------------------------------------------------------------------
    # Phase 1: Candidate selection (Round-Robin Group Scheduling)
    # ------------------------------------------------------------------

    def _select_candidates(self, context: CoevolutionContext) -> list[_DiscoveryTask]:
        groups = self.func_eq_selector.select_functionally_equivalent_codes(context)
        logger.info(f"Found {len(groups)} functionally equivalent code groups")

        candidates_by_group: list[list[_DiscoveryTask]] = []

        for group in groups:
            inds = group.code_individuals
            if len(inds) < 2:
                continue

            sorted_inds = sorted(inds, key=lambda x: x.id)
            group_pairs: list[_DiscoveryTask] = [
                _DiscoveryTask(code_a=sorted_inds[i], code_b=sorted_inds[j], group=group)
                for i in range(len(sorted_inds))
                for j in range(i + 1, len(sorted_inds))
                if not self._is_pair_explored(sorted_inds[i], sorted_inds[j])
            ]

            if group_pairs:
                group_pairs.sort(
                    key=lambda t: t.code_a.probability + t.code_b.probability,
                    reverse=True,
                )
                candidates_by_group.append(group_pairs[: self.max_pairs_per_group])

        # Round-robin interleave across groups
        interleaved: list[_DiscoveryTask] = []
        while candidates_by_group:
            for group_list in list(candidates_by_group):
                if group_list:
                    interleaved.append(group_list.pop(0))
                if not group_list:
                    candidates_by_group.remove(group_list)

        return interleaved

    # ------------------------------------------------------------------
    # Phase 2: Script generation (I/O-bound, parallel)
    # ------------------------------------------------------------------

    def _generate_single_script(
        self, context: CoevolutionContext, task: _DiscoveryTask
    ) -> Optional[_DiscoveryContext]:
        try:
            all_snippets = [
                t.snippet
                for tests in task.group.passing_test_individuals.values()
                for t in tests
            ]
            passing = sample(
                all_snippets, min(len(all_snippets), self.num_passing_tests_to_sample)
            )

            dto = DifferentialGenScriptInput(
                equivalent_code_snippet_1=task.code_a.snippet,
                equivalent_code_snippet_2=task.code_b.snippet,
                passing_test_cases=passing,
                num_inputs_to_generate=100,
                question_content=context.problem.question_content,
            )
            script = self.llm_service.generate_script(dto)
            logger.debug(f"Script generated for pair ({task.code_a.id}, {task.code_b.id})")
            return _DiscoveryContext(task=task, generator_script=script)
        except Exception as e:
            logger.error(f"Script generation failed for {task.code_a.id} vs {task.code_b.id}: {e}")
            return None

    def _batch_generate_scripts(
        self, context: CoevolutionContext, tasks: list[_DiscoveryTask]
    ) -> list[_DiscoveryContext]:
        results: list[_DiscoveryContext] = []

        with ThreadPoolExecutor(max_workers=self.llm_workers) as executor:
            future_to_task = {
                executor.submit(self._generate_single_script, context, task): task
                for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    ctx = future.result()
                    if ctx:
                        self._mark_pair_explored(task.code_a, task.code_b)
                        results.append(ctx)
                except Exception as e:
                    logger.error(f"Script generation task failed: {e}")

        return results

    # ------------------------------------------------------------------
    # Phase 3: Divergence finding (CPU-bound, sequential — finder handles internal parallelism)
    # ------------------------------------------------------------------

    def _batch_find_divergences(
        self,
        context: CoevolutionContext,
        discovery_ctxs: list[_DiscoveryContext],
    ) -> list[TestIndividual]:
        offspring: list[TestIndividual] = []

        for ctx in discovery_ctxs:
            task = ctx.task
            try:
                divergences = self.differential_finder.find_differential(
                    task.code_a.snippet,
                    task.code_b.snippet,
                    ctx.generator_script,
                    limit=self.divergence_limit,
                )
                if not divergences:
                    logger.debug(f"No divergences for pair ({task.code_a.id}, {task.code_b.id})")
                    continue

                # Probability: inherit from existing differential test parents if any
                parent_probs = [
                    ind.probability
                    for tests in task.group.passing_test_individuals.values()
                    for ind in tests
                    if hasattr(ind, "creation_op") and ind.creation_op == OPERATION_DISCOVERY
                ]
                prob = (
                    self.prob_assigner.assign_probability(OPERATION_DISCOVERY, parent_probs)
                    if parent_probs
                    else self.prob_assigner.initial_prior
                )

                new_tests = self._create_divergence_tests(context, task, divergences, prob)
                offspring.extend(new_tests)
                logger.debug(
                    f"Pair ({task.code_a.id}, {task.code_b.id}): "
                    f"{len(divergences)} divergences → {len(new_tests)} tests"
                )
            except Exception as e:
                logger.error(f"Divergence finding failed for {task.code_a.id} vs {task.code_b.id}: {e}")

        return offspring

    # ------------------------------------------------------------------
    # Individual construction
    # ------------------------------------------------------------------

    def _create_divergence_tests(
        self,
        context: CoevolutionContext,
        task: _DiscoveryTask,
        divergences: list[DifferentialResult],
        probability: float,
    ) -> list[TestIndividual]:
        """Create two competing TestIndividuals per divergence (one per hypothesis).

        Scenario A: code_a's outputs are correct.
        Scenario B: code_b's outputs are correct.
        """
        code_a, code_b = task.code_a, task.code_b
        results: list[TestIndividual] = []

        scenarios: list[tuple[CodeIndividual, CodeIndividual, list[DifferentialInputOutput]]] = [
            (code_a, code_b, [{"inputdata": d.input_data, "output": d.output_a} for d in divergences]),
            (code_b, code_a, [{"inputdata": d.input_data, "output": d.output_b} for d in divergences]),
        ]

        for winner, loser, io_pairs in scenarios:
            for i, io_pair in enumerate(io_pairs):
                snippet = self.llm_service.get_test_method_from_io(
                    context.problem.starter_code,
                    [io_pair],
                    [winner.id, loser.id],
                    i,
                )
                results.append(
                    TestIndividual(
                        snippet=snippet,
                        probability=probability,
                        creation_op=OPERATION_DISCOVERY,
                        generation_born=context.code_population.generation + 1,
                        parents={"code": [winner.id, loser.id], "test": []},
                        metadata={"io_pair": io_pair},
                    )
                )

        logger.debug(f"Created {len(results)} divergence tests for pair ({code_a.id}, {code_b.id})")
        return results

    # ------------------------------------------------------------------
    # Explored-pairs cache helpers
    # ------------------------------------------------------------------

    def _mark_pair_explored(self, a: CodeIndividual, b: CodeIndividual) -> None:
        self._explored_pairs_cache.add(cast(tuple[str, str], tuple(sorted((a.id, b.id)))))

    def _is_pair_explored(self, a: CodeIndividual, b: CodeIndividual) -> bool:
        return tuple(sorted((a.id, b.id))) in self._explored_pairs_cache


# ---------------------------------------------------------------------------
# Initializer (differential always starts empty)
# ---------------------------------------------------------------------------


class DifferentialInitializer(BaseLLMInitializer[TestIndividual]):
    """Differential tests start empty — Gen 0 is always []."""

    def initialize(self, problem: Problem) -> list[TestIndividual]:
        logger.debug("DifferentialInitializer: starting with empty population")
        return []


__all__ = [
    "DifferentialDiscoveryOperator",
    "DifferentialInitializer",
    "FunctionallyEquivGroup",
    "IFunctionallyEquivalentCodeSelector",
    "DifferentialResult",
    "IDifferentialFinder",
    "OPERATION_DISCOVERY",
]
