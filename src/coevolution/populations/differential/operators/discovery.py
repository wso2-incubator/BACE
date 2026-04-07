"""DifferentialDiscoveryOperator — full 3-phase differential pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from random import sample
from typing import Optional, cast

from loguru import logger

from coevolution.core.individual import CodeIndividual, TestIndividual
from coevolution.core.interfaces import CoevolutionContext
from coevolution.core.interfaces.language import ICodeParser
from coevolution.core.interfaces.probability import IProbabilityAssigner
from coevolution.core.interfaces.selection import IParentSelectionStrategy

from coevolution.strategies.llm_base import BaseLLMOperator, ILanguageModel
from ..types import (
    OPERATION_DISCOVERY,
    DifferentialResult,
    FunctionallyEquivGroup,
    IDifferentialFinder,
    IFunctionallyEquivalentCodeSelector,
)
from .llm_operator import (
    DifferentialGenScriptInput,
    DifferentialInputOutput,
    DifferentialLLMOperator,
)


@dataclass
class _DiscoveryTask:
    code_a: CodeIndividual
    code_b: CodeIndividual
    group: FunctionallyEquivGroup


@dataclass
class _DiscoveryContext:
    task: _DiscoveryTask
    generator_script: str


class DifferentialDiscoveryOperator(BaseLLMOperator[TestIndividual]):
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
        parser: ICodeParser,
        language_name: str,
        parent_selector: IParentSelectionStrategy[TestIndividual],
        prob_assigner: IProbabilityAssigner,
        llm_service: DifferentialLLMOperator,
        differential_finder: IDifferentialFinder,
        func_eq_selector: IFunctionallyEquivalentCodeSelector,
        divergence_limit: int = 5,
        max_pairs_per_group: int = 5,
        num_passing_tests_to_sample: int = 5,
        llm_workers: int = 4,
        pair_workers: int = 1,
    ) -> None:
        super().__init__(llm, parser, language_name, parent_selector, prob_assigner)
        self.llm_service = llm_service
        self.differential_finder = differential_finder
        self.func_eq_selector = func_eq_selector
        self.divergence_limit = divergence_limit
        self.max_pairs_per_group = max_pairs_per_group
        self.num_passing_tests_to_sample = num_passing_tests_to_sample
        self.llm_workers = llm_workers
        self.pair_workers = pair_workers
        self._explored_pairs_cache: set[tuple[str, str]] = set()

    def operation_name(self) -> str:
        return OPERATION_DISCOVERY

    def reset_explored_pairs(self) -> None:
        """Clear the explored-pairs cache. Call between problems."""
        self._explored_pairs_cache.clear()
        logger.info("DifferentialDiscoveryOperator: explored-pairs cache reset")

    def execute(self, context: CoevolutionContext) -> list[TestIndividual]:
        """Run the full 3-phase differential discovery pipeline."""
        # Phase 1: candidate pair selection
        candidates = self._select_candidates(context)
        if not candidates:
            logger.warning("No unexplored pairs found for differential discovery")
            return []

        selected = candidates[: self.max_pairs_per_group]
        logger.info(
            f"Phase 1: selected {len(selected)} pairs for this execution cycle "
            f"(pair_workers={self.pair_workers})"
        )

        # Phase 2: generate scripts via LLM
        discovery_ctxs = self._batch_generate_scripts(context, selected)
        logger.info(f"Phase 2: generated {len(discovery_ctxs)} scripts")
        if not discovery_ctxs:
            return []

        # Phase 3: run divergence finding
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
                _DiscoveryTask(
                    code_a=sorted_inds[i], code_b=sorted_inds[j], group=group
                )
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
    # Phase 2: Script generation
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
            logger.debug(
                f"Script generated for pair ({task.code_a.id}, {task.code_b.id})"
            )
            return _DiscoveryContext(task=task, generator_script=script)
        except Exception as e:
            logger.error(
                f"Script generation failed for {task.code_a.id} vs {task.code_b.id}: {e}"
            )
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
    # Phase 3: Divergence finding
    # ------------------------------------------------------------------

    def _find_divergences_for_pair(
        self,
        context: CoevolutionContext,
        ctx: _DiscoveryContext,
    ) -> list[TestIndividual]:
        """Run find_differential for a single pair and return TestIndividuals.

        Designed to be submitted to a ThreadPoolExecutor so that multiple
        pairs execute concurrently (Level-1 parallelism). Each call may
        internally spawn a multiprocessing.Pool with `workers_per_pair`
        processes (Level-2 parallelism).
        """
        task = ctx.task
        all_divergences = self.differential_finder.find_differential(
            task.code_a.snippet,
            task.code_b.snippet,
            ctx.generator_script,
            limit=self.divergence_limit,
        )
        divergences = [d for d in all_divergences if d.output_a != d.output_b]

        if not divergences:
            logger.debug(
                f"No divergences for pair ({task.code_a.id}, {task.code_b.id})"
            )
            return []

        parent_probs = [
            ind.probability
            for tests in task.group.passing_test_individuals.values()
            for ind in tests
            if hasattr(ind, "creation_op")
            and ind.creation_op == OPERATION_DISCOVERY
        ]
        prob = (
            self.prob_assigner.assign_probability(
                OPERATION_DISCOVERY, parent_probs
            )
            if parent_probs
            else self.prob_assigner.initial_prior
        )

        new_tests = self._create_divergence_tests(context, task, divergences, prob)
        logger.debug(
            f"Pair ({task.code_a.id}, {task.code_b.id}): "
            f"{len(divergences)} divergences → {len(new_tests)} tests"
        )
        return new_tests

    def _batch_find_divergences(
        self,
        context: CoevolutionContext,
        discovery_ctxs: list[_DiscoveryContext],
    ) -> list[TestIndividual]:
        """Run divergence finding for all pairs concurrently (Level-1 parallelism).

        Each pair's find_differential() call runs in its own thread, so a
        slow or hanging pair does not block others. The inner multiprocessing.Pool
        inside find_differential() provides Level-2 (input-level) parallelism.
        """
        offspring: list[TestIndividual] = []
        with ThreadPoolExecutor(max_workers=self.pair_workers) as executor:
            future_to_ctx = {
                executor.submit(self._find_divergences_for_pair, context, ctx): ctx
                for ctx in discovery_ctxs
            }
            for future in as_completed(future_to_ctx):
                ctx = future_to_ctx[future]
                try:
                    new_tests = future.result()
                    offspring.extend(new_tests)
                except Exception as e:
                    logger.error(
                        f"Divergence finding failed for pair "
                        f"({ctx.task.code_a.id}, {ctx.task.code_b.id}): {e}"
                    )
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
        """Create two competing TestIndividuals per divergence (one per hypothesis)."""
        code_a, code_b = task.code_a, task.code_b
        results: list[TestIndividual] = []

        scenarios: list[
            tuple[CodeIndividual, CodeIndividual, list[DifferentialInputOutput]]
        ] = [
            (
                code_a,
                code_b,
                [
                    {"input_arg": d.input_data, "output": d.output_a}
                    for d in divergences
                ],
            ),
            (
                code_b,
                code_a,
                [
                    {"input_arg": d.input_data, "output": d.output_b}
                    for d in divergences
                ],
            ),
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
                        explanation=self.parser.get_docstring(snippet),
                        metadata={"io_pair": io_pair},
                    )
                )

        logger.debug(
            f"Created {len(results)} divergence tests for pair ({code_a.id}, {code_b.id})"
        )
        return results

    # ------------------------------------------------------------------
    # Explored-pairs cache helpers
    # ------------------------------------------------------------------

    def _mark_pair_explored(self, a: CodeIndividual, b: CodeIndividual) -> None:
        self._explored_pairs_cache.add(
            cast(tuple[str, str], tuple(sorted((a.id, b.id))))
        )

    def _is_pair_explored(self, a: CodeIndividual, b: CodeIndividual) -> bool:
        return tuple(sorted((a.id, b.id))) in self._explored_pairs_cache


__all__ = ["DifferentialDiscoveryOperator"]
