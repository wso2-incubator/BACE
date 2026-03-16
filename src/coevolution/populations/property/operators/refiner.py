"""AdversarialPropertyRefiner — refines property tests via counter-examples."""

from __future__ import annotations

import json
import re
import threading
from typing import TYPE_CHECKING

from loguru import logger

from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import CoevolutionContext, Problem
from coevolution.strategies.llm_base import (
    BaseLLMOperator,
    ILanguageModel,
    LLMGenerationError,
    LLMSyntaxError,
    llm_retry,
)

from .helpers import transform_public_tests

if TYPE_CHECKING:
    from coevolution.core.interfaces.language import ICodeParser
    from coevolution.core.interfaces.probability import IProbabilityAssigner
    from coevolution.core.interfaces.selection import IParentSelectionStrategy
    from coevolution.core.population import TestPopulation


class AdversarialPropertyRefiner(BaseLLMOperator[TestIndividual]):
    """Refines property tests using a two-phase CEGIS-like approach.

    Phase 1: Generate a counter-example (input_arg, output) that invalidates the
             current property test but is actually correct according to the problem.
    Phase 2: Refine the property test snippet to correctly handle the counter-example.
    """

    _lock = threading.Lock()
    _in_flight: set[str] = set()

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        parent_selector: IParentSelectionStrategy[TestIndividual],
        prob_assigner: IProbabilityAssigner,
        max_falsification_attempts: int = 3,
    ) -> None:
        super().__init__(llm, parser, language_name, parent_selector, prob_assigner)
        self.max_falsification_attempts = max_falsification_attempts

    def operation_name(self) -> str:
        return "adversarial_refinement"

    def execute(self, context: CoevolutionContext) -> list[TestIndividual]:
        # 1. Access population
        pop = context.test_populations.get("property")
        if not pop or pop.size == 0:
            return []

        # 2. Selection Loop (Re-roll strategy to prevent worker starvation)
        # We try up to 5 times to find an available parent.
        # Selection happens OUTSIDE the lock to allow parallel decision making.
        parent = None
        for _ in range(5):
            parents = self.parent_selector.select_parents(
                pop, count=1, coevolution_context=context
            )
            if not parents:
                break
            candidate = parents[0]

            with self._lock:
                # Is someone else already working on this individual?
                if candidate.id in self._in_flight:
                    continue

                # Check retry limit
                attempts = candidate.metadata.get("falsification_attempts", 0)
                if attempts >= self.max_falsification_attempts:
                    continue

                # Successfully claimed!
                self._in_flight.add(candidate.id)
                parent = candidate
                break

        if not parent:
            return []

        try:
            return self._do_execute(parent, context, pop)
        finally:
            with self._lock:
                self._in_flight.remove(parent.id)

    def _do_execute(
        self, parent: TestIndividual, context: CoevolutionContext, pop: TestPopulation
    ) -> list[TestIndividual]:
        # 3. Phase 1: Falsification
        problem = context.problem
        falsification_attempts = parent.metadata.get("falsification_attempts", 0)

        try:
            ce_data = self._generate_counter_example(parent, problem)
        except Exception as exc:
            logger.debug(
                f"AdversarialPropertyRefiner: counter-example generation failed: {exc}"
            )
            # Increment attempts on failure to generate CE (including LLM errors)
            with self._lock:
                parent.metadata["falsification_attempts"] = falsification_attempts + 1
            return []

        if not ce_data:
            logger.debug(
                f"AdversarialPropertyRefiner: no counter-example found for {parent.id}"
            )
            # Increment attempts when no CE is found
            with self._lock:
                parent.metadata["falsification_attempts"] = falsification_attempts + 1
            return []

        counter_example_json, reasoning = ce_data

        # 3. Phase 2: Refine property
        current_explanations = [
            ind.explanation for ind in pop if ind.id != parent.id and ind.explanation
        ]
        try:
            refined_snippet = self._refine_property(
                parent, counter_example_json, reasoning, problem, current_explanations
            )
        except Exception as exc:
            logger.debug(
                f"AdversarialPropertyRefiner: property refinement failed: {exc}"
            )
            # Note: We don't necessarily increment falsification_attempts here because we DID find a CE.
            return []

        # 4. Create offspring
        offspring_prob = self.prob_assigner.assign_probability(
            self.operation_name(), [parent.probability]
        )

        return [
            TestIndividual(
                snippet=refined_snippet,
                probability=offspring_prob,
                creation_op=self.operation_name(),
                generation_born=pop.generation + 1,
                parents={"code": [], "test": [parent.id]},
                explanation=self.parser.get_docstring(refined_snippet),
                metadata={
                    "refined_from": parent.id,
                    "counter_example": json.loads(counter_example_json),
                    "reasoning": reasoning,
                    "falsification_attempts": 0,  # Reset for the new child
                },
            )
        ]

    @llm_retry((LLMGenerationError, ValueError))
    def _generate_counter_example(
        self, parent: TestIndividual, problem: Problem
    ) -> tuple[str, str] | None:
        """
        Phase 1: Generate a counter-example and reasoning for a property test.
        Returns (counter_example_json, reasoning_str) or None.
        """
        public_tests = transform_public_tests(
            problem.public_test_cases, problem.starter_code, self.parser
        )
        prompt = self.prompt_manager.render_prompt(
            "operators/property/gen_counter_example.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
            snippet=parent.snippet,
            public_tests=public_tests,
        )
        response = self._generate(prompt)

        # Extract reasoning
        reasoning_match = re.search(
            r"<reasoning>(.*?)</reasoning>", response, re.DOTALL
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        # Extract counter-example
        ce_match = re.search(
            r"<counter_example>(.*?)</counter_example>", response, re.DOTALL
        )
        if not ce_match:
            if reasoning:
                logger.debug(
                    f"AdversarialPropertyRefiner: no counter-example found. Reasoning: {reasoning}"
                )
            return None

        content = ce_match.group(1).strip()
        try:
            # Validate it's valid JSON and contains required keys
            ce_data = json.loads(content)
            if "input_arg" not in ce_data or "output" not in ce_data:
                logger.warning(
                    "AdversarialPropertyRefiner: counter-example JSON missing keys."
                )
                raise ValueError(
                    "AdversarialPropertyRefiner: counter-example JSON missing required keys."
                )
            logger.trace(
                f"Counter-example generation prompt for {parent.id}\n: {prompt}"
            )
            logger.trace(
                f"Counter-example generation response for {parent.id}\n: {response}"
            )
            return content, reasoning
        except json.JSONDecodeError:
            logger.warning(
                "AdversarialPropertyRefiner: counter-example JSON decode error."
            )
            raise ValueError(
                "AdversarialPropertyRefiner: counter-example JSON could not be decoded."
            )

    @llm_retry((LLMGenerationError, LLMSyntaxError, ValueError))
    def _refine_property(
        self,
        parent: TestIndividual,
        counter_example: str,
        reasoning: str,
        problem: Problem,
        current_explanations: list[str],
    ) -> str:
        """
        Phase 2: Refine a property test using a counter-example and reasoning.
        """
        public_tests = transform_public_tests(
            problem.public_test_cases, problem.starter_code, self.parser
        )
        prompt = self.prompt_manager.render_prompt(
            "operators/property/refine_property.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
            snippet=parent.snippet,
            counter_example=counter_example,
            reasoning=reasoning,
            current_explanations=current_explanations,
            public_tests=public_tests,
        )
        response = self._generate(prompt)
        blocks = self.parser.extract_code_blocks(response)
        if not blocks:
            raise LLMGenerationError("refine_property: no code blocks found.")

        snippet = blocks[0]
        snippet = self.parser.remove_main_block(snippet)
        if not self.parser.is_syntax_valid(snippet):
            raise LLMSyntaxError("refine_property: invalid syntax")

        logger.trace(f"Refine property prompt for {parent.id}\n: {prompt}")
        logger.trace(f"Refine property response for {parent.id}\n: {response}")

        return snippet


__all__ = ["AdversarialPropertyRefiner"]
