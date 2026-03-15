"""AdversarialPropertyRefiner — refines property tests via counter-examples."""

from __future__ import annotations

import json
import re
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
from infrastructure.languages import PythonLanguage

if TYPE_CHECKING:
    from coevolution.core.interfaces.language import ICodeParser
    from coevolution.core.interfaces.probability import IProbabilityAssigner
    from coevolution.core.interfaces.selection import IParentSelectionStrategy


class AdversarialPropertyRefiner(BaseLLMOperator[TestIndividual]):
    """Refines property tests using a two-phase CEGIS-like approach.

    Phase 1: Generate a counter-example (inputdata, output) that invalidates the
             current property test but is actually correct according to the problem.
    Phase 2: Refine the property test snippet to correctly handle the counter-example.
    """

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
        self._python_lang = PythonLanguage()

    def operation_name(self) -> str:
        return "adversarial_refinement"

    def execute(self, context: CoevolutionContext) -> list[TestIndividual]:
        # 1. Select a property test to refine
        pop = context.test_populations.get("property")
        if not pop or pop.size == 0:
            return []

        parents = self.parent_selector.select_parents(
            pop, count=1, coevolution_context=context
        )
        if not parents:
            return []

        parent = parents[0]

        # Check retry limit
        attempts = parent.metadata.get("falsification_attempts", 0)
        if attempts >= self.max_falsification_attempts:
            logger.debug(
                f"AdversarialPropertyRefiner: giving up on {parent.id} "
                f"after {attempts} failed falsification attempts."
            )
            return []

        problem = context.problem

        # 2. Phase 1: Generate counter-example and reasoning
        try:
            ce_data = self._generate_counter_example(parent, problem)
        except Exception as exc:
            logger.debug(
                f"AdversarialPropertyRefiner: counter-example generation failed: {exc}"
            )
            # Increment attempts on failure to generate CE (including LLM errors)
            parent.metadata["falsification_attempts"] = attempts + 1
            return []

        if not ce_data:
            logger.debug(
                f"AdversarialPropertyRefiner: no counter-example found for {parent.id}"
            )
            # Increment attempts when no CE is found
            parent.metadata["falsification_attempts"] = attempts + 1
            return []

        counter_example_json, reasoning = ce_data

        # 3. Phase 2: Refine property
        try:
            refined_snippet = self._refine_property(
                parent, counter_example_json, reasoning, problem
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
                generation_born=context.code_population.generation,
                parents={"test": [parent.id]},
                explanation=f"Refined {parent.id} using counter-example and reasoning.",
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
        prompt = self.prompt_manager.render_prompt(
            "operators/property/gen_counter_example.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
            snippet=parent.snippet,
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
            if "inputdata" not in ce_data or "output" not in ce_data:
                logger.warning(
                    "AdversarialPropertyRefiner: counter-example JSON missing keys."
                )
                return None
            return content, reasoning
        except json.JSONDecodeError:
            logger.warning(
                "AdversarialPropertyRefiner: counter-example JSON decode error."
            )
            return None

    @llm_retry((LLMGenerationError, LLMSyntaxError, ValueError))
    def _refine_property(
        self,
        parent: TestIndividual,
        counter_example: str,
        reasoning: str,
        problem: Problem,
    ) -> str:
        """
        Phase 2: Refine a property test using a counter-example and reasoning.
        """
        prompt = self.prompt_manager.render_prompt(
            "operators/property/refine_property.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
            snippet=parent.snippet,
            counter_example=counter_example,
            reasoning=reasoning,
        )
        response = self._generate(prompt)
        blocks = self.parser.extract_code_blocks(response)
        if not blocks:
            raise LLMGenerationError("refine_property: no code blocks found.")

        snippet = blocks[0]
        snippet = self.parser.remove_main_block(snippet)
        if not self.parser.is_syntax_valid(snippet):
            raise LLMSyntaxError("refine_property: invalid syntax")

        return snippet


__all__ = ["AdversarialPropertyRefiner"]
