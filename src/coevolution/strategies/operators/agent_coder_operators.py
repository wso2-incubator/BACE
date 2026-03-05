"""Concrete AgentCoder Operator implementations.

AgentCoder uses a stateful single-agent loop (population size = 1):
- Turn 0: AgentCoderInitializer generates the first solution, seeding conversation history.
- Turn N: AgentCoderEditOperator repairs the current solution guided by failing tests.

The conversation history lives inside the operator between evaluations.
The Orchestrator must call reset_session() when switching to a new problem.
"""

from __future__ import annotations

from loguru import logger

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import (
    OPERATION_EDIT,
    OPERATION_INITIAL,
    CoevolutionContext,
    PopulationConfig,
    Problem,
)
from coevolution.core.interfaces.language import (
    ILanguage,
    LanguageParsingError,
    LanguageTransformationError,
)
from coevolution.core.interfaces.probability import IProbabilityAssigner
from coevolution.core.interfaces.selection import IParentSelectionStrategy

from .base_llm_service import (
    BaseEvolutionaryOperator,
    BaseLLMInitializer,
    ILanguageModel,
    LLMGenerationError,
    llm_retry,
)


class AgentCoderEditOperator(BaseEvolutionaryOperator[CodeIndividual]):
    """Stateful edit operator for the AgentCoder single-agent loop.

    Maintains conversation history across calls within one problem.
    Call reset_session() between problems.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        language_adapter: ILanguage,
        parent_selector: IParentSelectionStrategy[CodeIndividual],
        prob_assigner: IProbabilityAssigner,
    ) -> None:
        super().__init__(llm, language_adapter, parent_selector, prob_assigner)
        self._conversation_history: list[dict[str, str]] = []

    def reset_session(self) -> None:
        """Wipe conversation memory. Call between problems."""
        logger.info("AgentCoderEditOperator: session reset")
        self._conversation_history.clear()

    def operation_name(self) -> str:
        return OPERATION_EDIT

    @llm_retry(
        (
            ValueError,
            LanguageParsingError,
            LanguageTransformationError,
            LLMGenerationError,
        )
    )
    def execute(self, context: CoevolutionContext) -> list[CodeIndividual]:
        code_pop = context.code_population
        problem = context.problem

        if len(code_pop) != 1:
            raise ValueError(
                f"AgentCoderEditOperator requires exactly 1 individual, got {len(code_pop)}"
            )
        code_parent = code_pop[0]

        # Gather failing tests from unittest interaction
        unittest_exec = context.interactions["unittest"].execution_results
        if code_parent.id not in unittest_exec:
            logger.warning(f"No execution results for {code_parent.id}")
            return [code_parent]

        test_results = unittest_exec[code_parent.id]
        unittest_pop = context.test_populations["unittest"]

        failing = [
            (t.snippet, test_results[t.id].error_log or "")
            for t in unittest_pop
            if test_results[t.id].status == "failed"
        ]

        if not failing:
            logger.info(
                "AgentCoderEditOperator: no failing tests — returning parent unchanged"
            )
            return [code_parent]

        if not self._conversation_history:
            raise ValueError(
                "AgentCoderEditOperator: no conversation history. "
                "Run AgentCoderInitializer first, or call reset_session()."
            )

        feedback = "\n\n".join(
            f"Test Case:\n{test}\n\nError Trace:\n{trace}" for test, trace in failing
        )
        prompt = self.prompt_manager.render_prompt(
            "operators/agent_coder/edit.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
            feedback=feedback,
        )
        self._conversation_history.append({"role": "user", "content": prompt})

        logger.debug(
            f"AgentCoder: repairing (history depth: {len(self._conversation_history)})"
        )
        response = self._generate(self._conversation_history)
        self._conversation_history.append({"role": "assistant", "content": response})

        edited_code = self._extract_code_block(response)
        if not self.language_adapter.contains_starter_code(
            edited_code, problem.starter_code
        ):
            raise ValueError("AgentCoder edit result does not contain starter code")

        probability = self.prob_assigner.assign_probability(
            OPERATION_EDIT, [code_parent.probability]
        )
        return [
            CodeIndividual(
                snippet=edited_code,
                probability=probability,
                creation_op=OPERATION_EDIT,
                generation_born=code_pop.generation + 1,
                parents={
                    "code": [code_parent.id],
                    "test": [
                        t.id
                        for t in unittest_pop
                        if test_results[t.id].status == "failed"
                    ],
                },
            )
        ]


class AgentCoderInitializer(BaseLLMInitializer[CodeIndividual]):
    """Turn 0 of the AgentCoder loop: generates the first solution and seeds history.

    MUST be called before AgentCoderEditOperator.execute().
    Pass the same operator instance to both so they share conversation history.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        language_adapter: ILanguage,
        pop_config: PopulationConfig,
        edit_operator: AgentCoderEditOperator,
    ) -> None:
        super().__init__(llm, language_adapter, pop_config)
        if pop_config.initial_population_size != 1:
            raise ValueError("AgentCoder only supports initial_population_size=1")
        self._edit_operator = edit_operator

    def initialize(self, problem: Problem) -> list[CodeIndividual]:
        self._edit_operator.reset_session()
        return [self._generate_initial(problem)]

    @llm_retry(
        (
            ValueError,
            LanguageParsingError,
            LanguageTransformationError,
            LLMGenerationError,
        )
    )
    def _generate_initial(self, problem: Problem) -> CodeIndividual:
        prompt = self.prompt_manager.render_prompt(
            "operators/agent_coder/init.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
        )
        # Seed conversation in the shared edit operator
        self._edit_operator._conversation_history.append(
            {"role": "user", "content": prompt}
        )
        response = self._generate(self._edit_operator._conversation_history)
        self._edit_operator._conversation_history.append(
            {"role": "assistant", "content": response}
        )

        # Agent coder prompt has pseudocode first, take the last code block
        code_blocks = self.language_adapter.extract_code_blocks(response)
        if not code_blocks:
            raise ValueError("AgentCoderInitializer: no code block in LLM response")
        code = code_blocks[-1]

        if not self.language_adapter.contains_starter_code(code, problem.starter_code):
            raise ValueError(
                "AgentCoderInitializer: generated code missing starter code"
            )

        return CodeIndividual(
            snippet=code,
            probability=self.pop_config.initial_prior,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )


__all__ = [
    "AgentCoderEditOperator",
    "AgentCoderInitializer",
]
