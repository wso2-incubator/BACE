"""AgentCoderInitializer — Turn 0 of the AgentCoder loop."""

from __future__ import annotations


from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import (
    OPERATION_INITIAL,
    PopulationConfig,
    Problem,
)
from coevolution.core.interfaces.language import (
    ICodeParser,
    LanguageParsingError,
    LanguageTransformationError,
)

from coevolution.strategies.llm_base import (
    BaseLLMInitializer,
    ILanguageModel,
    LLMGenerationError,
    llm_retry,
)
from .edit import AgentCoderEditOperator


class AgentCoderInitializer(BaseLLMInitializer[CodeIndividual]):
    """Turn 0 of the AgentCoder loop: generates the first solution and seeds history.

    MUST be called before AgentCoderEditOperator.execute().
    Pass the same operator instance to both so they share conversation history.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        parser: ICodeParser,
        language_name: str,
        pop_config: PopulationConfig,
        edit_operator: AgentCoderEditOperator,
    ) -> None:
        super().__init__(llm, parser, language_name, pop_config)
        if pop_config.initial_population_size != 1:
            raise ValueError("AgentCoder only supports initial_population_size=1")
        self._edit_operator = edit_operator

    def initialize(self, problem: Problem) -> list[CodeIndividual]:
        self._edit_operator.reset_session()
        return [self._generate_initial(problem)]

    @llm_retry((ValueError, LanguageParsingError, LanguageTransformationError, LLMGenerationError))
    def _generate_initial(self, problem: Problem) -> CodeIndividual:
        prompt = self.prompt_manager.render_prompt(
            "operators/agent_coder/init.j2",
            question_content=problem.question_content,
            starter_code=problem.starter_code,
        )
        self._edit_operator._conversation_history.append({"role": "user", "content": prompt})
        response = self._generate(self._edit_operator._conversation_history)
        self._edit_operator._conversation_history.append({"role": "assistant", "content": response})

        code_blocks = self.parser.extract_code_blocks(response)
        if not code_blocks:
            raise ValueError("AgentCoderInitializer: no code block in LLM response")
        code = code_blocks[-1]

        if not self.parser.contains_starter_code(code, problem.starter_code):
            raise ValueError("AgentCoderInitializer: generated code missing starter code")

        return CodeIndividual(
            snippet=code,
            probability=self.pop_config.initial_prior,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )


__all__ = ["AgentCoderInitializer"]
