# tests/coevolution/operators/test_agent_coder_llm_operator.py
from unittest.mock import MagicMock

import pytest

from coevolution.core.interfaces import InitialInput
from coevolution.strategies.operators.agent_coder_llm_operator import (
    AgentCoderEditInput,
    AgentCoderLLMOperator,
)
from coevolution.utils.prompts import AGENT_CODER_PROGRAMMER_INIT
from infrastructure.llm_client import LLMClient  # Assuming this exists for typing


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Mock the LLM client to return predictable responses."""
    client = MagicMock(spec=LLMClient)
    # Default behavior: return a valid code block
    client.generate.return_value = "```python\ndef solution():\n    pass\n```"
    return client


@pytest.fixture
def operator(mock_llm_client: MagicMock) -> AgentCoderLLMOperator:
    """Create an operator instance with the mock client."""
    return AgentCoderLLMOperator(llm=mock_llm_client)


@pytest.fixture
def valid_initial_input() -> InitialInput:
    return InitialInput(
        operation="initial",
        question_content="Write a function to add two numbers.",
        starter_code="def solution():\n    pass",
        population_size=1,
    )


class TestAgentCoderLifecycle:
    """Tests the state management (Memory) of the Agent."""

    def test_initialization_starts_fresh_session(
        self,
        operator: AgentCoderLLMOperator,
        mock_llm_client: MagicMock,
        valid_initial_input: InitialInput,
    ) -> None:
        """Turn 0: Verify initialization seeds history correctly."""

        # Act
        output, _ = operator.generate_initial_snippets(valid_initial_input)

        # Assert 1: Result is returned
        assert len(output.results) == 1
        assert output.results[0].snippet == "def solution():\n    pass"

        # Assert 2: History contains exactly 2 messages (User Prompt + Assistant Reply)
        assert len(operator._conversation_history) == 2
        assert operator._conversation_history[0]["role"] == "user"
        assert operator._conversation_history[1]["role"] == "assistant"

        # Assert 3: Prompt format
        expected_prompt = AGENT_CODER_PROGRAMMER_INIT.format(
            question_content=valid_initial_input.question_content,
            starter_code=valid_initial_input.starter_code,
        )
        assert operator._conversation_history[0]["content"] == expected_prompt

    def test_edit_appends_to_history(
        self,
        operator: AgentCoderLLMOperator,
        mock_llm_client: MagicMock,
        valid_initial_input: InitialInput,
    ) -> None:
        """Turn 1: Verify edit adds context to the existing session."""

        # Setup: Initialize first (Turn 0)
        operator.generate_initial_snippets(valid_initial_input)

        # Prepare Turn 1 Input
        edit_input = AgentCoderEditInput(
            operation="edit",
            current_snippet="def solution():\n    pass",
            failing_tests_with_trace=[("test_1", "Error: 1 != 2")],
            starter_code="def solution():\n    pass",
            question_content="Write a function to add two numbers.",
        )

        # Act: Apply Edit
        mock_llm_client.generate.return_value = (
            "```python\ndef solution():\n    return 2\n```"
        )
        operator.apply(edit_input)

        # Assert: History grew to 4 messages (User0, Asst0, User1, Asst1)
        assert len(operator._conversation_history) == 4
        assert operator._conversation_history[2]["role"] == "user"
        assert operator._conversation_history[3]["role"] == "assistant"

        # Verify the Feedback Prompt construction
        last_user_msg = operator._conversation_history[2]["content"]
        assert "Error Trace:\nError: 1 != 2" in last_user_msg

    def test_reset_session_clears_history(
        self, operator: AgentCoderLLMOperator, valid_initial_input: InitialInput
    ) -> None:
        """Verify explicit reset wipes memory."""
        # Setup
        operator.generate_initial_snippets(valid_initial_input)
        assert len(operator._conversation_history) > 0

        # Act
        operator.reset_session()

        # Assert
        assert len(operator._conversation_history) == 0

    def test_auto_reset_on_initialization(
        self,
        operator: AgentCoderLLMOperator,
        valid_initial_input: InitialInput,
        mock_llm_client: MagicMock,
    ) -> None:
        """Verify calling generate_initial_snippets automatically wipes previous session."""

        # Setup: Run a full session (Init + Edit)
        operator.generate_initial_snippets(valid_initial_input)
        # Manually dirty the history to simulate a long session
        operator._conversation_history.append({"role": "user", "content": "junk"})
        assert len(operator._conversation_history) == 3

        # Act: Start a NEW problem (Initialize again)
        operator.generate_initial_snippets(valid_initial_input)

        # Assert: History should be reset to just 2 messages (User + Assistant)
        assert len(operator._conversation_history) == 2


class TestAgentCoderConstraints:
    """Tests input validation and safety checks."""

    def test_rejects_population_gt_1(
        self, operator: AgentCoderLLMOperator, valid_initial_input: InitialInput
    ) -> None:
        """AgentCoder is strictly linear (pop=1)."""
        invalid_input = InitialInput(
            operation="initial",
            question_content="...",
            starter_code="...",
            population_size=10,  # Invalid
        )

        with pytest.raises(
            ValueError, match="AgentCoder only supports population_size=1"
        ):
            operator.generate_initial_snippets(invalid_input)

    def test_rejects_edit_without_session(
        self, operator: AgentCoderLLMOperator
    ) -> None:
        """Cannot edit if the agent has no memory of the code."""
        # No initialization called
        operator.reset_session()

        edit_input = AgentCoderEditInput(
            operation="edit",
            current_snippet="...",
            failing_tests_with_trace=[],
            starter_code="...",
            question_content="...",
        )

        with pytest.raises(ValueError, match="No existing session found"):
            operator.apply(edit_input)

    def test_validates_starter_code_structure(
        self,
        operator: AgentCoderLLMOperator,
        mock_llm_client: MagicMock,
        valid_initial_input: InitialInput,
    ) -> None:
        """Ensure generated code maintains the required structure."""
        # Mock LLM returning code WITHOUT the starter function
        mock_llm_client.generate.return_value = (
            "```python\ndef random_func(): pass\n```"
        )

        with pytest.raises(ValueError, match="Generated code missing starter code"):
            operator.generate_initial_snippets(valid_initial_input)

    def test_handles_multiple_code_blocks(
        self,
        operator: AgentCoderLLMOperator,
        mock_llm_client: MagicMock,
        valid_initial_input: InitialInput,
    ) -> None:
        """
        If LLM returns pseudocode block first and real code second,
        it should pick the last block.
        """
        # Mock response with two blocks
        mock_llm_client.generate.return_value = """
        Here is the plan:
        ```text
        pseudocode...
        ```
        And here is the code:
        ```python
        def solution():
            return 42
        ```
        """

        output, _ = operator.generate_initial_snippets(valid_initial_input)

        # Should pick the python block, not the text block
        assert "return 42" in output.results[0].snippet
