"""
Integration test for LLM operators with actual LLM clients.

This test verifies that LLM operators work correctly with real LLM clients.
It can use OpenAI, Ollama, or other configured LLM providers.

Requirements:
- For OpenAI: Set OPENAI_API_KEY environment variable
- For Ollama: Ensure Ollama is running locally with a model available

Run with pytest:
    pytest tests/integration/test_llm_operators_integration.py -v

Run as standalone script:
    python tests/integration/test_llm_operators_integration.py

Skip if no LLM available:
    pytest tests/integration/test_llm_operators_integration.py -v -k "not llm"

Known Issues:
- Some tests may fail due to strict starter code validation with simple function names.
  The `contains_starter_code` function uses fuzzy matching but may be too strict for
  very simple starter code like "def add(a, b): pass". This is a known limitation
  and the tests help identify these edge cases.
- LLM responses can be non-deterministic, so some tests may occasionally fail.
"""

import os
from typing import Any

import pytest
from loguru import logger

from common.coevolution.core.interfaces import Problem
from common.coevolution.llm_operators import CodeLLMOperator, TestLLMOperator
from common.llm_client import OllamaClient, OpenAIChatClient

# ----------------------------
# Fixtures and Setup
# ----------------------------


def get_available_llm_client() -> tuple[Any, str] | None:
    """
    Get an available LLM client for testing.

    Returns:
        Tuple of (client, client_name) or None if no client is available.
    """
    # Try OpenAI first
    if os.getenv("OPENAI_API_KEY"):
        try:
            client = OpenAIChatClient(
                model="gpt-4o-mini",  # Use cheaper model for testing
                max_output_tokens=10000,  # Limit tokens for testing
                enable_token_limit=True,
            )
            logger.info("Using OpenAI client for integration test")
            return client, "OpenAI"
        except Exception as e:
            logger.warning(f"Failed to create OpenAI client: {e}")

    # Try Ollama as fallback
    try:
        client = OllamaClient(
            model="qwen2.5-coder:7b",  # Common local model
            max_output_tokens=10000,
            enable_token_limit=True,
        )  # type: ignore
        # Test if Ollama is actually running
        try:
            client.generate("test", max_tokens=10)
            logger.info("Using Ollama client for integration test")
            return client, "Ollama"
        except Exception:
            logger.warning("Ollama client created but server not responding")
    except Exception as e:
        logger.warning(f"Failed to create Ollama client: {e}")

    return None


@pytest.fixture
def llm_client() -> Any:
    """Provide an LLM client for testing."""
    result = get_available_llm_client()
    if result is None:
        pytest.skip("No LLM client available (set OPENAI_API_KEY or run Ollama)")
    client, name = result
    logger.info(f"Using {name} client for tests")
    return client


@pytest.fixture
def simple_problem() -> Problem:
    """Create a simple problem for testing."""
    return Problem(
        question_title="Add Two Numbers",
        question_content="Write a function that adds two numbers and returns the result.",
        question_id="add-two-numbers",
        starter_code="def add(a, b):\n    pass",
        public_test_cases=[],
        private_test_cases=[],
    )


@pytest.fixture
def coding_problem() -> Problem:
    """Create a more realistic coding problem."""
    return Problem(
        question_title="Find Maximum",
        question_content="""Write a function that finds the maximum element in a list.
        
Requirements:
- Function name: find_max
- Input: A list of integers (non-empty)
- Output: The maximum integer in the list
- Handle negative numbers correctly

Example:
    find_max([1, 5, 3, 9, 2]) -> 9
    find_max([-5, -1, -10]) -> -1
""",
        question_id="find-maximum",
        starter_code="def find_max(nums):\n    pass",
        public_test_cases=[],
        private_test_cases=[],
    )


# ----------------------------
# CodeLLMOperator Integration Tests
# ----------------------------


@pytest.mark.integration
@pytest.mark.llm
def test_code_operator_create_initial_snippets(
    llm_client: Any, simple_problem: Any
) -> None:
    """Test that CodeLLMOperator can create initial code snippets with real LLM."""
    operator = CodeLLMOperator(llm_client, simple_problem)

    # Generate 2 initial solutions
    snippets = operator.create_initial_snippets(population_size=2)

    # Verify we got the right number of snippets
    assert len(snippets) == 2, f"Expected 2 snippets, got {len(snippets)}"

    # Verify each snippet contains the starter code structure
    for i, snippet in enumerate(snippets):
        assert "def add" in snippet, f"Snippet {i} missing 'def add'"
        assert len(snippet) > 20, f"Snippet {i} seems too short"
        logger.info(f"Generated snippet {i}:\n{snippet}\n")

    logger.success("Successfully generated initial code snippets with real LLM")


@pytest.mark.integration
@pytest.mark.llm
def test_code_operator_crossover(llm_client: Any, simple_problem: Any) -> None:
    """Test that CodeLLMOperator can perform crossover with real LLM."""
    operator = CodeLLMOperator(llm_client, simple_problem)

    parent1 = """def add(a, b):
    # Simple approach
    return a + b
"""

    parent2 = """def add(a, b):
    # Verbose approach
    result = a + b
    return result
"""

    # Perform crossover
    child = operator.crossover(parent1, parent2)

    # Verify child contains starter code
    assert "def add" in child, "Child missing 'def add'"
    assert len(child) > 20, "Child seems too short"
    logger.info(f"Crossover result:\n{child}\n")

    logger.success("Successfully performed crossover with real LLM")


@pytest.mark.integration
@pytest.mark.llm
def test_code_operator_mutate(llm_client: Any, coding_problem: Any) -> None:
    """Test that CodeLLMOperator can mutate code with real LLM."""
    operator = CodeLLMOperator(llm_client, coding_problem)

    original = """def find_max(nums):
    max_val = nums[0]
    for num in nums:
        if num > max_val:
            max_val = num
    return max_val
"""

    # Perform mutation
    mutated = operator.mutate(original)

    # Verify mutated code contains starter code
    assert "def find_max" in mutated, "Mutated code missing 'def find_max'"
    assert mutated != original, "Mutated code should be different from original"
    logger.info(f"Mutation result:\n{mutated}\n")

    logger.success("Successfully performed mutation with real LLM")


@pytest.mark.integration
@pytest.mark.llm
def test_code_operator_edit(llm_client: Any, simple_problem: Any) -> None:
    """Test that CodeLLMOperator can edit code based on feedback with real LLM."""
    operator = CodeLLMOperator(llm_client, simple_problem)

    buggy_code = """def add(a, b):
    return a - b  # Bug: should use + not -
"""

    feedback = "The function is subtracting instead of adding. Fix the operator."

    # Perform edit
    fixed = operator.edit(buggy_code, feedback)

    # Verify fixed code contains starter code
    assert "def add" in fixed, "Fixed code missing 'def add'"
    # We can't guarantee the LLM will fix it, but it should return something
    assert len(fixed) > 20, "Fixed code seems too short"
    logger.info(f"Edit result:\n{fixed}\n")

    logger.success("Successfully performed edit with real LLM")


# ----------------------------
# TestLLMOperator Integration Tests
# ----------------------------


@pytest.mark.integration
@pytest.mark.llm
def test_test_operator_create_initial_snippets(
    llm_client: Any, simple_problem: Any
) -> None:
    """Test that TestLLMOperator can create initial test snippets with real LLM."""
    operator = TestLLMOperator(llm_client, simple_problem)

    # Generate 3 initial test cases
    test_methods, full_test_code = operator.create_initial_snippets(population_size=3)

    # Verify we got the right number of test methods
    assert len(test_methods) == 3, f"Expected 3 test methods, got {len(test_methods)}"

    # Verify each test method looks like a test
    for i, test_method in enumerate(test_methods):
        assert "def test_" in test_method, f"Test {i} missing 'def test_'"
        assert "assert" in test_method or "assertEqual" in test_method, (
            f"Test {i} missing assertion"
        )
        logger.info(f"Generated test {i}:\n{test_method}\n")

    # Verify full test code contains class definition
    assert "class" in full_test_code, "Full test code missing class definition"
    logger.info(f"Full test code:\n{full_test_code}\n")

    logger.success("Successfully generated initial test snippets with real LLM")


@pytest.mark.integration
@pytest.mark.llm
def test_test_operator_crossover(llm_client: Any, simple_problem: Any) -> None:
    """Test that TestLLMOperator can perform crossover with real LLM."""
    operator = TestLLMOperator(llm_client, simple_problem)

    parent1 = """def test_positive_numbers(self):
    self.assertEqual(add(2, 3), 5)
"""

    parent2 = """def test_negative_numbers(self):
    self.assertEqual(add(-1, -1), -2)
"""

    # Perform crossover
    child = operator.crossover(parent1, parent2)

    # Verify child looks like a test
    assert "def test_" in child, "Child missing 'def test_'"
    logger.info(f"Test crossover result:\n{child}\n")

    logger.success("Successfully performed test crossover with real LLM")


@pytest.mark.integration
@pytest.mark.llm
def test_test_operator_mutate(llm_client: Any, coding_problem: Any) -> None:
    """Test that TestLLMOperator can mutate tests with real LLM."""
    operator = TestLLMOperator(llm_client, coding_problem)

    original_test = """def test_basic_case(self):
    self.assertEqual(find_max([1, 2, 3]), 3)
"""

    # Perform mutation
    mutated = operator.mutate(original_test)

    # Verify mutated test looks like a test
    assert "def test_" in mutated, "Mutated test missing 'def test_'"
    logger.info(f"Test mutation result:\n{mutated}\n")

    logger.success("Successfully performed test mutation with real LLM")


@pytest.mark.integration
@pytest.mark.llm
def test_test_operator_edit(llm_client: Any, simple_problem: Any) -> None:
    """Test that TestLLMOperator can edit tests based on feedback with real LLM."""
    operator = TestLLMOperator(llm_client, simple_problem)

    original_test = """def test_addition(self):
    self.assertEqual(add(2, 2), 4)
"""

    feedback = "Add a test case for adding zero to check edge cases"

    # Perform edit
    edited = operator.edit(original_test, feedback)

    # Verify edited test looks like a test
    assert "def test_" in edited, "Edited test missing 'def test_'"
    logger.info(f"Test edit result:\n{edited}\n")

    logger.success("Successfully performed test edit with real LLM")


# ----------------------------
# Token Tracking Tests
# ----------------------------


@pytest.mark.integration
@pytest.mark.llm
def test_token_tracking(llm_client: Any, simple_problem: Any) -> None:
    """Test that token tracking works correctly with LLM operators."""
    # Reset token count
    initial_tokens = llm_client.total_output_tokens
    logger.info(f"Initial tokens: {initial_tokens}")

    operator = CodeLLMOperator(llm_client, simple_problem)

    # Generate some code
    _ = operator.create_initial_snippets(population_size=1)

    # Check that tokens were tracked
    final_tokens = llm_client.total_output_tokens
    logger.info(f"Final tokens: {final_tokens}")

    assert final_tokens > initial_tokens, "Token count should have increased"
    logger.success(
        f"Token tracking working: {final_tokens - initial_tokens} tokens used"
    )


# ----------------------------
# Retry Logic Tests
# ----------------------------


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.slow
def test_retry_on_wrong_count(llm_client: Any, simple_problem: Any) -> None:
    """Test that retry logic works when LLM generates wrong number of snippets.

    Note: This test may be flaky as it depends on LLM behavior.
    """
    operator = CodeLLMOperator(llm_client, simple_problem)

    # Try to generate 5 snippets - LLM might not always get this right first try
    # If it succeeds first try, this test still passes
    snippets = operator.create_initial_snippets(population_size=5)

    assert len(snippets) == 5, f"Expected 5 snippets, got {len(snippets)}"
    logger.success("Retry logic test passed (may have succeeded first try)")
