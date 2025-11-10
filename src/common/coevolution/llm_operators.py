"""
Minimal LLM-based implementations of code and test operators.

This module provides bare-minimum implementations of ICodeOperator and ITestOperator
using the BaseLLMOperator infrastructure. These implementations:
- Use _generate() to call the LLM (client handles retry)
- Return raw responses (no extraction or validation)
- Provide simple prompts for each operation
- Delegate all processing to external utilities

Concrete implementations should:
1. Import code_preprocessing utilities for extraction and validation
2. Add domain-specific prompt engineering
3. Use LLM clients that handle retry logic internally
"""

from loguru import logger

from .base_llm_operator import BaseLLMOperator
from .core.interfaces import ICodeOperator, ITestOperator


class CodeLLMOperator(BaseLLMOperator, ICodeOperator):
    """
    Bare minimum implementation of ICodeOperator using LLM.

    This provides the simplest possible implementation that satisfies
    the ICodeOperator protocol. It generates raw LLM responses without
    any extraction or validation.

    Subclasses or external code should:
    - Extract code blocks from responses using code_preprocessing utilities
    - Validate syntax using code_preprocessing utilities
    - Implement proper prompt engineering for better code generation
    """

    def create_initial_snippets(self, population_size: int) -> list[str]:
        """
        Generate initial code snippets.

        Returns raw LLM responses. External code should extract and validate.

        Args:
            population_size: Number of code snippets to generate

        Returns:
            List of raw LLM responses
        """
        logger.info(f"Generating {population_size} initial code snippets")

        # Use self.starter_code to get starter code with default fallback
        prompt = f"""Generate {population_size} different solutions to this problem:

{self.problem.question_content}

Starting template:
```python
{self.starter_code}
```

Each solution should be complete and in a separate code block."""

        response = self._generate(prompt)
        logger.debug(f"Generated initial code response ({len(response)} chars)")

        # Return as single-item list - caller should split/extract
        return [response]

    def mutate(self, individual: str) -> str:
        """
        Mutate a code snippet.

        Returns raw LLM response. External code should extract and validate.

        Args:
            individual: Code snippet to mutate

        Returns:
            Raw LLM response
        """
        logger.debug("Mutating code snippet")

        prompt = f"""Mutate this Python code by making a small change:

{individual}

Return only the mutated code."""

        response = self._generate(prompt)
        logger.debug(f"Generated mutation response ({len(response)} chars)")

        return response

    def crossover(self, parent1: str, parent2: str) -> str:
        """
        Crossover two code snippets.

        Returns raw LLM response. External code should extract and validate.

        Args:
            parent1: First parent code snippet
            parent2: Second parent code snippet

        Returns:
            Raw LLM response
        """
        logger.debug("Performing code crossover")

        prompt = f"""Combine these two Python code snippets into one:

Parent 1:
{parent1}

Parent 2:
{parent2}

Return only the combined code."""

        response = self._generate(prompt)
        logger.debug(f"Generated crossover response ({len(response)} chars)")

        return response

    def edit(self, individual: str, feedback: str) -> str:
        """
        Edit a code snippet based on feedback.

        Returns raw LLM response. External code should extract and validate.

        Args:
            individual: Code snippet to edit
            feedback: Feedback to guide the edit

        Returns:
            Raw LLM response
        """
        logger.debug("Editing code snippet based on feedback")

        prompt = f"""Edit this Python code based on the feedback:

Code:
{individual}

Feedback:
{feedback}

Return only the edited code."""

        response = self._generate(prompt)
        logger.debug(f"Generated edit response ({len(response)} chars)")

        return response


class TestLLMOperator(BaseLLMOperator, ITestOperator):
    """
    Bare minimum implementation of ITestOperator using LLM.

    This provides the simplest possible implementation that satisfies
    the ITestOperator protocol. It generates raw LLM responses without
    any extraction or validation.

    Subclasses or external code should:
    - Extract test methods from responses using code_preprocessing utilities
    - Validate test syntax using code_preprocessing utilities
    - Implement proper prompt engineering for better test generation
    """

    def create_initial_snippets(self, population_size: int) -> tuple[list[str], str]:
        """
        Generate initial test snippets.

        Returns raw LLM response as both the class block and snippet list.
        External code should extract and validate individual test methods.

        Args:
            population_size: Number of test methods to generate

        Returns:
            Tuple of ([raw_response], raw_response)
        """
        logger.info(f"Generating {population_size} initial test snippets")

        # Use self.starter_code to show the solution structure to test against
        prompt = f"""Generate a Python unittest class with {population_size} test methods for this problem:

{self.problem.question_content}

The solution being tested has this structure:
```python
{self.starter_code}
```

Include the class definition and all test methods.
Each test method should test a different aspect."""

        response = self._generate(prompt)
        logger.debug(f"Generated initial test response ({len(response)} chars)")

        # Return raw response as both class block and single snippet
        # Caller should extract individual test methods
        return ([response], response)

    def mutate(self, individual: str) -> str:
        """
        Mutate a test snippet.

        Returns raw LLM response. External code should extract and validate.

        Args:
            individual: Test snippet to mutate

        Returns:
            Raw LLM response
        """
        logger.debug("Mutating test snippet")

        prompt = f"""Mutate this Python test method by making a small change:

{individual}

Return only the mutated test method."""

        response = self._generate(prompt)
        logger.debug(f"Generated test mutation response ({len(response)} chars)")

        return response

    def crossover(self, parent1: str, parent2: str) -> str:
        """
        Crossover two test snippets.

        Returns raw LLM response. External code should extract and validate.

        Args:
            parent1: First parent test snippet
            parent2: Second parent test snippet

        Returns:
            Raw LLM response
        """
        logger.debug("Performing test crossover")

        prompt = f"""Combine these two Python test methods into one:

Parent 1:
{parent1}

Parent 2:
{parent2}

Return only the combined test method."""

        response = self._generate(prompt)
        logger.debug(f"Generated test crossover response ({len(response)} chars)")

        return response

    def edit(self, individual: str, feedback: str) -> str:
        """
        Edit a test snippet based on feedback.

        Returns raw LLM response. External code should extract and validate.

        Args:
            individual: Test snippet to edit
            feedback: Feedback to guide the edit

        Returns:
            Raw LLM response
        """
        logger.debug("Editing test snippet based on feedback")

        prompt = f"""Edit this Python test method based on the feedback:

Test:
{individual}

Feedback:
{feedback}

Return only the edited test method."""

        response = self._generate(prompt)
        logger.debug(f"Generated test edit response ({len(response)} chars)")

        return response
