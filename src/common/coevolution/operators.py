"""
LLM-based genetic operators for coevolutionary algorithms.

This module provides LLM-powered genetic operators for evolutionary algorithms
that evolve both code solutions and test cases. It defines an abstract base class
and two concrete implementations with built-in validation and retry mechanisms.

Classes:
    BaseLLMOperator: Abstract base class defining the genetic operator interface
    CodeOperator: Concrete operator for evolving code solutions
    TestOperator: Concrete operator for evolving test cases

Custom Exceptions:
    LLMGenerationError: Raised when LLM fails to generate valid output
    CodeValidationError: Raised when generated code fails validation checks

Genetic Operations:
    - create_initial_population: Generate initial population of individuals
    - mutate: Modify an individual to explore variations
    - crossover: Combine two parents to create offspring
    - edit: Fix an individual based on feedback/errors

Robustness Features:
    - Automatic retry with exponential backoff for transient LLM failures
    - Code validation (syntax checking, content validation)
    - Configurable retry parameters (max_retries, backoff settings)
    - Graceful handling of network errors, timeouts, and rate limits
    - Partial results for batch operations when some generations fail

The operators use an LLM (Large Language Model) to perform intelligent
transformations and maintain a problem context (CodeGenerationProblem)
that guides the evolutionary process. All generated code is validated
for syntax correctness before being returned.

All LLM interactions are logged using loguru for debugging and analysis:
    - logger.info: High-level operation tracking
    - logger.debug: Medium-level details (LLM calls, feedback)
    - logger.warning: Retry attempts and validation failures
    - logger.trace: Full verbose output (prompts, responses, code)

Example:
    >>> from common.llm_client import LLMClient
    >>> from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    >>>
    >>> llm = LLMClient(model="gpt-4")
    >>> # Configure retry behavior
    >>> code_op = CodeOperator(llm, max_retries=5, retry_backoff_max=30.0)
    >>> code_op.set_problem(problem)
    >>>
    >>> # Generate initial population (with automatic validation)
    >>> solutions = code_op.create_initial_population(population_size=5)
    >>>
    >>> # Apply genetic operations (with automatic retry on failures)
    >>> mutated = code_op.mutate(solutions[0])
    >>> offspring = code_op.crossover(solutions[0], solutions[1])
    >>> fixed = code_op.edit(solutions[0], "NameError: undefined variable")
"""

import ast
from abc import ABC, abstractmethod  # Import ABC and abstractmethod
from typing import List, Optional

from lcb_runner.benchmarks.code_generation import CodeGenerationProblem  # type: ignore
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from common.code_preprocessing.parsers import (
    extract_all_code_blocks_from_response,
    extract_code_block_from_response,
)
from common.llm_client import LLMClient

# === Custom Exceptions ===


class LLMGenerationError(Exception):
    """Raised when LLM fails to generate valid output after retries."""

    pass


class CodeValidationError(Exception):
    """Raised when generated code fails basic validation checks."""

    pass


class BaseLLMOperator(ABC):
    """
    Abstract base class for LLM-based genetic operators in evolutionary algorithms.

    This class defines the interface for genetic operators that use Large Language
    Models (LLMs) to perform intelligent transformations on code and test individuals.
    It maintains a problem context that can be updated and reused across multiple
    operations.

    Robustness Features:
        - Automatic retry with exponential backoff on LLM failures
        - Syntax validation using Python's ast module
        - Content validation (minimum code length, non-empty)
        - Configurable retry parameters
        - Graceful error handling with custom exceptions

    Abstract Methods (must be implemented by subclasses):
        create_initial_population(population_size: int) -> List[str] | str
            Generate an initial population of individuals

        mutate(individual: str) -> str
            Apply mutation to create a variation of an individual

        crossover(parent1: str, parent2: str) -> str
            Combine two parents to create offspring

        edit(individual: str, feedback: str) -> str
            Fix an individual based on error feedback

    Helper Methods:
        _generate_and_extract(prompt: str) -> str
            Send prompt to LLM and extract single code block with validation and retry

        _generate_and_extract_many(prompt: str, n: int) -> List[str]
            Send prompt to LLM and extract multiple code blocks with validation and retry

    Validation Methods:
        _validate_python_syntax(code: str) -> tuple[bool, str]
            Check if code is syntactically valid Python

        _validate_has_content(code: str) -> tuple[bool, str]
            Check if code has meaningful content

        _validate_generated_code(code: str) -> None
            Perform all validation checks (raises CodeValidationError if invalid)

    Problem Management:
        set_problem(problem: CodeGenerationProblem) -> None
            Set or update the current problem context

        get_problem() -> Optional[CodeGenerationProblem]
            Get the current problem context

    Attributes:
        llm (LLMClient): The LLM client for generating responses
        problem (Optional[CodeGenerationProblem]): Current problem context
        max_retries (int): Maximum number of retry attempts (default: 3)
        retry_backoff_base (float): Exponential backoff multiplier (default: 2.0)
        retry_backoff_max (float): Maximum wait between retries in seconds (default: 10.0)

    Raises:
        LLMGenerationError: When LLM fails to generate valid output after all retries
        CodeValidationError: When generated code fails validation checks
        RuntimeError: When abstract methods are called or problem is not set

    Usage:
        # Subclasses implement the abstract methods
        class MyOperator(BaseLLMOperator):
            def create_initial_population(self, population_size):
                # Implementation here
                pass
            # ... implement other abstract methods

        # Use the operator with custom retry settings
        operator = MyOperator(llm_client, max_retries=5, retry_backoff_max=30.0)
        operator.set_problem(problem)
        result = operator.mutate(code)  # Automatically retries on failures
    """

    def __init__(
        self,
        llm_client: LLMClient,
        problem: Optional[CodeGenerationProblem] = None,
        max_retries: int = 3,
        retry_backoff_base: float = 2.0,
        retry_backoff_max: float = 10.0,
    ) -> None:
        """
        Initialize the LLMOperator with an LLM client and optional problem.

        Args:
            llm_client: An instance of LLMClient for interacting with the LLM
            problem: Optional initial problem context. Can be set later via set_problem()
            max_retries: Maximum number of retry attempts for LLM operations (default: 3).
                         Higher values increase reliability but may slow down operations.
            retry_backoff_base: Base multiplier for exponential backoff in seconds (default: 2.0).
                                Wait time = backoff_base ^ (attempt_number - 1)
            retry_backoff_max: Maximum wait time between retries in seconds (default: 10.0).
                               Caps the exponential backoff to prevent excessive delays.

        Example:
            >>> # Standard configuration
            >>> operator = CodeOperator(llm_client)
            >>>
            >>> # More aggressive retry for unreliable networks
            >>> operator = CodeOperator(llm_client, max_retries=5, retry_backoff_max=30.0)
            >>>
            >>> # Faster retry for development/testing
            >>> operator = CodeOperator(llm_client, max_retries=2, retry_backoff_base=1.5)
        """
        self.llm = llm_client
        self.problem: Optional[CodeGenerationProblem] = None

        # Retry configuration
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.retry_backoff_max = retry_backoff_max

        # If a problem was passed, call the setter to process and assign it.
        if problem:
            self.set_problem(problem)

    def set_problem(self, problem: CodeGenerationProblem) -> None:
        """
        Set or update the current problem context.

        This also ensures that a default starter_code is present
        if the problem provides an empty one.

        Args:
            problem: The new problem context to use for operations
        """

        # Use 'not' for a more Pythonic check than len() == 0
        if not problem.starter_code or len(problem.starter_code.strip()) == 0:
            # Note: This modifies the 'starter_code' attribute on the
            # problem object itself.
            problem.starter_code = """
class Solution:
    def sol(self, input_str):
"""
        self.problem = problem

    def get_problem(self) -> Optional[CodeGenerationProblem]:
        """
        Get the current problem context.

        Returns:
            The current problem, or None if not set
        """
        return self.problem

    def _ensure_problem(self) -> CodeGenerationProblem:
        """
        Internal helper to ensure a problem is set.

        Returns:
            The current problem

        Raises:
            ValueError: If no problem has been set
        """
        if self.problem is None:
            raise ValueError(
                "No problem context set. Call set_problem() before performing operations."
            )
        return self.problem

    def _validate_python_syntax(self, code: str) -> tuple[bool, str]:
        """
        Validate that code is syntactically correct Python.

        This is a basic quality check - does not validate semantics,
        imports, or problem-specific requirements.

        Args:
            code: Python code string to validate

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if syntax is valid, False otherwise
            - error_message: Empty string if valid, error details if invalid
        """
        if not code or not code.strip():
            return False, "Empty code string"

        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            logger.warning(f"Syntax validation failed: {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error during syntax validation: {str(e)}"
            logger.warning(error_msg)
            return False, error_msg

    def _validate_has_content(self, code: str, min_lines: int = 2) -> tuple[bool, str]:
        """
        Validate that generated code has meaningful content.

        Checks:
        - Not empty
        - Has minimum number of non-comment lines
        - Not just whitespace or comments

        Args:
            code: Code string to validate
            min_lines: Minimum number of non-empty, non-comment lines

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not code or not code.strip():
            return False, "Empty or whitespace-only code"

        # Count non-empty, non-comment lines
        lines = [
            line.strip()
            for line in code.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]

        if len(lines) < min_lines:
            return False, f"Too few lines of code ({len(lines)} < {min_lines})"

        return True, ""

    def _validate_generated_code(self, code: str) -> None:
        """
        Perform basic validation on generated code.

        Validates:
        1. Has meaningful content
        2. Syntactically valid Python

        Raises:
            CodeValidationError: If validation fails
        """
        # Check content
        is_valid, error = self._validate_has_content(code)
        if not is_valid:
            raise CodeValidationError(f"Content validation failed: {error}")

        # Check syntax
        is_valid, error = self._validate_python_syntax(code)
        if not is_valid:
            raise CodeValidationError(f"Syntax validation failed: {error}")

        logger.trace("Code validation passed")

    def _generate_and_extract(self, prompt: str) -> str:
        """
        Generate LLM response and extract clean code block with retry logic.

        This helper method wraps the LLM generate call, automatically extracts
        the first Python code block, and validates the result. Includes retry
        logic for transient failures using tenacity.

        Retries on:
        - LLM API failures (network, rate limits, etc.)
        - Empty or invalid responses
        - Syntax errors in generated code

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            Extracted and validated Python code string

        Raises:
            LLMGenerationError: If LLM fails after all retries
            CodeValidationError: If generated code is invalid after all retries
        """

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=self.retry_backoff_base, max=self.retry_backoff_max
            ),
            retry=retry_if_exception_type(
                (LLMGenerationError, CodeValidationError, ConnectionError, TimeoutError)
            ),
        )
        def _generate_with_retry() -> str:
            logger.debug("Sending prompt to LLM")
            logger.trace(f"Prompt content:\n{prompt}")

            try:
                raw_response = self.llm.generate(prompt)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                raise LLMGenerationError(f"LLM API call failed: {e}") from e

            logger.debug("Received response from LLM")
            logger.trace(f"Raw LLM response:\n{raw_response}")

            # Extract code
            extracted_code = extract_code_block_from_response(raw_response)

            if not extracted_code or not extracted_code.strip():
                logger.warning("LLM returned empty code block")
                raise LLMGenerationError("LLM returned empty code block")

            logger.trace(f"Extracted code:\n{extracted_code}")

            # Validate extracted code
            self._validate_generated_code(extracted_code)

            return extracted_code

        return _generate_with_retry()

    def _generate_and_extract_many(self, prompt: str, n: int) -> List[str]:
        """
        Generate multiple solutions in a single LLM call with retry logic.

        This helper method wraps the LLM generate call for batch generation,
        automatically extracting all Python code blocks and validating each one.
        Includes retry logic for transient failures using tenacity.

        Retries on:
        - LLM API failures (network, rate limits, etc.)
        - Empty or invalid responses
        - Syntax errors in generated code

        If some code blocks are valid and others invalid, returns the valid ones.
        Only fails if no valid blocks are found after all retries.

        Args:
            prompt: The prompt to send to the LLM
            n: The number of code blocks to extract (used for logging)

        Returns:
            A list of extracted and validated Python code strings (up to n items)

        Raises:
            LLMGenerationError: If LLM fails after all retries
            CodeValidationError: If no valid code blocks found after all retries
        """

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=self.retry_backoff_base, max=self.retry_backoff_max
            ),
            retry=retry_if_exception_type(
                (LLMGenerationError, CodeValidationError, ConnectionError, TimeoutError)
            ),
        )
        def _generate_with_retry() -> List[str]:
            logger.debug(f"Sending prompt to LLM (requesting {n} items)")
            logger.trace(f"Prompt content:\n{prompt}")

            try:
                raw_response = self.llm.generate(prompt)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                raise LLMGenerationError(f"LLM API call failed: {e}") from e

            logger.debug("Received response from LLM")
            logger.trace(f"Raw LLM response:\n{raw_response}")

            # Extract all code blocks
            extracted_codes = extract_all_code_blocks_from_response(raw_response)

            if not extracted_codes:
                logger.warning("LLM returned no code blocks")
                raise LLMGenerationError("LLM returned no code blocks")

            logger.debug(f"Extracted {len(extracted_codes)} code blocks from response")

            # Validate each code block - keep valid ones, log invalid ones
            valid_blocks = []
            for idx, code_block in enumerate(extracted_codes[:n]):
                try:
                    self._validate_generated_code(code_block)
                    valid_blocks.append(code_block)
                    logger.trace(f"Code block {idx + 1} is valid")
                except CodeValidationError as e:
                    logger.warning(
                        f"Code block {idx + 1} validation failed: {e}. Skipping."
                    )
                    continue

            if not valid_blocks:
                logger.error("No valid code blocks found after validation")
                raise CodeValidationError(
                    f"All {len(extracted_codes[:n])} code blocks failed validation"
                )

            if len(valid_blocks) < n:
                logger.warning(
                    f"Only {len(valid_blocks)} valid blocks found (expected {n})"
                )

            logger.trace(f"Returning {len(valid_blocks)} valid code blocks")
            return valid_blocks

        return _generate_with_retry()

    # --- Define the Abstract Interface for Operators ---
    @abstractmethod
    def create_initial_population(self, population_size: int) -> List[str] | str:
        """
        Generate an initial population of individuals.

        Args:
            population_size: The number of individuals to generate.

        Returns:
            A list of strings, each a unique individual.
        """
        pass

    @abstractmethod
    def mutate(self, individual: str) -> str:
        """
        Apply a mutation to a single individual (code or test).

        Args:
            individual: The code or test string to mutate.

        Returns:
            A new, mutated individual string.
        """
        pass

    @abstractmethod
    def crossover(self, parent1: str, parent2: str) -> str:
        """
        Apply crossover to two individuals (code or test).

        Args:
            parent1: The first parent string.
            parent2: The second parent string.

        Returns:
            A new, child string resulting from crossover.
        """
        pass

    @abstractmethod
    def edit(self, individual: str, feedback: str) -> str:
        """
        Apply an edit to an individual based on feedback.

        Args:
            individual: The code or test string to edit.
            feedback: The feedback (e.g., error message) to guide the edit.

        Returns:
            A new, edited individual string.
        """
        pass


# ============================================================================
# Concrete Operator Implementations
# ============================================================================


class CodeOperator(BaseLLMOperator):
    """
    Concrete operator specialized for code solution evolution.

    This operator implements all genetic operations specifically tailored for
    evolving code solutions to programming problems. It generates, mutates,
    combines, and fixes Python code using LLM-powered transformations with
    automatic validation and retry mechanisms.

    Return Types:
        create_initial_population: Returns List[str] - multiple validated code solutions
        mutate: Returns str - single validated mutated code solution
        crossover: Returns str - single validated offspring code solution
        edit: Returns str - single validated fixed code solution

    Operations:
        - Initial Population: Generates multiple independent code solutions, each in
          a separate code block. Solutions may use different algorithms or approaches.
          All solutions are validated for syntax correctness before being returned.

        - Mutation: Modifies a solution to explore different implementation approaches
          while maintaining functionality (e.g., different algorithm, data structure).
          Automatically retries if LLM generates invalid syntax.

        - Crossover: Intelligently combines the best aspects of two parent solutions
          (e.g., better algorithm from parent1, clearer logic from parent2).
          Validates the offspring before returning.

        - Edit: Fixes a solution based on error feedback (e.g., syntax errors, test
          failures) while maintaining the overall approach. Retries if fix introduces
          new syntax errors.

    Robustness:
        All operations automatically retry on:
        - LLM API failures (network errors, timeouts, rate limits)
        - Empty or invalid responses
        - Syntax errors in generated code

        Configuration is inherited from BaseLLMOperator (max_retries, backoff settings).

    Example:
        >>> code_op = CodeOperator(llm_client, max_retries=5)
        >>> code_op.set_problem(problem)
        >>>
        >>> # Generate 5 diverse solutions (automatically validated)
        >>> solutions = code_op.create_initial_population(5)
        >>> # Returns: ["def solution1()...", "def solution2()...", ...]
        >>> # All solutions guaranteed to be syntactically valid
        >>>
        >>> # Mutate a solution (with automatic retry on failures)
        >>> variant = code_op.mutate(solutions[0])
        >>>
        >>> # Combine two solutions (validated before return)
        >>> child = code_op.crossover(solutions[0], solutions[1])
        >>>
        >>> # Fix an error (retries if fix introduces new errors)
        >>> fixed = code_op.edit(solutions[0], "NameError: x is not defined")
    """

    def create_initial_population(self, population_size: int) -> List[str]:
        """
        Generate an initial population of code solutions.

        Guarantees to return exactly `population_size` solutions.

        Args:
            population_size: The number of solutions to generate

        Returns:
            A list of exactly `population_size` code solution strings

        Raises:
            LLMGenerationError: If unable to generate the requested number of solutions
        """
        logger.info(
            f"CodeOperator: Creating initial population of size {population_size}"
        )
        problem: CodeGenerationProblem = self._ensure_problem()

        prompt: str = f"Write {population_size} distinct solutions to solve the following problem: {problem.question_content}\n```python\n{problem.starter_code}```. Each solution should be in a separate code block."

        solutions = self._generate_and_extract_many(prompt, population_size)

        # Verify we got exactly the requested number
        if len(solutions) != population_size:
            error_msg = (
                f"Failed to generate requested population size. "
                f"Expected {population_size}, got {len(solutions)} valid solutions."
            )
            logger.error(error_msg)
            raise LLMGenerationError(error_msg)

        logger.info(
            f"CodeOperator: Successfully generated {len(solutions)} solutions for initial population"
        )
        return solutions

    def mutate(self, individual: str) -> str:
        """
        Mutate a code solution.

        Generates a modified version that maintains functionality but
        explores a different implementation approach.

        Args:
            individual: The code solution to mutate

        Returns:
            A mutated code solution
        """
        logger.info("CodeOperator: Performing mutation on code solution")
        problem = self._ensure_problem()

        prompt = f"""Given this programming problem:
{problem.question_content}

And this code solution:
```python
{individual}
```

Generate a modified version of the code that:
1. Maintains the same functionality
2. Explores a different algorithmic approach or implementation style
3. Could potentially be more efficient or clearer

Return the modified code in a python code block"""

        logger.trace(f"Original individual to mutate:\n{individual}")
        result = self._generate_and_extract(prompt)
        logger.info("CodeOperator: Mutation completed")
        return result

    def crossover(self, parent1: str, parent2: str) -> str:
        """
        Combine two code solutions to create a new offspring.

        Args:
            parent1: First parent code solution
            parent2: Second parent code solution

        Returns:
            A new code solution combining ideas from both parents
        """
        logger.info("CodeOperator: Performing crossover on two code solutions")
        problem = self._ensure_problem()

        prompt = f"""Given this programming problem:
{problem.question_content}

And these two code solutions:

Solution 1:
```python
{parent1}
```

Solution 2:
```python
{parent2}
```

Create a new solution that intelligently combines the best aspects of both solutions.
Consider combining:
- Better algorithms from either solution
- More efficient data structures
- Clearer variable names or logic flow

Return the new combined code in a python code block."""

        logger.trace(f"Parent 1:\n{parent1}")
        logger.trace(f"Parent 2:\n{parent2}")
        result = self._generate_and_extract(prompt)
        logger.info("CodeOperator: Crossover completed")
        return result

    def edit(self, individual: str, feedback: str) -> str:
        """
        Edit a code solution based on error feedback.

        Args:
            individual: The code solution to edit
            feedback: Error message or test failure information

        Returns:
            An edited code solution that addresses the feedback
        """
        logger.info("CodeOperator: Editing code solution based on feedback")
        logger.trace(f"Feedback: {feedback}")
        problem = self._ensure_problem()

        prompt = f"""Given this programming problem:
{problem.question_content}

This code solution:
```python
{individual}
```

And this error/feedback:
{feedback}

Fix the code to address the error while maintaining the overall approach.
Return the fixed code in a python code block."""

        logger.trace(f"Original individual to edit:\n{individual}")
        result = self._generate_and_extract(prompt)
        logger.info("CodeOperator: Edit completed")
        return result


class TestOperator(BaseLLMOperator):
    """
    Concrete operator specialized for test case evolution.

    This operator implements all genetic operations specifically tailored for
    evolving test cases with automatic validation and retry. Unlike CodeOperator
    which returns multiple separate solutions, TestOperator works with unittest
    classes containing multiple test methods.

    Return Types:
        create_initial_population: Returns str - single validated unittest class with multiple test methods
        mutate: Returns str - single validated mutated test case
        crossover: Returns str - single validated combined test case
        edit: Returns str - single validated improved test case

    Design Note:
        TestOperator returns a single unittest class (str) for initial population,
        not a list. This is intentional because test methods belong together in a
        unittest class for proper execution. Individual test methods should be
        extracted separately in the main algorithm using test analyzers.

    Operations:
        - Initial Population: Generates a unittest class containing multiple test
          methods. Methods test different scenarios, edge cases, and boundary conditions.
          The entire class is validated for syntax before return.

        - Mutation: Modifies a test to check different edge cases or boundary
          conditions while maintaining unittest structure. Automatically retries if
          mutation breaks test structure.

        - Crossover: Combines assertions and test scenarios from two test cases to
          create a more comprehensive test. Validates the combined test structure.

        - Edit: Improves a test based on feedback (e.g., test too broad, missed
          edge case, assertion errors). Retries if improvement introduces syntax errors.

    Robustness:
        All operations automatically retry on:
        - LLM API failures (network errors, timeouts, rate limits)
        - Empty or invalid responses
        - Syntax errors in generated test code

        Configuration is inherited from BaseLLMOperator (max_retries, backoff settings).

    Example:
        >>> test_op = TestOperator(llm_client, max_retries=5)
        >>> test_op.set_problem(problem)
        >>>
        >>> # Generate unittest class with 5 test methods (validated)
        >>> test_class = test_op.create_initial_population(5)
        >>> # Returns: "import unittest\\nclass TestSolution(unittest.TestCase):\\n    def test_1()...\\n    def test_2()..."
        >>> # Guaranteed to be syntactically valid unittest code
        >>>
        >>> # Extract individual tests later in your algorithm
        >>> individual_tests = extract_test_methods(test_class)
        >>>
        >>> # Mutate a test (with automatic retry on failures)
        >>> variant = test_op.mutate(individual_tests[0])
        >>>
        >>> # Combine two tests (validated before return)
        >>> comprehensive_test = test_op.crossover(individual_tests[0], individual_tests[1])
        >>>
        >>> # Improve based on feedback (retries if needed)
        >>> improved = test_op.edit(individual_tests[0], "Test too broad, add edge cases")
    """

    def create_initial_population(self, population_size: int) -> str:
        """
        Generate an initial population of test cases as a single unittest class.

        Note: Unlike CodeOperator which returns a list of separate solutions,
        TestOperator returns a single unittest class containing multiple test methods.
        Individual test methods should be extracted separately in the main algorithm.

        Guarantees the unittest class contains at least `population_size` test methods.

        Args:
            population_size: The number of test methods to request in the unittest class

        Returns:
            A single string containing a unittest class with at least `population_size` test methods

        Raises:
            LLMGenerationError: If unable to generate the requested number of test methods
        """
        logger.info(
            f"TestOperator: Creating initial test population (requesting {population_size} test methods)"
        )
        problem: CodeGenerationProblem = self._ensure_problem()

        prompt: str = f"Write {population_size} distinct unit tests in a Python code block for the following problem:\n{problem.question_content}\nThe solution is imported in the format:\n```python\n{problem.starter_code}```\nWrite the tests using unittest framework. Do not use the examples given in the problem."

        result = self._generate_and_extract(prompt)

        # Count test methods and verify we got at least the requested number
        test_count = result.count("def test_")

        if test_count < population_size:
            error_msg = (
                f"Failed to generate requested number of test methods. "
                f"Expected at least {population_size}, got {test_count} test methods."
            )
            logger.error(error_msg)
            raise LLMGenerationError(error_msg)

        logger.info(
            f"TestOperator: Successfully generated unittest class with {test_count} test methods"
        )

        return result

    def mutate(self, individual: str) -> str:
        """
        Mutate a test case.

        Generates a modified test that checks different edge cases
        or scenarios while maintaining the same testing structure.

        Args:
            individual: The test case to mutate

        Returns:
            A mutated test case
        """
        logger.info("TestOperator: Performing mutation on test case")
        problem = self._ensure_problem()

        prompt = f"""Given this programming problem:
{problem.question_content}

And this test case:
```python
{individual}
```

Generate a modified test case that tests different edge cases or scenarios.
Ensure it remains a valid unittest test case and contains only a single assertion.

Return only the test method code in a python code block."""

        logger.trace(f"Original test to mutate:\n{individual}")
        result = self._generate_and_extract(prompt)
        logger.info("TestOperator: Mutation completed")
        return result

    def crossover(self, parent1: str, parent2: str) -> str:
        """
        Combine two test cases to create a new test.

        Args:
            parent1: First parent test case
            parent2: Second parent test case

        Returns:
            A new test case combining assertions from both parents
        """
        logger.info("TestOperator: Performing crossover on two test cases")
        problem = self._ensure_problem()

        prompt = f"""Given this programming problem:
{problem.question_content}

And these two test cases:

Test 1:
```python
{parent1}
```

Test 2:
```python
{parent2}
```

Summarize test 1 and test 2, then create a new test case. 

Return only the new test method code in a python code block."""

        logger.trace(f"Test parent 1:\n{parent1}")
        logger.trace(f"Test parent 2:\n{parent2}")
        result = self._generate_and_extract(prompt)
        logger.info("TestOperator: Crossover completed")
        return result

    def edit(self, individual: str, feedback: str) -> str:
        """
        Edit a test case based on feedback.

        Args:
            individual: The test case to edit
            feedback: Feedback about the test (e.g., too broad, missed edge case)

        Returns:
            An edited test case that addresses the feedback
        """
        logger.info("TestOperator: Editing test case based on feedback")
        logger.trace(f"Feedback: {feedback}")
        problem = self._ensure_problem()

        prompt = f"""Given this programming problem:
{problem.question_content}

This test case:
```python
{individual}
```

And this feedback:
{feedback}

Improve the test case to address the feedback.
Ensure it remains a valid unittest test case and contains only a single assertion.
Only return the improved test code in a python code block."""

        logger.trace(f"Original test to edit:\n{individual}")
        result = self._generate_and_extract(prompt)
        logger.info("TestOperator: Edit completed")
        return result
