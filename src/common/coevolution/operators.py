"""
LLM-based genetic operators for coevolutionary algorithms.

This module provides LLM-powered genetic operators for evolutionary algorithms
that evolve both code solutions and test cases. It defines an abstract base class
and two concrete implementations:

Classes:
    BaseLLMOperator: Abstract base class defining the genetic operator interface
    CodeOperator: Concrete operator for evolving code solutions
    TestOperator: Concrete operator for evolving test cases

Genetic Operations:
    - create_initial_population: Generate initial population of individuals
    - mutate: Modify an individual to explore variations
    - crossover: Combine two parents to create offspring
    - edit: Fix an individual based on feedback/errors

The operators use an LLM (Large Language Model) to perform intelligent
transformations and maintain a problem context (CodeGenerationProblem)
that guides the evolutionary process.

All LLM interactions are logged using loguru for debugging and analysis:
    - logger.info: High-level operation tracking
    - logger.debug: Medium-level details (LLM calls, feedback)
    - logger.trace: Full verbose output (prompts, responses, code)

Example:
    >>> from common.llm_client import LLMClient
    >>> from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    >>>
    >>> llm = LLMClient(model="gpt-4")
    >>> code_op = CodeOperator(llm)
    >>> code_op.set_problem(problem)
    >>>
    >>> # Generate initial population
    >>> solutions = code_op.create_initial_population(population_size=5)
    >>>
    >>> # Apply genetic operations
    >>> mutated = code_op.mutate(solutions[0])
    >>> offspring = code_op.crossover(solutions[0], solutions[1])
    >>> fixed = code_op.edit(solutions[0], "NameError: undefined variable")
"""

from abc import ABC, abstractmethod  # Import ABC and abstractmethod
from typing import List, Optional

from lcb_runner.benchmarks.code_generation import CodeGenerationProblem  # type: ignore
from loguru import logger

from common.code_preprocessing.parsers import (
    extract_all_code_blocks_from_response,
    extract_code_block_from_response,
)
from common.llm_client import LLMClient


class BaseLLMOperator(ABC):
    """
    Abstract base class for LLM-based genetic operators in evolutionary algorithms.

    This class defines the interface for genetic operators that use Large Language
    Models (LLMs) to perform intelligent transformations on code and test individuals.
    It maintains a problem context that can be updated and reused across multiple
    operations.

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
            Send prompt to LLM and extract single code block

        _generate_and_extract_many(prompt: str, n: int) -> List[str]
            Send prompt to LLM and extract multiple code blocks

    Problem Management:
        set_problem(problem: CodeGenerationProblem) -> None
            Set or update the current problem context

        get_problem() -> Optional[CodeGenerationProblem]
            Get the current problem context

    Attributes:
        llm (LLMClient): The LLM client for generating responses
        problem (Optional[CodeGenerationProblem]): Current problem context

    Usage:
        # Subclasses implement the abstract methods
        class MyOperator(BaseLLMOperator):
            def create_initial_population(self, population_size):
                # Implementation here
                pass
            # ... implement other abstract methods

        # Use the operator
        operator = MyOperator(llm_client)
        operator.set_problem(problem)
        result = operator.mutate(code)
    """

    def __init__(
        self, llm_client: LLMClient, problem: Optional[CodeGenerationProblem] = None
    ) -> None:
        """
        Initialize the LLMOperator with an LLM client and optional problem.

        Args:
            llm_client: An instance of LLMClient for interacting with the LLM
            problem: Optional initial problem context. Can be set later via set_problem()
        """
        self.llm = llm_client
        self.problem: Optional[CodeGenerationProblem] = None

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

    def _generate_and_extract(self, prompt: str) -> str:
        """
        Generate LLM response and extract clean code block.

        This helper method wraps the LLM generate call and automatically
        extracts the first Python code block from the response, removing
        any markdown formatting or explanatory text.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            Extracted Python code string (or raw response if no code block found)
        """
        logger.debug("Sending prompt to LLM")
        logger.trace(f"Prompt content:\n{prompt}")

        raw_response = self.llm.generate(prompt)

        logger.debug("Received response from LLM")
        logger.trace(f"Raw LLM response:\n{raw_response}")

        extracted_code = extract_code_block_from_response(raw_response)
        logger.trace(f"Extracted code:\n{extracted_code}")

        return extracted_code

    def _generate_and_extract_many(self, prompt: str, n: int) -> List[str]:
        """
        Generate LLM response and extract multiple code blocks.

        This helper method wraps the LLM generate call and automatically
        extracts multiple Python code blocks from the response, removing
        any markdown formatting or explanatory text.

        Args:
            prompt: The prompt to send to the LLM
            n: The number of code blocks to extract (used for logging)

        Returns:
            A list of extracted Python code strings (up to n items)
        """
        logger.debug(f"Sending prompt to LLM (requesting {n} items)")
        logger.trace(f"Prompt content:\n{prompt}")

        raw_response = self.llm.generate(prompt)

        logger.debug("Received response from LLM")
        logger.trace(f"Raw LLM response:\n{raw_response}")

        extracted_codes = extract_all_code_blocks_from_response(raw_response)
        logger.debug(f"Extracted {len(extracted_codes)} code blocks from response")

        # Return up to the requested number of code blocks
        result = extracted_codes[:n] if extracted_codes else []
        logger.trace(f"Returning {len(result)} code blocks")

        return result

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
    combines, and fixes Python code using LLM-powered transformations.

    Return Types:
        create_initial_population: Returns List[str] - multiple separate code solutions
        mutate: Returns str - single mutated code solution
        crossover: Returns str - single offspring code solution
        edit: Returns str - single fixed code solution

    Operations:
        - Initial Population: Generates multiple independent code solutions, each in
          a separate code block. Solutions may use different algorithms or approaches.

        - Mutation: Modifies a solution to explore different implementation approaches
          while maintaining functionality (e.g., different algorithm, data structure).

        - Crossover: Intelligently combines the best aspects of two parent solutions
          (e.g., better algorithm from parent1, clearer logic from parent2).

        - Edit: Fixes a solution based on error feedback (e.g., syntax errors, test
          failures) while maintaining the overall approach.

    Example:
        >>> code_op = CodeOperator(llm_client)
        >>> code_op.set_problem(problem)
        >>>
        >>> # Generate 5 diverse solutions
        >>> solutions = code_op.create_initial_population(5)
        >>> # Returns: ["def solution1()...", "def solution2()...", ...]
        >>>
        >>> # Mutate a solution
        >>> variant = code_op.mutate(solutions[0])
        >>>
        >>> # Combine two solutions
        >>> child = code_op.crossover(solutions[0], solutions[1])
        >>>
        >>> # Fix an error
        >>> fixed = code_op.edit(solutions[0], "NameError: x is not defined")
    """

    def create_initial_population(self, population_size: int) -> List[str]:
        """
        Generate an initial population of code solutions.

        Args:
            population_size: The number of solutions to generate

        Returns:
            A list of code solution strings
        """
        logger.info(
            f"CodeOperator: Creating initial population of size {population_size}"
        )
        problem: CodeGenerationProblem = self._ensure_problem()

        prompt: str = f"Write {population_size} distinct solutions to solve the following problem: {problem.question_content}\n```python\n{problem.starter_code}```. Each solution should be in a separate code block."

        solutions = self._generate_and_extract_many(prompt, population_size)

        logger.info(
            f"CodeOperator: Generated {len(solutions)} solutions for initial population"
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
1. Maintains the same functionality and passes the same tests
2. Explores a different algorithmic approach or implementation style
3. Could potentially be more efficient or clearer

Only return the modified code, no explanations or markdown."""

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

Only return the new combined code, no explanations or markdown."""

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
        logger.debug(f"Feedback: {feedback}")
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
Only return the fixed code, no explanations or markdown."""

        logger.trace(f"Original individual to edit:\n{individual}")
        result = self._generate_and_extract(prompt)
        logger.info("CodeOperator: Edit completed")
        return result


class TestOperator(BaseLLMOperator):
    """
    Concrete operator specialized for test case evolution.

    This operator implements all genetic operations specifically tailored for
    evolving test cases. Unlike CodeOperator which returns multiple separate
    solutions, TestOperator works with unittest classes containing multiple
    test methods.

    Return Types:
        create_initial_population: Returns str - single unittest class with multiple test methods
        mutate: Returns str - single mutated test case
        crossover: Returns str - single combined test case
        edit: Returns str - single improved test case

    Design Note:
        TestOperator returns a single unittest class (str) for initial population,
        not a list. This is intentional because test methods belong together in a
        unittest class for proper execution. Individual test methods should be
        extracted separately in the main algorithm using test analyzers.

    Operations:
        - Initial Population: Generates a unittest class containing multiple test
          methods. Methods test different scenarios, edge cases, and boundary conditions.

        - Mutation: Modifies a test to check different edge cases or boundary
          conditions while maintaining unittest structure.

        - Crossover: Combines assertions and test scenarios from two test cases to
          create a more comprehensive test.

        - Edit: Improves a test based on feedback (e.g., test too broad, missed
          edge case, assertion errors).

    Example:
        >>> test_op = TestOperator(llm_client)
        >>> test_op.set_problem(problem)
        >>>
        >>> # Generate unittest class with 5 test methods
        >>> test_class = test_op.create_initial_population(5)
        >>> # Returns: "import unittest\\nclass TestSolution(unittest.TestCase):\\n    def test_1()...\\n    def test_2()..."
        >>>
        >>> # Extract individual tests later in your algorithm
        >>> individual_tests = extract_test_methods(test_class)
        >>>
        >>> # Mutate a test
        >>> variant = test_op.mutate(individual_tests[0])
        >>>
        >>> # Combine two tests
        >>> comprehensive_test = test_op.crossover(individual_tests[0], individual_tests[1])
        >>>
        >>> # Improve based on feedback
        >>> improved = test_op.edit(individual_tests[0], "Test too broad, add edge cases")
    """

    def create_initial_population(self, population_size: int) -> str:
        """
        Generate an initial population of test cases as a single unittest class.

        Note: Unlike CodeOperator which returns a list of separate solutions,
        TestOperator returns a single unittest class containing multiple test methods.
        Individual test methods should be extracted separately in the main algorithm.

        Args:
            population_size: The number of test methods to request in the unittest class

        Returns:
            A single string containing a unittest class with multiple test methods
        """
        logger.info(
            f"TestOperator: Creating initial test population (requesting {population_size} test methods)"
        )
        problem: CodeGenerationProblem = self._ensure_problem()

        prompt: str = f"Write {population_size} distinct unit tests in a Python code block for the following problem:\n{problem.question_content}\nThe solution is imported in the format:\n```python\n{problem.starter_code}```\nWrite the tests using unittest framework. Do not use the examples given in the problem."

        result = self._generate_and_extract(prompt)

        # Count test methods for logging
        test_count = result.count("def test_")
        logger.info(
            f"TestOperator: Generated unittest class with {test_count} test methods"
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

Generate a modified test case that:
1. Tests different edge cases or boundary conditions
2. Maintains the same unittest structure
3. Explores scenarios not covered by the original test

Only return the modified test code, no explanations or markdown."""

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

Create a new test case that combines assertions and test scenarios from both.
The new test should:
1. Include complementary assertions from both tests
2. Cover a broader range of cases
3. Maintain proper unittest structure

Only return the new combined test code, no explanations or markdown."""

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
        logger.debug(f"Feedback: {feedback}")
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
Ensure it remains a valid unittest test case.
Only return the improved test code, no explanations or markdown."""

        logger.trace(f"Original test to edit:\n{individual}")
        result = self._generate_and_extract(prompt)
        logger.info("TestOperator: Edit completed")
        return result
