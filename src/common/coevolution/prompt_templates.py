"""
Centralized prompt templates for coevolution LLM operators.

Use named placeholders and Python str.format() to substitute values.
Keeping templates in a separate file makes them easier to review and
modify without touching code logic.
"""

# Code generation prompts
INITIAL_CODE = (
    "Write {population_size} distinct solutions to solve the following problem:\n\n"
    "{question_content}\n\n"
    "Starter Code:\n```python\n{starter_code}\n```\n\n"
    "Each solution should be in a separate Python code block."
)

CROSSOVER_CODE = (
    "Given this programming problem:\n{question_content}\n\n"
    "And these two code solutions:\n\n"
    "Solution 1:\n```python\n{parent1}\n```\n\n"
    "Solution 2:\n```python\n{parent2}\n```\n\n"
    "Create a new solution that intelligently combines the best aspects of both solutions.\n"
    "Consider combining:\n- Better algorithms from either solution\n- More efficient data structures\n- Clearer variable names or logic flow\n\n"
    "Return the new combined code in a python code block."
)

MUTATE_CODE = (
    "Given this programming problem:\n{question_content}\n\n"
    "And this code solution:\n```python\n{individual}\n```\n\n"
    "Generate a modified version of the code that:\n"
    "1. Maintains the same functionality\n"
    "2. Explores a different algorithmic approach or implementation style\n"
    "3. Could potentially be more efficient or clearer\n\n"
    "Return the modified code in a python code block."
)

EDIT_CODE = (
    "Given this programming problem:\n{question_content}\n\n"
    "This code solution:\n```python\n{individual}\n```\n\n"
    "And this error/feedback:\n{feedback}\n\n"
    "Fix the code to address the error while maintaining the overall approach.\n"
    "Return the fixed code in a python code block."
)

# Test generation/editing prompts
INITIAL_TEST = (
    "Write {population_size} distinct unit tests in a Python code block for the following problem:\n"
    "{question_content}\n"
    "The solution is imported in the format:\n```python\n{starter_code}```\n"
    "Write the tests using unittest framework. Do not use the examples given in the problem."
)

INITIAL_TEST_AGENT_CODER_STYLE = """**Role**: As a tester, your task is to create comprehensive test cases for the incomplete function. These test cases should encompass Basic, Edge, and Large Scale scenarios to ensure the code's robustness, reliability, and scalability.

PROBLEM: {question_content}

The solution code is imported in the format:
```python
{starter_code}
```

**1. Basic Test Cases**:
- **Objective**: To verify the fundamental functionality of the `has_close_elements` function under normal conditions.

**2. Edge Test Cases**:
- **Objective**: To evaluate the function's behavior under extreme or unusual conditions.

**3. Large Scale Test Cases**:
- **Objective**: To assess the function’s performance and scalability with large data samples.

**Instructions**:
- Implement a comprehensive set of test cases following the guidelines above.
- Ensure each test case is well-documented with comments explaining the scenario it covers.
- Pay special attention to edge cases as they often reveal hidden bugs.
- For large-scale tests, focus on the function's efficiency and performance under heavy loads.

- The format of test cases should be in Python unittest framework. Do not use the examples given in the problem. You should write {population_size} distinct test cases.
"""


CROSSOVER_TEST = (
    "Given this programming problem:\n{question_content}\n\n"
    "And these two test cases:\n\n"
    "Test 1:\n```python\n{parent1}\n```\n\n"
    "Test 2:\n```python\n{parent2}\n```\n\n"
    "Summarize test 1 and test 2, then create a new test case.\n\n"
    "Return only the new test method code in a python code block."
)

MUTATE_TEST = (
    "Given this programming problem:\n{question_content}\n\n"
    "And this test case:\n```python\n{individual}\n```\n\n"
    "Generate a modified test case that tests different edge cases or scenarios.\n"
    "Ensure it remains a valid unittest test case and contains only a single assertion.\n\n"
    "Return only the test method code in a python code block."
)

EDIT_TEST = (
    "Given this programming problem:\n{question_content}\n\n"
    "This test case:\n```python\n{individual}\n```\n\n"
    "And this feedback:\n{feedback}\n\n"
    "Summarize the feedback and provide a single new test case method that can help identify these issues.\n"
    "Only return the new test code in a python code block"
)

__all__ = [
    "INITIAL_CODE",
    "CROSSOVER_CODE",
    "MUTATE_CODE",
    "EDIT_CODE",
    "INITIAL_TEST",
    "CROSSOVER_TEST",
    "MUTATE_TEST",
    "EDIT_TEST",
]
