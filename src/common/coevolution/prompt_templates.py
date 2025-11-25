"""
Centralized prompt templates for coevolution LLM operators.

Use named placeholders and Python str.format() to substitute values.
Keeping templates in a separate file makes them easier to review and
modify without touching code logic.
"""

_CODER_ROLE = "You are a software programmer."
_TESTER_ROLE = "You are a software tester."
_TEST_METHOD_FORMAT_INSTRUCTION = (
    "Return only the code for a single unittest test method in a python code block."
)
_STARTER_CODE_NOTE = "Make sure to build upon the provided starter code.\nStarter Code:\n```python\n{starter_code}\n```"
_CODE_FORMAT_INSTRUCTION = "Return the code in a python code block."


# Code generation prompts
INITIAL_CODE = (
    _CODER_ROLE + "\n\n"
    "Write {population_size} distinct solutions to solve the following problem:\n\n"
    "{question_content}\n\n"
    "Starter Code:\n```python\n{starter_code}\n```\n\n"
    "Each solution should be in a separate Python code block following the starter code structure.\n"
)

CROSSOVER_CODE = (
    _CODER_ROLE + "\n\n"
    "Given this programming problem:\n{question_content}\n\n"
    "And these two code solutions:\n\n"
    "Solution 1:\n```python\n{parent1}\n```\n\n"
    "Solution 2:\n```python\n{parent2}\n```\n\n"
    "Create a new solution that intelligently combines the best aspects of both solutions.\n"
    + _STARTER_CODE_NOTE
    + "\n"
    + _CODE_FORMAT_INSTRUCTION
    + "\n"
)

MUTATE_CODE = (
    _CODER_ROLE + "\n\n"
    "Given this programming problem:\n{question_content}\n\n"
    "And this code solution:\n```python\n{individual}\n```\n\n"
    "Generate a modified version of the code that:\n"
    "1. Maintains the same functionality\n"
    "2. Explores a different algorithmic approach or implementation style\n"
    "3. Could potentially be more efficient or clearer\n\n"
    + _STARTER_CODE_NOTE
    + "\n"
    + _CODE_FORMAT_INSTRUCTION
    + "\n"
)

EDIT_CODE = (
    _CODER_ROLE + "\n\n"
    "Given this programming problem:\n{question_content}\n\n"
    "This code solution:\n```python\n{individual}\n```\n\n"
    "And this error/feedback:\n{feedback}\n\n"
    "Utilizing the feedback, generate a new code solution that addresses the issues raised.\n"
    "If no issues are found or if you strongly believe the original code is correct, return a slightly modified version of the original code.\n"
    + _STARTER_CODE_NOTE
    + "\n"
    + _CODE_FORMAT_INSTRUCTION
    + "\n"
)

# Test generation/editing prompts
INITIAL_TEST = (
    _TESTER_ROLE + "\n\n"
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
- **Objective**: To assess the function's performance and scalability with large data samples.

**Instructions**:
- Implement a comprehensive set of test cases following the guidelines above.
- Ensure each test case is well-documented with comments explaining the scenario it covers.
- Pay special attention to edge cases as they often reveal hidden bugs.
- For large-scale tests, focus on the function's efficiency and performance under heavy loads.

- The format of test cases should be in Python unittest framework in a python code block. Do not use the examples given in the problem. You should write {population_size} distinct test cases. 
"""


CROSSOVER_TEST = (
    _TESTER_ROLE + "\n\n"
    "PROBLEM: \n{question_content}\n\n"
    "The following two test cases were identified to be good tests\n\n"
    "Test 1:\n```python\n{parent1}\n```\n\n"
    "Test 2:\n```python\n{parent2}\n```\n\n"
    "Your task is to create a new test case that covers the gaps in the existing tests.\n"
    + _TEST_METHOD_FORMAT_INSTRUCTION
    + "\n"
)

MUTATE_TEST = (
    _TESTER_ROLE + "\n\n"
    "PROBLEM:\n{question_content}\n\n"
    "The following test case was identified to be a good test\n```python\n{individual}\n```\n\n"
    "Generate a new test case that explores different input scenarios or edge cases.\n"
    "Ensure it remains a valid unittest test case\n\n"
    + _TEST_METHOD_FORMAT_INSTRUCTION
    + "\n"
)

EDIT_TEST = (
    _TESTER_ROLE + "\n\n"
    "Given this programming problem:\n{question_content}\n\n"
    "This test case:\n```python\n{individual}\n```\n\n"
    "And this feedback:\n{feedback}\n\n"
    "Utilize the feedback to generate a new test case that addresses the issues raised.\n"
    "If no issues are found or if you strongly believe the original test case was correct, return a slightly modified version of the original test case.\n"
    + _TEST_METHOD_FORMAT_INSTRUCTION
    + "\n"
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
