# src/common/coevolution/prompt_templates.py
"""
Centralized prompt templates for coevolution LLM operators.

Use named placeholders and Python str.format() to substitute values.
Keeping templates in a separate file makes them easier to review and
modify without touching code logic.
"""

_CODER_ROLE = "<system_role>You are an expert software programmer.</system_role>"
_TESTER_ROLE = "<system_role>You are an expert software tester.</system_role>"

_TEST_METHOD_FORMAT_INSTRUCTION = (
    "<output_formatting>\n"
    "Return only the code for a single unittest test method in a python code block.\n"
    "</output_formatting>"
)

_STARTER_CODE_BLOCK = "<starter_code>\n```python\n{starter_code}\n```\n</starter_code>"

_STARTER_CODE_NOTE = (
    "<instruction>\n"
    "You MUST build upon the provided starter code found in the <starter_code> tag.\n"
    "Keep the function signature exactly as defined.\n"
    "</instruction>"
)

_CODE_FORMAT_INSTRUCTION = (
    "<output_formatting>\n"
    "Return the solution code in a valid python code block (```python ... ```).\n"
    "</output_formatting>"
)


# ==========================================
# Code Generation Prompts
# ==========================================

INITIAL_CODE = (
    _CODER_ROLE + "\n\n"
    "<task>\n"
    "Write {population_size} distinct solutions to solve the problem described below.\n"
    "Each solution should implement a different algorithmic approach or technique.\n"
    "Return each solution in a separate Python code block.\n"
    "</task>\n\n"
    "<problem>\n"
    "{question_content}\n"
    "</problem>\n\n" + _STARTER_CODE_BLOCK + "\n\n"
)

CROSSOVER_CODE = (
    _CODER_ROLE + "\n\n"
    "<task>\n"
    "Create a new solution that intelligently combines the best aspects of two parent solutions.\n"
    "</task>\n\n"
    "<problem>\n"
    "{question_content}\n"
    "</problem>\n\n"
    "<candidate_solution_1>\n"
    "```python\n{parent1}\n```\n"
    "</candidate_solution_1>\n\n"
    "<candidate_solution_2>\n"
    "```python\n{parent2}\n```\n"
    "</candidate_solution_2>\n\n"
    + _STARTER_CODE_BLOCK
    + "\n"
    + _STARTER_CODE_NOTE
    + "\n"
    + _CODE_FORMAT_INSTRUCTION
    + "\n"
)

MUTATE_CODE = (
    _CODER_ROLE + "\n\n"
    "<task>\n"
    "Generate a modified version of the provided code solution.\n"
    "1. Maintain the same functionality.\n"
    "2. Explore a different algorithmic approach or implementation style.\n"
    "3. Improve efficiency or clarity where possible.\n"
    "</task>\n\n"
    "<problem>\n"
    "{question_content}\n"
    "</problem>\n\n"
    "<individual_solution>\n"
    "```python\n{individual}\n```\n"
    "</individual_solution>\n\n"
    + _STARTER_CODE_BLOCK
    + "\n"
    + _STARTER_CODE_NOTE
    + "\n"
    + _CODE_FORMAT_INSTRUCTION
    + "\n"
)


EDIT_CODE = (
    _CODER_ROLE + "\n\n"
    "<problem>\n"
    "{question_content}\n"
    "</problem>\n\n"
    "<current_solution>\n"
    "```python\n{individual}\n```\n"
    "</current_solution>\n\n"
    "<feedback>\n"
    "{feedback}\n"
    "</feedback>\n\n"
    "<task>\n"
    "Utilizing the feedback, generate a new code solution that addresses the issues raised.\n"
    "</task>\n\n"
    + _STARTER_CODE_BLOCK
    + "\n"
    + _STARTER_CODE_NOTE
    + "\n"
    + _CODE_FORMAT_INSTRUCTION
    + "\n"
)


EDIT_CODE_AGENTIC = (
    """<system_role>
You are an expert AI programming assistant specializing in code completion and bug fixing. 
You do not have access to a file system, compilers, or execution environments. 
Your input will consist of:
1. A problem description
2. A buggy code snippet
3. Test results, error logs, or stack traces.

Your goal is to analyze the test results, identify the root cause of the bug, and provide the corrected code.
</system_role>

<personality>
Your default personality is concise, direct, and friendly. 
- Communicate efficiently; avoid excessively verbose explanations.
- Prioritize actionable fixes.
- Act like a senior pair programmer: helpful, precise, and logically sound.
</personality>

<coding_guidelines>
When providing fixed code, adhere to these standards:
- **Fix the Root Cause:** Do not apply surface-level patches if a deeper logic error exists.
- **Minimal Changes:** Keep changes consistent with the style of the existing code. Avoid reformatting unrelated code.
- **Complexity:** Avoid unneeded complexity. Simple is better.
- **Naming:** Do not use one-letter variable names unless standard for the language (e.g., `i` in loops).
</coding_guidelines>

<analysis_strategy>
Before generating the code, perform the following internal analysis:
1. **Analyze Test Results:** Look at the provided failure logs to pinpoint exactly where the code diverges from expected behavior.
2. **Trace Execution:** Mentally simulate the code execution with the failing input to find the logic gap.
3. **Formulate Fix:** Determine the smallest specific change required to make the tests pass without breaking other functionality.
</analysis_strategy>

<output_formatting>
1. **Brief Explanation:** Start with a concise sentence identifying the bug (e.g., "The off-by-one error in the loop caused the index out of bounds exception.").
2. **Code Blocks:** Provide the fixed code in Markdown code blocks (e.g., ```python ... ```). 
3. **Starter code:** ensure your solution adheres to the starter code structure.
</output_formatting>"""
    + "\n\n"
    """
<problem>
{question_content}
</problem>

<current_solution>
```python\n{individual}```
</current_solution>

<feedback>
{feedback}
</feedback>

<starter_code>
```python\n{starter_code}```
</starter_code>
"""
)

# ==========================================
# Test Generation Prompts
# ==========================================

INITIAL_TEST = (
    _TESTER_ROLE + "\n\n"
    "<task>\n"
    "Write {population_size} distinct unit tests in a Python code block for the problem below.\n"
    "</task>\n\n"
    "<problem>\n"
    "{question_content}\n"
    "</problem>\n\n"
    "<context>\n"
    "The solution is imported in the format:\n"
    "```python\n{starter_code}```\n"
    "</context>\n\n"
    "<constraints>\n"
    "1. Write tests using the unittest framework.\n"
    "2. Do NOT use the examples given in the problem description.\n"
    "</constraints>"
)


INITIAL_TEST_AGENT_CODER_STYLE = """
<system_role>
As a tester, your task is to create comprehensive test cases for the incomplete function. These test cases should encompass Basic, Edge, and Large Scale scenarios to ensure the code's robustness, reliability, and scalability.
</system_role>

<problem>
{question_content}
</problem>

<context>
The solution code is imported in the format:
```python
{starter_code}
```

</context>

<instructions>
**1. Basic Test Cases**:

    - **Objective**: To verify the fundamental functionality of the function under normal conditions.

**2. Edge Test Cases**:

    - **Objective**: To evaluate the function's behavior under extreme or unusual conditions.

**3. Large Scale Test Cases**:

    - **Objective**: To assess the function's performance and scalability with large data samples.

**General Instructions**:

    - Implement a comprehensive set of test cases following the guidelines above.
    - Ensure each test case is well-documented with comments explaining the scenario it covers.
    - Pay special attention to edge cases as they often reveal hidden bugs.
    - For large-scale tests, focus on the function's efficiency and performance under heavy loads.
    - The format of test cases should be in Python unittest framework in a python code block.
    - Do not use the examples given in the problem.
    - Do not write any other top-level functions or classes.
    - You should write {population_size} distinct test cases.

</instructions>
"""


CROSSOVER_TEST = (
    _TESTER_ROLE + "\n\n"
    "<problem>\n"
    "{question_content}\n"
    "</problem>\n\n"
    "<context>\n"
    "The following two test cases were identified to be good tests.\n"
    "</context>\n\n"
    "<test_case_1>\n"
    "`python\n{parent1}\n`\n"
    "</test_case_1>\n\n"
    "<test_case_2>\n"
    "`python\n{parent2}\n`\n"
    "</test_case_2>\n\n"
    "<task>\n"
    "Create a new test case that covers the gaps in the existing tests.\n"
    "</task>\n" + _TEST_METHOD_FORMAT_INSTRUCTION + "\n"
)

MUTATE_TEST = (
    _TESTER_ROLE + "\n\n"
    "<problem>\n"
    "{question_content}\n"
    "</problem>\n\n"
    "<base_test>\n"
    "The following test case was identified to be a good test:\n"
    "`python\n{individual}\n`\n"
    "</base_test>\n\n"
    "<task>\n"
    "Generate a new test case that explores different input scenarios or edge cases.\n"
    "Ensure it remains a valid unittest test case.\n"
    "</task>\n" + _TEST_METHOD_FORMAT_INSTRUCTION + "\n"
)

EDIT_TEST = (
    _TESTER_ROLE + "\n\n"
    "<problem>\n"
    "{question_content}\n"
    "</problem>\n\n"
    "<current_test>\n"
    "`python\n{individual}\n`\n"
    "</current_test>\n\n"
    "<feedback_log>\n"
    "{feedback}\n"
    "</feedback_log>\n\n"
    "<task>\n"
    "Utilize the feedback to generate a new test case that addresses the issues raised.\n"
    "If no issues are found or if you strongly believe the original test case was correct, return a slightly modified version of the original test case.\n"
    "</task>\n" + _TEST_METHOD_FORMAT_INSTRUCTION + "\n"
)

__all__ = [
    "INITIAL_TEST_AGENT_CODER_STYLE",
    "INITIAL_CODE",
    "CROSSOVER_CODE",
    "MUTATE_CODE",
    "EDIT_CODE",
    "INITIAL_TEST",
    "CROSSOVER_TEST",
    "MUTATE_TEST",
    "EDIT_TEST",
]
