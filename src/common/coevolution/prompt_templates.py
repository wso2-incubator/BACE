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
    "Test method name should be unique and descriptive of the test case.\n"
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
    "- Write {population_size} distinct solutions to solve the problem described below.\n"
    "- Each solution should implement a different algorithmic approach or technique.\n"
    "- Each code should strictly follow the starter code structure provided in <starter_code>\n"
    "- Maintain good code qualities and adhere to coding best practices.\n"
    "- Write a concise docstring for each solution with a clear explanation of its approach and reasoning.\n"
    "- Return each solution in a separate Python code block.\n"
    "</task>\n\n"
    "<problem>\n"
    "{question_content}\n"
    "</problem>\n\n" + _STARTER_CODE_BLOCK + "\n\n"
)

CROSSOVER_CODE = (
    _CODER_ROLE + "\n\n"
    "<task>\n"
    "Create a new solution that intelligently combines the best aspects of two candidate solutions provided in <candidate_solution_1> and <candidate_solution_2>.\n"
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
    "1. First verify the correctness of the provided solution.\n"
    "-- If any issues are found, fix them.\n"
    "-- If no issues are found: "
    "\ti. Maintain the same functionality.\n"
    "\tii. Explore a different algorithmic approach or implementation style.\n"
    "\tiii. Improve efficiency or clarity where possible.\n"
    "</task>\n\n"
    "<problem>\n"
    "{question_content}\n"
    "</problem>\n\n"
    "<candidate_solution>\n"
    "```python\n{individual}\n```\n"
    "</candidate_solution>\n\n"
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
    "If no issues are found, try to improve the code slightly in terms of efficiency or clarity.\n"
    "</task>\n\n"
    + _STARTER_CODE_BLOCK
    + "\n"
    + _STARTER_CODE_NOTE
    + "\n"
    + _CODE_FORMAT_INSTRUCTION
    + "\n"
)

EDIT_CODE_FIX_FAIL_ONLY = (
    _CODER_ROLE + "\n\n"
    "<problem>\n"
    "{question_content}\n"
    "</problem>\n\n"
    "<current_solution>\n"
    "```python\n{individual}\n```\n"
    "</current_solution>\n\n"
    "<failing_test_case>\n"
    "{failing_test_case}\n"
    "</failing_test_case>\n\n"
    "<error_trace>\n"
    "{error_trace}\n"
    "</error_trace>\n\n"
    "<task>\n"
    "Utilizing the feedback, generate a new code solution that addresses the issues raised.\n"
    "Your revised solution just have to pass the failing test cases. you do not need to optimize or improve beyond that.\n"
    "The problem and the code might not align well, so focus on making the code pass the tests rather than fully solving the problem.\n"
    "Again: only ensure the code passes the failing tests; full problem compliance is not required.\n"
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
Your input will consist of:
1. A problem description in <problem> tag
2. A code snippet in <current_solution> tag
3. Candidate test cases and their results in <feedback> tag showing how various test cases passed or failed against this code.
4. For each test case we provide code, test results, error logs, or stack traces.

Your goal is to analyze the test failures and identify the root cause of the error. 
Re-evaluate the code against the problem statement to find logic gaps or missed edge cases.

Critically, you must verify the validity of the test cases against the problem description. 
The provided test cases may contain errors or incorrect expected outputs. 
Treat the <problem> description as the absolute source of truth.

If a test case contradicts the problem description, do not modify the code to satisfy the incorrect test; instead, ensure the code correctly implements the logic defined in the problem description. 
Once the issue is isolated, fix the <current_solution> with minimal changes.

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
</coding_guidelines>

<analysis_strategy>
1. **Analyze Test Results:** Look at the failure logs to pinpoint where behavior diverges from expectations.
2. **Verify Tests:** Check if the failing test aligns with the <problem> description. If the test is invalid, note this in your explanation.
3. **Formulate Fix:** Determine the smallest specific change required.
</analysis_strategy>

<output_formatting>
You must structure your response in exactly the following order:

1. **## Root cause identification**
   - Explicitly state the root cause found during the trace (e.g., "The loop terminates one step too early because of `range(n-1)`").
   - State how the root cause and the ground truth from the problem description in <problem> relate?
   - State how the root cause explain the test failures observed in <feedback>?
   - If a test case was invalid/incorrect based on the problem description, explicitly state: "Test case [Input] was ignored because it contradicts the problem statement."

2. **## Explanation of Fix**
   - A concise summary of how the new code resolves the issue identified above.

3. **## Corrected Code**
   - Provide the **ENTIRE** fixed script in a single Markdown code block (```python ... ```).
   - Ensure the solution adheres to the `<starter_code>` function signature.
   - Include a concise docstring explaining the approach.

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
    - Step by step verify that each expected output is accurate and reflects the function's intended behavior.
    - Do not use the examples given in the problem.
    - Do not write any other top-level functions or classes.
    - The solution code will be appended to the test cases for execution, do not include it in your response.
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
    "Step by step verify that the expected output is accurate and reflects the function's intended behavior.\n"
    "</task>\n" + _TEST_METHOD_FORMAT_INSTRUCTION + "\n"
)

MUTATE_TEST = (
    _TESTER_ROLE + "\n\n"
    "<problem>\n"
    "{question_content}\n"
    "</problem>\n\n"
    "<base_test>\n"
    "The following is a test case candidate:\n"
    "`python\n{individual}\n`\n"
    "</base_test>\n\n"
    "<task>\n"
    "1. Verify the correctness of the provided test case.\n"
    "2. If any issues are found, fix them.\n"
    "3. Else if no issues are found, generate a new test case that explores different input scenarios or edge cases.\n"
    "4. Step-by-step verify that the expected output is accurate and reflects the function's intended behavior.\n"
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
    "<feedback>\n"
    "{feedback}\n"
    "</feedback>\n\n"
    "<task>\n"
    "Utilize the feedback to generate a new test case that addresses the issues raised.\n"
    "If no issues are found or if you strongly believe the original test case was correct, return a slightly modified version of the original test case.\n"
    "</task>\n" + _TEST_METHOD_FORMAT_INSTRUCTION + "\n"
)


EDIT_TEST_AGENTIC = (
    """<system_role>
You are an expert AI QA engineer and software tester.
You do not have access to a file system or execution environments.
Your input will consist of:
1. A problem description in <problem> tag
2. A current test case (unit test) in <current_test> tag
3. Feedback logs in <feedback_from_solutions> tag showing how various solution candidates (some correct, some incorrect) performed against this test.

Your goal is to analyze the feedback to identify weaknesses or errors in the test case and provide a refined, robust version.
Critically, you must verify the validity of the solution candidates against the problem description.
Treat the <problem> description as the absolute source of truth.
If a solution candidate contradicts the problem description, do not modify the test to satisfy the incorrect solution; instead, ensure the test correctly validates the logic defined in the problem description.
</system_role>

<personality>
Your default personality is concise, direct, and critical (in a constructive way).
- Focus on edge cases and correctness.
- Act like a senior Test Engineer: rigorous, precise, and logical.
</personality>

<testing_guidelines>
When providing the fixed test case, adhere to these standards:
- **Validity:** The test MUST pass for a correct solution. If valid code failed, the test logic is flawed—fix it.
- **Discrimination:** The test MUST fail for incorrect solutions. If buggy code passed, the test is too weak—add specific assertions to catch those bugs.
- **Standard Library:** This test is a unit test method using Python's unittest framework.
- **Isolation:** Ensure tests are independent and do not rely on external state unless provided.
- **Complexity:** Avoid unneeded complexity. Simple is better.
- **Naming:** Use descriptive names for test methods reflecting the scenario being tested.
</testing_guidelines>

<analysis_strategy>
Before generating the test code, perform the following internal analysis:
1. **Analyze Feedback:**
    -- Did valid solutions fail? -> The test has a logic error or incorrect assumption.
    -- Did invalid solutions pass? -> The test lacks coverage for specific edge cases.
    -- Did the test crash (Syntax/Runtime error)? -> The test code is malformed.
2. **Trace Logic:** Compare the problem requirements against the current test assertions.
3. **Formulate Fix:** -- If strengthening: Add specific input/output pairs that target the missed edge case.
    -- If fixing: Correct the expected output values or assertion logic.
</analysis_strategy>

<output_formatting>
1. **Brief Explanation:** Start with a concise sentence explaining why the test is being changed (e.g., "Added an assertion for empty list input to catch the breakdown in Solution B" or "Fixed incorrect expected value for input 'xyz'.").
2. **Code Blocks:** Provide the fixed test method in a Python code block (e.g., ```python ... ```). This should be a single unittest test method.
3. **Test Name**: The test method name should be changed to reflect the specific scenario being tested.
4. **Helper Functions**: Use helper functions only if absolutely necessary to keep the test clear and focused. Helper functions if used should be within the test method.
</output_formatting>"""
    + "\n\n"
    """
<problem>
{question_content}
</problem>

<current_test>
```python
{individual}
```
</current_test>

<feedback_from_solutions> {feedback} </feedback_from_solutions>
 """
)

__all__ = [
    "INITIAL_TEST_AGENT_CODER_STYLE",
    "INITIAL_CODE",
    "CROSSOVER_CODE",
    "MUTATE_CODE",
    "EDIT_CODE",
    "EDIT_CODE_AGENTIC",
    "INITIAL_TEST",
    "CROSSOVER_TEST",
    "MUTATE_TEST",
    "EDIT_TEST",
    "EDIT_TEST_AGENTIC",
]
