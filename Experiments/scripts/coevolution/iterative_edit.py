from common.code_preprocessing.extraction import extract_code_block_from_response
from common.code_preprocessing.transformation import extract_test_methods_code
from common.coevolution import lcb_dataset
from common.coevolution.core.individual import CodeIndividual, TestIndividual
from common.coevolution.core.interfaces import Operations
from common.coevolution.core.mock import MockPareto, MockTestBlockRebuilder
from common.coevolution.core.population import CodePopulation, TestPopulation
from common.coevolution.execution import ExecutionSystem
from common.coevolution.feedback import CodeFeedbackGenerator
from common.llm_client import create_llm_client
from common.sandbox import create_safe_test_environment

_SYSTEM_PROMPT = """<system_role>
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
</output_formatting>
"""

EDIT_CODE = """
{system_prompt}

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


def main() -> None:
    # Load problem
    problems = lcb_dataset.load_code_generation_dataset(
        release_version="release_v6",
        start_date="2025-03-01",
        end_date="2025-05-10",
        difficulty=lcb_dataset.Difficulty.HARD,
    )

    problem_id = "arc194_c"
    problem = next((p for p in problems if p.question_id == problem_id), None)
    if problem is None:
        raise ValueError(f"Problem with ID {problem_id} not found.")

    # Initialize Test Population
    test_class_block = lcb_dataset.LCBDatasetTestBlockBuilder.build_test_class_block(
        problem.public_test_cases, problem.starter_code
    )

    test_methods = extract_test_methods_code(test_class_block)

    FIXED_TEST_PROBABILITY = 1.0
    test_individuals = [
        TestIndividual(
            snippet=method,
            probability=FIXED_TEST_PROBABILITY,
            creation_op=Operations.INITIAL,
            generation_born=0,
            parent_ids=[],
        )
        for method in test_methods
    ]

    test_population = TestPopulation(
        individuals=test_individuals,
        pareto=MockPareto(),
        test_block_rebuilder=MockTestBlockRebuilder(),
        test_class_block=test_class_block,
        generation=0,
    )

    # Initial Code
    C1 = """
class Solution:
    def sol(self, input_str: str) -> str:
        data = list(map(int, input_str.split()))
        it = iter(data)
        N = next(it)
        A = [next(it) for _ in range(N)]
        B = [next(it) for _ in range(N)]
        C = [next(it) for _ in range(N)]
        # Strategy: flip some 1->0 that reduce future costs before doing 0->1 that raise them.
        ones_sum = sum(C[i] for i in range(N) if A[i] == 1)
        idxs_to_zero = [i for i in range(N) if A[i] == 1 and B[i] == 0]
        idxs_to_one = [i for i in range(N) if A[i] == 0 and B[i] == 1]
        # Flip those 1->0 in descending C to maximize immediate reduction early.
        idxs_to_zero.sort(key=lambda i: -C[i])
        total = 0
        for i in idxs_to_zero:
            A[i] = 0
            ones_sum -= C[i]
            total += ones_sum
        # Then flip 0->1 in ascending C to minimize increments when many ones exist
        idxs_to_one.sort(key=lambda i: C[i])
        for i in idxs_to_one:
            A[i] = 1
            ones_sum += C[i]
            total += ones_sum
        return str(total)
"""

    code_individual = CodeIndividual(
        snippet=C1,
        creation_op=Operations.INITIAL,
        generation_born=0,
        parent_ids=[],
        probability=0.75,
    )

    code_population = CodePopulation([code_individual], 0)

    # Setup execution
    sandbox = create_safe_test_environment()
    exec_system = ExecutionSystem(enable_multiprocessing=True, num_workers=10)
    code_feedback_generator = CodeFeedbackGenerator()
    llm_client = create_llm_client(
        provider="openai", model="gpt-5-codex", reasoning_effort=None
    )

    max_iterations = 2
    current_generation = 0

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        # Execute tests
        exec_results = exec_system.execute_tests(
            code_population, test_population, sandbox=sandbox
        )
        observation_matrix = exec_system.build_observation_matrix(
            code_population, test_population, exec_results
        )

        print(f"Observation Matrix:\n{observation_matrix}")

        # Check if all tests pass
        observation_matrix_row = observation_matrix[0]
        # sum should equal number of tests
        if sum(observation_matrix_row) == len(test_population.individuals):
            print("All tests passed! Stopping coevolution.")
            break

        # Generate feedback
        feedback = code_feedback_generator.generate_feedback(
            observation_matrix, exec_results, test_population, 0
        )
        # Edit code with LLM
        prompt = EDIT_CODE.format(
            question_content=problem.question_content,
            individual=code_population.individuals[0].snippet,
            feedback=feedback,
            starter_code=problem.starter_code,
            system_prompt=_SYSTEM_PROMPT,
        )

        print(f"LLM Prompt:\n{prompt}\n{'-' * 40}")
        response = llm_client.generate(prompt)
        print(f"LLM Response:\n{response}")

        code_block = extract_code_block_from_response(response)

        # Create new code individual
        edited_code_individual = CodeIndividual(
            snippet=code_block,
            creation_op=Operations.EDIT,
            generation_born=current_generation + 1,
            parent_ids=[code_population.individuals[0].id],
            probability=0.75,
        )

        # Update population
        code_population.set_next_generation([edited_code_individual])
        current_generation += 1  # Final results
    print("\n--- Final Results ---")
    exec_results = exec_system.execute_tests(
        code_population, test_population, sandbox=sandbox
    )
    observation_matrix = exec_system.build_observation_matrix(
        code_population, test_population, exec_results
    )
    print(f"Final Observation Matrix:\n{observation_matrix}")

    for test_result in exec_results[0].test_results:
        print(f"{test_result.status}: {test_result.details}")


if __name__ == "__main__":
    main()
