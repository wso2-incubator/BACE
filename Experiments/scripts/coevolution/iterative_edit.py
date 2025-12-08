from common.code_preprocessing.extraction import extract_code_block_from_response
from common.code_preprocessing.transformation import extract_test_methods_code
from common.coevolution import lcb_dataset
from common.coevolution.core.individual import CodeIndividual, TestIndividual
from common.coevolution.core.interfaces import Operations
from common.coevolution.core.mock import MockPareto, MockTestBlockRebuilder
from common.coevolution.core.population import CodePopulation, TestPopulation
from common.coevolution.execution import ExecutionSystem
from common.coevolution.feedback import CodeFeedbackGenerator
from common.coevolution.prompt_templates import (
    _CODE_FORMAT_INSTRUCTION,
    _CODER_ROLE,
    _STARTER_CODE_BLOCK,
    _STARTER_CODE_NOTE,
)
from common.llm_client import create_llm_client
from common.sandbox import create_safe_test_environment

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
        provider="openai", model="gpt-5-mini", reasoning_effort="minimal"
    )

    max_iterations = 5
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
