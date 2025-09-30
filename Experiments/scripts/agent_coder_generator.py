"""
AgentCoder Generator for HumanEval Dataset - Refactored Version

This module provides a professional, modular implementation of the AgentCoder
multi-agent code generation system with proper separation of concerns.
"""

from human_eval.data import write_jsonl, read_problems
from common import AgentCoderConfig, create_llm_from_params, CodeProcessor, create_safe_test_environment, TestExecutionResult
from typing import List, TypedDict, Dict, Annotated, Literal, Optional, Tuple, Any
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

from tqdm import tqdm
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class TestResultsSummary:
    """Summary of test execution results."""
    tests_passed: int
    tests_failed: int
    tests_errors: int

    @property
    def total_tests(self) -> int:
        """Total number of tests run."""
        return self.tests_passed + self.tests_failed + self.tests_errors

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.tests_passed / self.total_tests) * 100


@dataclass(frozen=True)
class IterationResult:
    """Result from a single iteration of the agent workflow."""
    task_id: str
    iteration: int
    completion: str
    success: bool
    test_results_summary: Optional[TestResultsSummary]
    test_cases: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        return result


@dataclass(frozen=True)
class FinalCompletion:
    """Final completion result for a task."""
    task_id: str
    completion: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class AgentState(TypedDict):
    """State definition for the multi-agent workflow."""
    solution: str
    tests: str
    test_results: Optional[TestExecutionResult]
    max_iterations: int
    tests_sent_to_programmer: bool
    solution_sent_to_tester: bool
    programmer_messages: Annotated[List[BaseMessage], add_messages]
    tester_messages: Annotated[List[BaseMessage], add_messages]


class PromptProvider:
    """Handles generation of prompts for different agents."""

    def __init__(self) -> None:
        self.programmer_system_prompt = "You are an expert software developer."
        self.test_designer_system_prompt = "You are an expert program tester."

    def get_programmer_prompt(self, problem: str) -> str:
        """Get the programmer agent prompt."""
        return f"""As a programmer, you are required to complete the function. Use a Chain-of-Thought approach to break down the problem, create pseudocode, and then write the code in Python language. Ensure that your code is efficient, readable, and well-commented.

**Input Code Snippet**: 
```python 
{problem}
# Add your code here to complete the function 
```  
**Instructions**: 
1. **Understand and Clarify**: Make sure you understand the task. 
2. **Algorithm/Method Selection**: Decide on the most efficient way. 
3. **Pseudocode Creation**: Write down the steps you will follow in pseudocode. 
4. **Code Generation**: Translate your pseudocode into executable Python code."""

    def get_test_designer_prompt(self, problem: str, function_name: str) -> str:
        """Get the test designer agent prompt."""
        return f"""As a tester, your task is to create comprehensive test cases for the incomplete `{function_name}` function. These test cases should encompass Basic, Edge, and Large Scale scenarios to ensure the code's robustness, reliability, and scalability. You will never write the function itself, only the tests.

**Input Code Snippet**: 
```python 
{problem}
```

**1. Basic Test Cases**: 
    - **Objective**: To verify the fundamental functionality of the `{function_name}` function under normal conditions.  
**2. Edge Test Cases**: 
    - **Objective**: To evaluate the function's behavior under extreme or unusual conditions.  
**3. Large Scale Test Cases**: 
    - **Objective**: To assess the function's performance and scalability with large data samples.  

**Instructions**: 
    - Use the `unittest` framework for writing the test cases.
    - Implement a comprehensive set of test cases following the guidelines above. 
    - Ensure each test case is well-documented with comments explaining the scenario it covers. 
    - Pay special attention to edge cases as they often reveal hidden bugs. 
    - For large-scale tests, focus on the function's efficiency and performance under heavy loads.
    - Do not implement the function, another expert programmer will do that."""

    def generate_failure_feedback(self, state: AgentState, test_results: TestExecutionResult) -> str:
        """Generate feedback message for failed tests."""
        if not state.get('tests_sent_to_programmer'):
            feedback = "Your code attempt failed. An expert python tester wrote these test cases for you to evaluate against:\n\n"
            feedback += f"```python\n{state['tests']}\n```\n\n"
        else:
            feedback = "Your revised code still failed the tests. Please focus on the following errors:\n\n"

        for failed_test in test_results.test_details.failed_test_details:
            feedback += f"- Test Name: {failed_test.name}\n"
            feedback += f"- Details: {failed_test.details}\n\n"

        feedback += "Please analyze the errors and provide a corrected version of the code."
        return feedback


class FileManager:
    """Handles all file operations and path management for both input datasets and output results."""

    def __init__(self, config: AgentCoderConfig) -> None:
        self.config = config

    def get_base_output_path(self) -> str:
        """Get the base output file path without iteration suffix."""
        if self.config.output_filename:
            return self.config.output_filename

        # Use the config's built-in output path generation with additional suffix
        additional_suffix = f"-max{self.config.max_iterations}"
        return self.config.get_output_path("agent_coder", additional_suffix)

    def get_iteration_output_path(self, iteration: int) -> str:
        """Get the output file path for a specific iteration."""
        return self.config.get_iteration_output_path(iteration)

    def clear_output_files(self) -> None:
        """Clear all output files."""
        # Clear final output file
        final_output_file = self.get_base_output_path()
        Path(final_output_file).unlink(missing_ok=True)

        # Clear iteration files if per-iteration saving is enabled
        if self.config.save_per_iteration:
            for iter_num in range(1, self.config.max_iterations + 1):
                iteration_file = self.get_iteration_output_path(iter_num)
                Path(iteration_file).unlink(missing_ok=True)

    def save_final_completion(self, task_id: str, completion: str) -> None:
        """Save final completion to the main output file."""
        final_completion = FinalCompletion(
            task_id=task_id, completion=completion)
        final_output_file = self.get_base_output_path()
        write_jsonl(final_output_file, [
                    final_completion.to_dict()], append=True)

    def load_evaluation_problems(self) -> Tuple[Dict[str, Any], str]:
        """Load HumanEval problems from the appropriate dataset.

        Returns:
            Tuple of (problems_dict, dataset_info_string) for logging/display purposes.
        """
        dataset_path = self.config.get_dataset_path()

        if dataset_path:
            problems = read_problems(dataset_path)
            dataset_info = f"subset ({Path(dataset_path).name})"
        else:
            problems = read_problems()  # Use default full dataset
            dataset_info = "full dataset"

        return problems, dataset_info


class IterationTracker:
    """Handles iteration result tracking and saving."""

    def __init__(self, config: AgentCoderConfig, file_manager: FileManager) -> None:
        self.config = config
        self.file_manager = file_manager

    def save_iteration_result(self, iteration_result: IterationResult) -> None:
        """Save a single iteration result."""
        if not self.config.save_per_iteration:
            return

        iteration = iteration_result.iteration
        iteration_file = self.file_manager.get_iteration_output_path(iteration)
        write_jsonl(iteration_file, [iteration_result.to_dict()], append=True)

        # If successful and early completion, save to remaining iteration files
        if iteration_result.success and iteration < self.config.max_iterations:
            for remaining_iter in range(iteration + 1, self.config.max_iterations + 1):
                remaining_iteration_file = self.file_manager.get_iteration_output_path(
                    remaining_iter)
                write_jsonl(remaining_iteration_file, [
                            iteration_result.to_dict()], append=True)

    def create_iteration_result(self, task_id: str, iteration: int, completion: str,
                                success: bool, test_results: Optional[TestExecutionResult],
                                test_cases: Optional[str] = None) -> IterationResult:
        """Create an iteration result dataclass."""
        test_summary = None
        if test_results:
            test_summary = TestResultsSummary(
                tests_passed=test_results.tests_passed,
                tests_failed=test_results.tests_failed,
                tests_errors=test_results.tests_errors
            )

        return IterationResult(
            task_id=task_id,
            iteration=iteration,
            completion=completion,
            success=success,
            test_results_summary=test_summary,
            test_cases=test_cases
        )


class WorkflowBuilder:
    """Builds and manages the LangGraph workflow."""

    def __init__(self, config: AgentCoderConfig, programmer_llm: Any, tester_llm: Any, code_processor: Any, sandbox: Any) -> None:
        self.config = config
        self.programmer_llm = programmer_llm
        self.tester_llm = tester_llm
        self.code_processor = code_processor
        self.sandbox = sandbox
        self.prompt_provider = PromptProvider()

    def build_workflow(self) -> "CompiledStateGraph[AgentState, None, Any, Any]":
        """Build and compile the multi-agent workflow graph."""
        graph_builder = StateGraph(AgentState)

        # Add nodes
        graph_builder.add_node("programmer", self._programmer_agent)
        graph_builder.add_node("test_designer", self._test_designer_agent)
        graph_builder.add_node("test_executor", self._test_executor_agent)

        # Define edges
        graph_builder.add_edge(START, "programmer")
        graph_builder.add_conditional_edges(
            "programmer",
            self._route_after_programming,
            {
                "test_designer": "test_designer",
                "test_executor": "test_executor"
            }
        )
        graph_builder.add_edge("test_designer", "test_executor")
        graph_builder.add_conditional_edges(
            "test_executor",
            self._route_by_test_results,
            {
                "programmer": "programmer",
                "__end__": END
            }
        )

        return graph_builder.compile()

    def _programmer_agent(self, state: AgentState) -> Dict[str, Any]:
        """Programmer agent implementation."""
        response = self.programmer_llm.invoke(state['programmer_messages'])
        solution = self.code_processor.extract_code_block_from_response(
            str(response.content))
        return {
            "solution": solution,
            "programmer_messages": [response]
        }

    def _test_designer_agent(self, state: AgentState) -> Dict[str, Any]:
        """Test designer agent implementation."""
        response = self.tester_llm.invoke(state['tester_messages'])
        tests = self.code_processor.extract_code_block_from_response(
            str(response.content))
        return {
            "tests": tests,
            "tester_messages": [response]
        }

    def _test_executor_agent(self, state: AgentState) -> Dict[str, Any]:
        """Test executor agent implementation."""
        script = self.code_processor.build_test_script(
            state['solution'], state['tests'])
        test_results = self.sandbox.execute_test_script(script)

        # Construct feedback if tests fail
        if test_results.tests_failed > 0 or test_results.tests_errors > 0:
            feedback = self.prompt_provider.generate_failure_feedback(
                state, test_results)
            return {
                "test_results": test_results,
                "max_iterations": state['max_iterations'] - 1,
                "programmer_messages": [HumanMessage(content=feedback)],
                "tests_sent_to_programmer": True
            }

        return {
            "test_results": test_results,
            "max_iterations": state['max_iterations'] - 1,
        }

    def _route_after_programming(self, state: AgentState) -> Literal["test_designer", "test_executor"]:
        """Route after programming phase."""
        if not state.get('tests') or state.get('tests') == "":
            return "test_designer"
        return "test_executor"

    def _route_by_test_results(self, state: AgentState) -> Literal["programmer", "__end__"]:
        """Route based on test results."""
        if state.get('max_iterations', 0) <= 0:
            return "__end__"

        test_results = state.get('test_results')
        if test_results and (test_results.tests_failed > 0 or test_results.tests_errors > 0):
            return "programmer"
        return "__end__"

    def create_initial_state(self, problem: str) -> AgentState:
        """Create initial state for a problem."""
        function_name = self.code_processor.extract_function_name_from_problem(
            problem)

        return {
            "solution": "",
            "tests": "",
            "test_results": None,
            "max_iterations": self.config.max_iterations,
            "tests_sent_to_programmer": False,
            "solution_sent_to_tester": False,
            "programmer_messages": [
                SystemMessage(
                    content=self.prompt_provider.programmer_system_prompt),
                HumanMessage(
                    content=self.prompt_provider.get_programmer_prompt(problem))
            ],
            "tester_messages": [
                SystemMessage(
                    content=self.prompt_provider.test_designer_system_prompt),
                HumanMessage(content=self.prompt_provider.get_test_designer_prompt(
                    problem, function_name))
            ]
        }


class CompletionGenerator:
    """Handles the generation of completions from workflow results."""

    def __init__(self, workflow: Any, code_processor: Any, iteration_tracker: IterationTracker) -> None:
        self.workflow = workflow
        self.code_processor = code_processor
        self.iteration_tracker = iteration_tracker

    def generate_completion_with_iterations(self, problem: str, task_id: str, initial_state: AgentState) -> Tuple[str, bool, List[IterationResult]]:
        """Generate a completion for a single problem with per-iteration tracking."""
        try:
            iteration_results = []
            current_solution = ""
            current_test_results = None
            current_test_cases = None
            iteration = 0

            # Stream the workflow with updates mode
            for update in self.workflow.stream(initial_state, stream_mode="updates"):
                # Process programmer updates: Start of iteration
                if 'programmer' in update:
                    current_solution = update['programmer']['solution']
                    iteration += 1

                # Process test_designer updates: Capture test cases
                elif 'test_designer' in update:
                    current_test_cases = update['test_designer']['tests']

                # Process test_executor updates: End of iteration
                elif 'test_executor' in update:
                    current_test_results = update['test_executor']['test_results']

                    # Extract completion for this iteration
                    if current_solution:
                        completion = self._extract_completion(
                            problem, current_solution)
                        success = self._check_success(current_test_results)

                        # Create and save iteration result
                        iteration_result = self.iteration_tracker.create_iteration_result(
                            task_id, iteration, completion, success, current_test_results, current_test_cases)

                        iteration_results.append(iteration_result)
                        self.iteration_tracker.save_iteration_result(
                            iteration_result)

            # Get final results
            final_completion = ""
            final_success = False

            if iteration_results:
                last_result = iteration_results[-1]
                final_completion = last_result.completion
                final_success = last_result.success

            return final_completion, final_success, iteration_results

        except Exception as e:
            print(f"Error generating completion: {e}")
            return "", False, []

    def _extract_completion(self, problem: str, solution: str) -> str:
        """Extract completion from solution."""
        function_name = self.code_processor.extract_function_name_from_problem(
            problem)
        if function_name:
            result = self.code_processor.extract_function_with_helpers(
                self.code_processor.remove_comments(solution),
                function_name
            )
            return str(result) if result is not None else ""
        return ""

    def _check_success(self, test_results: Optional[TestExecutionResult]) -> bool:
        """Check if tests passed."""
        return (test_results is not None and test_results.has_failures is False)


class AgentCoderGenerator:
    """
    Refactored AgentCoder generator with proper separation of concerns.

    This class now focuses on orchestration and delegates specific responsibilities
    to specialized components.
    """

    def __init__(self, config: AgentCoderConfig) -> None:
        """Initialize the AgentCoder generator with configuration."""
        self.config = config

        # Initialize components
        self.code_processor = CodeProcessor()
        self.sandbox = create_safe_test_environment()

        # These should not be None after __post_init__ sets defaults
        assert config.programmer_llm_provider is not None
        assert config.programmer_llm_model is not None
        assert config.tester_llm_provider is not None
        assert config.tester_llm_model is not None

        self.programmer_llm = create_llm_from_params(
            config.programmer_llm_provider,
            config.programmer_llm_model
        )
        self.tester_llm = create_llm_from_params(
            config.tester_llm_provider,
            config.tester_llm_model
        )

        # Initialize specialized components
        self.file_manager = FileManager(config)
        self.iteration_tracker = IterationTracker(config, self.file_manager)
        self.workflow_builder = WorkflowBuilder(
            config, self.programmer_llm, self.tester_llm, self.code_processor, self.sandbox)

        # Build workflow
        self.workflow = self.workflow_builder.build_workflow()
        self.completion_generator = CompletionGenerator(
            self.workflow, self.code_processor, self.iteration_tracker)

    def generate_completion_with_iterations(self, problem: str, task_id: str) -> Tuple[str, bool, List[IterationResult]]:
        """Generate a completion for a single problem with per-iteration tracking."""
        initial_state = self.workflow_builder.create_initial_state(problem)
        return self.completion_generator.generate_completion_with_iterations(problem, task_id, initial_state)

    def generate_completion(self, problem: str) -> Tuple[str, bool]:
        """Generate a completion for a single problem (legacy method)."""
        completion, success, _ = self.generate_completion_with_iterations(
            problem, "")
        return completion, success

    def process_evaluation_dataset(self) -> None:
        """Process HumanEval problems (full dataset or subset) and generate completions with per-iteration tracking."""
        # Load problems using FileManager (proper separation of concerns)
        problems, dataset_info = self.file_manager.load_evaluation_problems()

        print(
            f"📊 Processing HumanEval {dataset_info} with {len(problems)} problems")

        # Clear output files
        self.file_manager.clear_output_files()

        # Process problems
        total_samples = len(problems) * self.config.num_samples_per_task
        successful_completions = 0
        processed_samples = 0

        with tqdm(problems.items(), desc="Processing problems", unit="problem") as pbar:
            for task_id, problem_data in pbar:
                pbar.set_description(f"Processing {task_id}")

                for sample_idx in range(self.config.num_samples_per_task):
                    completion, success, _ = self.generate_completion_with_iterations(
                        problem_data["prompt"], task_id)

                    processed_samples += 1
                    if success:
                        successful_completions += 1

                    # Save final completion
                    self.file_manager.save_final_completion(
                        task_id, completion)

                    # Update progress bar
                    pbar.set_postfix({
                        "successful": f"{successful_completions}/{processed_samples}",
                        "rate": f"{successful_completions/max(1, processed_samples)*100:.1f}%"
                    })

        # Print summary
        self._print_summary(len(problems), total_samples,
                            successful_completions)

    def process_humaneval_subset(self, subset_size: int = 20) -> None:
        """Convenience method to process the HumanEval subset with specified size.

        Args:
            subset_size: Size of the subset to process (default: 20)
        """
        # Temporarily override config to use subset
        original_use_subset = self.config.use_humaneval_subset
        original_subset_path = self.config.humaneval_subset_path

        self.config.use_humaneval_subset = True
        # Set subset path if not specified
        if not self.config.humaneval_subset_path:
            project_root = self.config.get_project_root()
            self.config.humaneval_subset_path = str(
                project_root / "data" / "human_eval" /
                f"HumanEval_{subset_size}.jsonl.gz"
            )

        try:
            self.process_evaluation_dataset()
        finally:
            # Restore original config
            self.config.use_humaneval_subset = original_use_subset
            self.config.humaneval_subset_path = original_subset_path

    def _print_summary(self, num_problems: int, total_samples: int, successful_completions: int) -> None:
        """Print processing summary."""
        print(
            f"\nCompleted processing {num_problems} problems ({total_samples} samples)")
        print(
            f"Successful completions: {successful_completions}/{total_samples} ({successful_completions/total_samples*100:.1f}%)")
        print(
            f"Final results saved to: {self.file_manager.get_base_output_path()}")

        if self.config.save_per_iteration:
            print("Per-iteration results saved to:")
            for iter_num in range(1, self.config.max_iterations + 1):
                iteration_file = self.file_manager.get_iteration_output_path(
                    iter_num)
                if Path(iteration_file).exists():
                    with open(iteration_file, 'r') as f:
                        count = sum(1 for _ in f)
                    print(
                        f"  Iteration {iter_num}: {iteration_file} ({count} entries)")


def main() -> None:
    """Main entry point."""
    config = AgentCoderConfig(
        llm_model='qwen2.5-coder:7b',  # Default fallback
        llm_provider='ollama',
        max_iterations=3,
        num_samples_per_task=1,
        save_per_iteration=True,
        use_humaneval_subset=True,  # Enable subset processing for faster development

        # Separate LLMs for each agent
        programmer_llm_provider='ollama',
        programmer_llm_model='qwen2.5-coder:7b',
        tester_llm_provider='openai',
        tester_llm_model='gpt-5',
    )

    generator = AgentCoderGenerator(config)

    # Process the evaluation dataset (subset or full based on config)
    generator.process_evaluation_dataset()

    # Alternative: Use the convenience method for explicit subset processing
    # generator.process_humaneval_subset(subset_size=20)


if __name__ == "__main__":
    main()
