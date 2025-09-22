"""
AgentCoder Generator for HumanEval Dataset

This module provides a professional, modular implementation of the AgentCoder
multi-agent code generation system that can process all HumanEval problems.
"""

from human_eval.data import write_jsonl, read_problems
import common
from typing import List, TypedDict, Dict, Annotated, Literal
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentCoderConfig:
    """Configuration for the AgentCoder system."""
    llm_model: str = "qwen2.5-coder:7b"
    max_iterations: int = 5
    timeout: int = 30
    output_file: Optional[str] = None
    num_samples_per_task: int = 1


class AgentState(TypedDict):
    """State definition for the multi-agent workflow."""
    solution: str
    tests: str
    test_results: Optional[common.TestExecutionResult]
    max_iterations: int
    tests_sent_to_programmer: bool
    solution_sent_to_tester: bool
    programmer_messages: Annotated[List[BaseMessage], add_messages]
    tester_messages: Annotated[List[BaseMessage], add_messages]


class AgentCoderGenerator:
    """
    Professional AgentCoder generator for HumanEval problems.

    This class encapsulates the AgentCoder multi-agent workflow with proper separation
    of concerns and type safety.
    """

    def __init__(self, config: AgentCoderConfig) -> None:
        """Initialize the AgentCoder generator with configuration."""
        self.config = config
        self.code_processor = common.CodeProcessor()
        self.sandbox = common.create_safe_test_environment()
        self.llm = ChatOllama(model=config.llm_model)
        self.workflow = self._build_workflow()

        # System prompts
        self.programmer_system_prompt = "You are an expert software developer."
        self.test_designer_system_prompt = "You are an expert program tester."

    def _build_workflow(self) -> Any:
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

    def _get_programmer_prompt(self, problem: str) -> str:
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

    def _get_test_designer_prompt(self, problem: str, function_name: str) -> str:
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

    def _programmer_agent(self, state: AgentState) -> Dict[str, Any]:
        """Programmer agent implementation."""
        response = self.llm.invoke(state['programmer_messages'])
        solution = self.code_processor.extract_code_block_from_response(
            str(response.content))

        return {
            "solution": solution,
            "programmer_messages": [response]
        }

    def _test_designer_agent(self, state: AgentState) -> Dict[str, Any]:
        """Test designer agent implementation."""
        response = self.llm.invoke(state['tester_messages'])
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
            feedback = self._generate_failure_feedback(state, test_results)
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

    def _generate_failure_feedback(self, state: AgentState, test_results: common.TestExecutionResult) -> str:
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

    def _create_initial_state(self, problem: str) -> AgentState:
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
                SystemMessage(content=self.programmer_system_prompt),
                HumanMessage(content=self._get_programmer_prompt(problem))
            ],
            "tester_messages": [
                SystemMessage(content=self.test_designer_system_prompt),
                HumanMessage(content=self._get_test_designer_prompt(
                    problem, function_name))
            ]
        }

    def generate_completion(self, problem: str) -> Tuple[str, bool]:
        """
        Generate a completion for a single problem.

        Args:
            problem: The problem prompt

        Returns:
            Tuple of (completion_code, success)
        """
        try:
            initial_state = self._create_initial_state(problem)
            result = self.workflow.invoke(initial_state)

            # Extract the function implementation only
            solution = result.get('solution', '')
            if solution:
                function_name = self.code_processor.extract_function_name_from_problem(
                    problem)
                if function_name:
                    # Extract just the function body for completion
                    completion = self.code_processor.extract_function_with_helpers(
                        self.code_processor.remove_comments(solution),
                        function_name
                    )

                    # Check if tests passed
                    test_results = result.get('test_results')
                    success = (test_results is not None and
                               test_results.tests_failed == 0 and
                               test_results.tests_errors == 0)

                    return completion, success

            return "", False

        except Exception as e:
            print(f"Error generating completion: {e}")
            return "", False

    def process_all_problems(self) -> None:
        """Process all HumanEval problems and generate completions."""
        problems = read_problems()

        # Create output file path with proper directory structure and naming
        if self.config.output_file:
            output_file = self.config.output_file
        else:
            # Create data/generations/agent_coder directory
            output_dir = Path(__file__).parent.parent.parent / \
                "data" / "generations" / "agent_coder"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with specified format (sanitize model name for filesystem)
            safe_model_name = self.config.llm_model.replace(":", "_")
            filename = f"{safe_model_name}-max{self.config.max_iterations}-numSamples{self.config.num_samples_per_task}.jsonl"
            output_file = str(output_dir / filename)

        # Clear output file
        Path(output_file).unlink(missing_ok=True)

        total_problems = len(problems)
        successful_completions = 0

        with tqdm(problems.items(), desc="Processing problems", unit="problem") as pbar:
            for task_id, problem_data in pbar:
                pbar.set_description(f"Processing {task_id}")

                for sample_idx in range(self.config.num_samples_per_task):
                    completion, success = self.generate_completion(
                        problem_data["prompt"])

                    if success:
                        successful_completions += 1

                    # Write completion
                    completion_data = {
                        "task_id": task_id,
                        "completion": completion,
                        "success": success
                    }

                    write_jsonl(output_file, [completion_data], append=True)

                pbar.set_postfix({
                    "successful": f"{successful_completions}/{total_problems}",
                    "rate": f"{successful_completions/max(1, pbar.n)*100:.1f}%"
                })

        print(f"\nCompleted processing {total_problems} problems")
        print(
            f"Successful completions: {successful_completions}/{total_problems} ({successful_completions/total_problems*100:.1f}%)")
        print(f"Results saved to: {output_file}")


def main() -> None:
    """Main entry point."""

    config = AgentCoderConfig(
        llm_model='qwen2.5-coder:7b',
        max_iterations=3,  # Reduced for efficiency
        num_samples_per_task=1,
    )

    generator = AgentCoderGenerator(config)
    generator.process_all_problems()


if __name__ == "__main__":
    main()
