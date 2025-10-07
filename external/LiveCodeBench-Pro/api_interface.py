from abc import ABC, abstractmethod
from typing import Any, Tuple

from openai import OpenAI


class LLMInterface(ABC):
    """
    Abstract base class for integrating Large Language Models (LLMs) into a competitive programming context.

    Attributes:
        prompt (str): Initial prompt to set the context for the LLM.
    """

    def __init__(self):
        """
        Initialize the LLMInterface with a predefined prompt for generating competitive programming solutions.
        """
        self.prompt = """
        You are a competitive programmer. You will be given a problem statement, please implement solution in C++. The execution time and memory limit are also stated in the statement so be aware of the complexity of the program. Please wrap the code in ```cpp and ``` so that it is properly formatted.
        """

    @abstractmethod
    def call_llm(user_prompt: str) -> Tuple[str, Any]:
        """
        Abstract method to interact with the LLM.

        Args:
            user_prompt (str): The prompt containing the problem statement for the LLM.

        Returns:
            Tuple[str, Any]: A tuple containing the generated solution and additional metadata.
        """
        pass

    def generate_solution(self, problem_statement: str) -> Tuple[str, Any]:
        """
        Generates a solution to a given competitive programming problem using the LLM.

        Args:
            problem_statement (str): The competitive programming problem statement.

        Returns:
            Tuple[str, Any]: The generated solution and associated metadata.
        """
        user_prompt = self.prompt + problem_statement
        response, meta = self.call_llm(user_prompt)
        return response, meta


class OpenAICodex(LLMInterface):
    """
    Concrete implementation of LLMInterface using OpenAI's GPT-4o model.

    Attributes:
        client (OpenAI): Client instance for interacting with the OpenAI API.
    """

    def __init__(self):
        """
        Initializes the ExampleLLM class by creating an instance of the OpenAI client.
        """
        super().__init__()
        self.client = OpenAI()

    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        """
        Sends the user prompt to OpenAI's GPT-4o model and retrieves the solution.

        Args:
            user_prompt (str): The complete prompt including the initial context and problem statement.

        Returns:
            Tuple[str, Any]: The LLM's response and metadata about the completion.
        """
        response = self.client.responses.create(
            model="gpt-5-codex",
            input=user_prompt,
        )
        return response.output[1].content[0].text, str(response)


if __name__ == "__main__":
    """
    Example execution demonstrating how to use the ExampleLLM class to generate solutions.
    """
    llm = OpenAICodex()
    response, meta = llm.generate_solution("I need to count from 1 to 10")
    print(response)
    print("\n---\n")
    print(meta)
