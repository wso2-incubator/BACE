from dataclasses import dataclass
from datetime import datetime
from typing import Any

from datasets import load_dataset


@dataclass
class CodeExecutionProblem:
    question_id: str
    contest_id: str
    contest_date: datetime
    difficulty: str
    function_name: str
    code: str
    input: str
    output: str
    id: str
    problem_id: str
    numsteps: int

    def __post_init__(self) -> None:
        pass

    def insert_output(
        self, output_list: list[str], pred_list: list[str]
    ) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "difficulty": self.difficulty,
            "function_name": self.function_name,
            "code": self.code,
            "input": self.input,
            "output": self.output,
            "id": self.id,
            "problem_id": self.problem_id,
            "numsteps": self.numsteps,
            "output_list": output_list,
            "pred_list": pred_list,
        }

    def insert_output_evaluation(
        self, output_list: list[str], code_list: list[str], graded_list: list[bool]
    ) -> dict[str, Any]:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        return output

    def get_evaluation_sample(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "input": self.input,
            "output": self.output,
        }


def load_code_execution_dataset(
    release_version: str = "release_v1",
) -> list[CodeExecutionProblem]:
    dataset = [
        CodeExecutionProblem(**p)
        for p in load_dataset("livecodebench/execution-v2", split="test")
    ]
    print(f"Loaded {len(dataset)} problems")
    return dataset


if __name__ == "__main__":
    dataset = load_code_execution_dataset()
