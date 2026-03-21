import io
import logging
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tenacity
import typer
import yaml
from loguru import logger
from rich.console import Console, Group
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.syntax import Syntax
from rich.table import Table

from coevolution.dataset.lcb import Difficulty, load_code_generation_dataset
from infrastructure.languages.python.adapter import PythonLanguage
from infrastructure.llm_client.factory import create_llm_client

app = typer.Typer()
console = Console()

# Silence noisy loggers and redirect to rich
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
for logger_name in ["httpx", "openai", "urllib3"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Silence loguru
logger.remove()
logger.add(RichHandler(console=console), level="WARNING")

PLAN_PROMPT_TEMPLATE = """Problem:
{question_content}

Starter Code:
{starter_code}

Please provide a comprehensive step-by-step plan to solve the problem. 
Your plan should be detailed and logic-oriented, mapping out how to handle edge cases and the core algorithmic steps.
"""

SOLUTION_PROMPT_TEMPLATE = """Problem:
{question_content}

Implementation Plan:
{plan}

Starter Code:
{starter_code}

Based on the problem description and the implementation plan above, please provide a complete Python solution.
Wrap your code in ```python blocks. Strictly stick to the Starter Code format. 
"""


class PlanningLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        # Initialize the file
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"Run started at {datetime.now()}\n")

    def log_problem_block(self, block: str) -> None:
        with self.lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(block + "\n")
                f.write("=" * 120 + "\n")


class JsonlLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def log(self, data: Dict[str, Any]) -> None:
        import json

        with self.lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")


def process_problem(
    problem: Any,
    llm_client: Any,
    python_lang: PythonLanguage,
    problem_idx: int,
    total_problems: int,
    progress: Progress,
    task_id: Any,
) -> Tuple[str, bool, str, str, str]:
    """Process a single problem and return results and log buffer."""

    log_stream = io.StringIO()
    p_console = Console(file=log_stream, force_terminal=True, width=120)

    problem_passed = True

    progress.update(
        task_id, description=f"Processing: {problem.question_title[:40]}..."
    )

    p_console.rule(
        f"[bold magenta]Problem {problem_idx + 1}/{total_problems}: {problem.question_title} ({problem.question_id})[/bold magenta]"
    )
    p_console.print(
        Panel(problem.question_content, title="Question Content", border_style="blue")
    )

    # Step 1: Generate Plan
    p_console.print("\n[bold cyan]Generating Implementation Plan...[/bold cyan]")
    plan_prompt = PLAN_PROMPT_TEMPLATE.format(
        question_content=problem.question_content,
        starter_code=problem.starter_code,
    )

    try:

        @tenacity.retry(
            stop=tenacity.stop_after_attempt(5),
            wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
            retry=tenacity.retry_if_exception_type(Exception),
        )
        def generate_plan() -> str:
            return str(llm_client.generate(plan_prompt))

        plan_response = generate_plan()
        p_console.print(
            Panel(
                Syntax(plan_response, "markdown", theme="monokai", padding=1),
                title="Implementation Plan",
            )
        )
    except Exception as e:
        p_console.print(f"[bold red]Failed to generate plan: {e}[/bold red]")
        plan_response = "Failed to generate plan."
        problem_passed = False

    # Step 2: Generate solution
    generated_code = ""
    if problem_passed:
        for sol_attempt in range(3):
            p_console.print(
                f"\n[bold cyan]Generating Python solution (Attempt {sol_attempt + 1}/3)...[/bold cyan]"
            )
            solution_prompt = SOLUTION_PROMPT_TEMPLATE.format(
                question_content=problem.question_content,
                plan=plan_response,
                starter_code=problem.starter_code,
            )

            try:

                @tenacity.retry(
                    stop=tenacity.stop_after_attempt(5),
                    wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
                    retry=tenacity.retry_if_exception_type(Exception),
                )
                def generate_solution() -> str:
                    return str(llm_client.generate(solution_prompt))

                solution_response = generate_solution()

                # Extract code from Markdown blocks
                code_match = re.search(
                    r"```python\n(.*?)\n```", solution_response, re.DOTALL
                )
                if code_match:
                    extracted_code = code_match.group(1).strip()
                else:
                    extracted_code = solution_response.strip()

                # Validate syntax
                if python_lang.parser.is_syntax_valid(extracted_code):
                    generated_code = extracted_code
                    p_console.print(
                        Panel(
                            Syntax(
                                solution_response,
                                "markdown",
                                theme="monokai",
                                padding=1,
                            ),
                            title="LLM Solution Response (Valid Syntax)",
                        )
                    )
                    break
                else:
                    p_console.print(
                        f"[bold red]Attempt {sol_attempt + 1}/3: Generated code has invalid syntax.[/bold red]"
                    )
                    if sol_attempt == 2:
                        problem_passed = False
                        generated_code = extracted_code
            except Exception as e:
                p_console.print(
                    f"[bold red]Failed to generate solution: {e}[bold red]"
                )
                if sol_attempt == 2:
                    problem_passed = False
                continue
    else:
        generated_code = ""

    status_str = (
        "[bold green]GENERATED[/bold green]"
        if problem_passed
        else "[bold red]FAILED[/bold red]"
    )
    p_console.print(f"\nFinal Problem Status: {status_str}")

    progress.advance(task_id)
    return (
        problem.question_id,
        problem_passed,
        log_stream.getvalue(),
        generated_code,
        plan_response,
    )


@app.command()
def run(
    llm: Path = typer.Option(
        Path("configs/llm/gpt-5-mini.yaml"), help="Path to LLM config YAML"
    ),
    count: Optional[int] = typer.Option(None, help="Number of problems to process"),
    difficulty: str = typer.Option(
        "hard", help="Difficulty of problems to load (easy, medium, hard)"
    ),
    version: str = typer.Option("release_v6", help="LCB dataset version"),
    start_date: Optional[str] = typer.Option(
        "2025-03-01", help="Start date (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = typer.Option("2025-05-10", help="End date (YYYY-MM-DD)"),
    output_dir: Path = typer.Option(
        Path("logs/planning"), help="Directory to save logs"
    ),
    workers: int = typer.Option(16, help="Number of parallel workers"),
) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    log_file = output_dir / f"{run_id}_planning.txt"
    jsonl_file = output_dir / f"{run_id}_solutions.jsonl"
    logger = PlanningLogger(log_file)
    jsonl_logger = JsonlLogger(jsonl_file)

    console.print(
        Panel(
            f"Starting Two-Step Planning Run: [bold cyan]{run_id}[/bold cyan]\n"
            f"Log file: [yellow]{log_file}[/yellow]\n"
            f"JSONL file: [yellow]{jsonl_file}[/yellow]\n"
            f"Workers: [green]{workers}[/green]"
        )
    )

    # Load LLM Config
    with open(llm, "r") as f:
        llm_cfg = yaml.safe_load(f)

    llm_client = create_llm_client(**llm_cfg)
    python_lang = PythonLanguage()

    # Load Dataset
    diff_enum = Difficulty(difficulty.lower())
    problems = load_code_generation_dataset(
        release_version=version,
        difficulty=diff_enum,
        start_date=start_date,
        end_date=end_date,
    )

    if count:
        problems = problems[:count]

    total_problems = len(problems)
    results = []
    generated_count = 0
    solved_problems = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task(
            "[cyan]Processing problems...", total=total_problems
        )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_problem = {
                executor.submit(
                    process_problem,
                    problem,
                    llm_client,
                    python_lang,
                    i,
                    len(problems),
                    progress,
                    overall_task,
                ): problem
                for i, problem in enumerate(problems)
            }

            for fut in future_to_problem:
                try:
                    q_id, passed, log_block, snippet, plan = fut.result()
                    results.append((q_id, passed))
                    if passed:
                        solved_problems += 1
                    logger.log_problem_block(log_block)
                    jsonl_logger.log(
                        {
                            "question_id": q_id,
                            "snippet": snippet,
                            "plan": plan,
                            "status": None,
                        }
                    )
                except Exception as e:
                    console.print(f"[bold red]Exception in worker: {e}[/bold red]")

    # Final result table
    table = Table(title="Planning Results")
    table.add_column("Question ID", style="cyan")
    table.add_column("Status", style="bold")

    for q_id, passed in results:
        status_text = "[green]Generated[/green]" if passed else "[red]Failed[/red]"
        table.add_row(q_id, status_text)

    console.print(table)

    summary_panel = Panel(
        Group(
            f"Total Problems: {total_problems}",
            f"Generated: [bold green]{generated_count}[/bold green]",
            f"Success Rate: [bold yellow]{(generated_count / total_problems) * 100:.2f}%[/bold yellow]"
            if total_problems > 0
            else "N/A",
        ),
        title="Summary",
        border_style="bold green",
    )
    console.print(summary_panel)

    # Capture final table and summary for the log file
    log_final_stream = io.StringIO()
    log_final_console = Console(file=log_final_stream, force_terminal=True, width=120)
    log_final_console.print(table)
    log_final_console.print(summary_panel)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 120 + "\n")
        f.write("FINAL SUMMARY\n")
        f.write("=" * 120 + "\n")
        f.write(log_final_stream.getvalue())
        f.write(f"\nRun completed at {datetime.now()}\n")


if __name__ == "__main__":
    app()
