import io
import json
import logging
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# --- Prompt Templates ---

PROMPT_IO_INSTRUCTIONS = """Problem Description:
{question_content}

Starter Code:
{starter_code}

Based on the problem description and the provided starter code, write explicit and detailed instructions on how the input data should be parsed and how the output data should be formatted. 
Focus only on the I/O mechanics required by the starter code structure.
"""

INITIAL_PLAN_PROMPT_TEMPLATE = """Problem Description:
{question_content}

Starter Code:
{starter_code}

I/O Parsing and Formatting Instructions:
{io_instructions}

Please provide a detailed step-by-step implementation plan for the problem described above, ensuring that you strictly follow the I/O Parsing and Formatting Instructions provided.
Your plan should be logic-oriented and map out how to handle edge cases and the core algorithmic steps.
"""

PLAN_REFINEMENT_PROMPT_TEMPLATE = """Problem Description:
{question_content}

Starter Code:
{starter_code}

I/O Parsing and Formatting Instructions:
{io_instructions}

Current Implementation Plan:
{previous_plan}

Based on the problem description and the current implementation plan above, please provide a refined and improved implementation plan.
Ensure the plan still rigidly adheres to the I/O Parsing and Formatting Instructions.
Focus on:
1. Improving the logic for clarity and efficiency.
2. Adding more detail to algorithmic steps.
3. Ensuring all edge cases mentioned in the problem description are addressed.
"""

SOLUTION_PROMPT_TEMPLATE = """Problem Description:
{question_content}

I/O Parsing and Formatting Instructions:
{io_instructions}

Implementation Plan:
{plan}

Starter Code:
{starter_code}

Based on the problem description and the implementation plan above, please provide a complete Python solution.
Pay close attention to parsing inputs and formatting outputs exactly as dictated by the I/O Parsing and Formatting Instructions.
Wrap your code in ```python blocks. Strictly stick to the Starter Code format. 
"""

CODE_REFINEMENT_PROMPT_TEMPLATE = """Problem Description:
{question_content}

I/O Parsing and Formatting Instructions:
{io_instructions}

Implementation Plan:
{plan}

Starter Code:
{starter_code}

Current Code Solution:
```python
{previous_code}
```

Based on the rules described above, the plan, and the current code solution, please provide a refined and improved complete Python solution.
Focus on:
1. Ensuring the code perfectly matches the implementation plan and I/O Parsing and Formatting Instructions.
2. Fixing any potential bugs or edge cases.
3. Enhancing code clarity and style.

Wrap your code in ```python blocks. Strictly stick to the Starter Code format.
"""


class DirectLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
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
    plan_iterations: int,
) -> Tuple[str, bool, str, str, str, List[str], List[str]]:
    """Process a single problem with explicit I/O instruction generation and iterative plan/code refinement."""

    log_stream = io.StringIO()
    p_console = Console(file=log_stream, force_terminal=True, width=120)

    plans = []
    snippets = []

    progress.update(
        task_id, description=f"Processing: {problem.question_title[:40]}..."
    )

    p_console.rule(
        f"[bold magenta]Problem {problem_idx + 1}/{total_problems}: {problem.question_title} ({problem.question_id})[/bold magenta]"
    )
    p_console.print(
        Panel(problem.question_content, title="Original Question Content", border_style="blue")
    )

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
    )
    def generate(p: str) -> str:
        return str(llm_client.generate(p))

    def extract_code(response: str) -> str:
        code_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
        return code_match.group(1).strip() if code_match else response.strip()

    io_instructions = ""
    problem_passed = False

    try:
        # Step 1: Generate I/O Instructions
        p_console.print("\n[bold cyan]Step 1: Generating I/O Parsing and Formatting Instructions...[/bold cyan]")
        io_p = PROMPT_IO_INSTRUCTIONS.format(
            question_content=problem.question_content,
            starter_code=problem.starter_code
        )
        io_instructions = generate(io_p)
        p_console.print(Panel(Syntax(io_instructions, "markdown", theme="monokai", padding=1), title="I/O Instructions"))

        # Step 2: Initial Plan
        p_console.print("\n[bold cyan]Step 2: Generating Initial Plan...[/bold cyan]")
        initial_plan_p = INITIAL_PLAN_PROMPT_TEMPLATE.format(
            question_content=problem.question_content,
            starter_code=problem.starter_code,
            io_instructions=io_instructions
        )
        current_plan = generate(initial_plan_p)
        plans.append(current_plan)
        p_console.print(Panel(Syntax(current_plan, "markdown", theme="monokai", padding=1), title="Plan v1"))

        # Step 3: Plan Refinement
        for i in range(plan_iterations):
            p_console.print(f"\n[bold cyan]Step 3.{i+1}: Refining Plan...[/bold cyan]")
            refine_plan_p = PLAN_REFINEMENT_PROMPT_TEMPLATE.format(
                question_content=problem.question_content,
                starter_code=problem.starter_code,
                io_instructions=io_instructions,
                previous_plan=current_plan
            )
            current_plan = generate(refine_plan_p)
            plans.append(current_plan)
            p_console.print(Panel(Syntax(current_plan, "markdown", theme="monokai", padding=1), title=f"Plan v{i+2}"))

        # Step 4: Code Generation
        p_console.print("\n[bold cyan]Step 4: Generating Code from Final Plan...[/bold cyan]")
        solution_p = SOLUTION_PROMPT_TEMPLATE.format(
            question_content=problem.question_content,
            io_instructions=io_instructions,
            plan=current_plan,
            starter_code=problem.starter_code
        )
        res_code = generate(solution_p)
        current_code = extract_code(res_code)
        snippets.append(current_code)
        p_console.print(Panel(Syntax(current_code, "python", theme="monokai", padding=1), title="Code v1"))

        # Step 5: Code Refinement
        p_console.print("\n[bold cyan]Step 5: Refining Final Code...[/bold cyan]")
        code_refine_p = CODE_REFINEMENT_PROMPT_TEMPLATE.format(
            question_content=problem.question_content,
            io_instructions=io_instructions,
            plan=current_plan,
            starter_code=problem.starter_code,
            previous_code=current_code
        )
        res_refine = generate(code_refine_p)
        final_code = extract_code(res_refine)
        snippets.append(final_code)
        p_console.print(Panel(Syntax(final_code, "python", theme="monokai", padding=1), title="Code v2 (Refined)"))
        problem_passed = True

    except Exception as e:
        p_console.print(f"[bold red]Failed during processing: {e}[/bold red]")

    status_str = "[bold green]GENERATED[/bold green]" if problem_passed else "[bold red]FAILED[/bold red]"
    p_console.print(f"\nFinal Problem Status: {status_str}")

    progress.advance(task_id)
    return (
        problem.question_id,
        problem_passed,
        log_stream.getvalue(),
        snippets[-1] if snippets else "",
        io_instructions,
        snippets,
        plans,
    )


@app.command()
def run(
    llm: Path = typer.Option(Path("configs/llm/gpt-5-mini.yaml"), help="Path to LLM config YAML"),
    count: Optional[int] = typer.Option(None, help="Number of problems to process"),
    difficulty: str = typer.Option("hard", help="Difficulty of problems to load (easy, medium, hard)"),
    version: str = typer.Option("release_v6", help="LCB dataset version"),
    start_date: Optional[str] = typer.Option("2025-03-01", help="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = typer.Option("2025-05-10", help="End date (YYYY-MM-DD)"),
    output_dir: Path = typer.Option(Path("logs/io_instruct_plan_rewrite"), help="Directory to save logs"),
    workers: int = typer.Option(8, help="Number of parallel workers"),
    plan_iterations: int = typer.Option(3, help="Number of plan refinement iterations"),
) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    log_file = output_dir / f"{run_id}_io_instruct_plan_rewrite.txt"
    jsonl_file = output_dir / f"{run_id}_solutions.jsonl"
    logger_txt = DirectLogger(log_file)
    jsonl_logger = JsonlLogger(jsonl_file)

    console.print(
        Panel(
            f"Starting I/O Instruct Plan Rewrite Generation Run: [bold cyan]{run_id}[/bold cyan]\n"
            f"Log file: [yellow]{log_file}[/yellow]\n"
            f"JSONL file: [yellow]{jsonl_file}[/yellow]\n"
            f"Workers: [green]{workers}[/green]\n"
            f"Plan Iterations: [green]{plan_iterations}[/green]"
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
    solved_problems = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task("[cyan]Processing io_instruct_plan_rewrite workflow...", total=total_problems)

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
                    plan_iterations,
                ): problem
                for i, problem in enumerate(problems)
            }

            for fut in future_to_problem:
                try:
                    q_id, passed, log_block, snippet, io_instr, all_snippets, all_plans = fut.result()
                    results.append((q_id, passed))
                    if passed:
                        solved_problems += 1
                    logger_txt.log_problem_block(log_block)
                    jsonl_logger.log(
                        {
                            "question_id": q_id,
                            "snippet": snippet,
                            "io_instructions": io_instr,
                            "all_snippets": all_snippets,
                            "all_plans": all_plans,
                            "status": None,
                        }
                    )
                except Exception as e:
                    console.print(f"[bold red]Exception in worker: {e}[/bold red]")

    # Final result table
    table = Table(title="I/O Instruct Plan Rewrite Results")
    table.add_column("Question ID", style="cyan")
    table.add_column("Status", style="bold")

    for q_id, passed in results:
        status_text = "[green]Generated[/green]" if passed else "[red]Failed[/red]"
        table.add_row(q_id, status_text)

    console.print(table)

    summary_panel = Panel(
        Group(
            f"Total Problems: {total_problems}",
            f"Solved: [bold green]{solved_problems}[/bold green]",
            f"Pass Rate: [bold cyan]{(solved_problems / total_problems) * 100:.2f}%[/bold cyan]" if total_problems > 0 else "N/A",
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
