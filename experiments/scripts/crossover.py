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

INITIAL_PROMPT_TEMPLATE = """Problem:
{question_content}

Starter Code:
{starter_code}

Based on the problem description above, please provide a complete Python solution.
Wrap your code in ```python blocks. Strictly stick to the Starter Code format. 
"""

CROSSOVER_PROMPT_TEMPLATE = """Problem:
{question_content}

Starter Code:
{starter_code}

Candidate Solution 1:
```python
{parent1}
```

Candidate Solution 2:
```python
{parent2}
```

Create a new complete Python solution that intelligently combines the best aspects of the two candidate solutions provided above.
Ensure the combined solution:
1. Addresses the problem correctly.
2. Is efficient and idiomatic.
3. Strictly adheres to the Starter Code format.

Wrap your code in ```python blocks.
"""


class DirectLogger:
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
    iterations: int,
) -> Tuple[str, bool, str, List[str]]:
    """Process a single problem with iterative crossover."""

    log_stream = io.StringIO()
    p_console = Console(file=log_stream, force_terminal=True, width=120)

    problem_passed = True
    snippets = []

    progress.update(
        task_id, description=f"Processing: {problem.question_title[:40]}..."
    )

    p_console.rule(
        f"[bold magenta]Problem {problem_idx + 1}/{total_problems}: {problem.question_title} ({problem.question_id})[/bold magenta]"
    )
    p_console.print(
        Panel(problem.question_content, title="Question Content", border_style="blue")
    )

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
    )
    def generate_solution(p: str) -> str:
        return str(llm_client.generate(p))

    def extract_code(response: str) -> str:
        code_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
        return code_match.group(1).strip() if code_match else response.strip()

    # Round 1: Initial Generations
    p_console.print(f"\n[bold cyan]Round 1/{iterations + 1}: Initial Generations...[/bold cyan]")
    prompt = INITIAL_PROMPT_TEMPLATE.format(
        question_content=problem.question_content,
        starter_code=problem.starter_code,
    )

    try:
        # Generate v1
        res1 = generate_solution(prompt)
        v1 = extract_code(res1)
        snippets.append(v1)
        p_console.print(Panel(Syntax(v1, "python", theme="monokai", padding=1), title="Version 1"))

        # Generate v2
        res2 = generate_solution(prompt)
        v2 = extract_code(res2)
        snippets.append(v2)
        p_console.print(Panel(Syntax(v2, "python", theme="monokai", padding=1), title="Version 2"))
    except Exception as e:
        p_console.print(f"[bold red]Failed initial generations: {e}[/bold red]")
        problem_passed = False
        return (problem.question_id, False, log_stream.getvalue(), snippets)

    current_parents = [v1, v2]

    # Round 2 to iterations+1: Crossover Loop
    for r in range(iterations):
        round_num = r + 2
        p_console.print(f"\n[bold cyan]Round {round_num}/{iterations + 1}: Crossover Iteration {r + 1}/{iterations}...[/bold cyan]")
        
        parent1, parent2 = current_parents
        crossover_prompt = CROSSOVER_PROMPT_TEMPLATE.format(
            question_content=problem.question_content,
            starter_code=problem.starter_code,
            parent1=parent1,
            parent2=parent2,
        )

        try:
            # Generate two children from the same parents
            res_a = generate_solution(crossover_prompt)
            v_a = extract_code(res_a)
            snippets.append(v_a)
            p_console.print(Panel(Syntax(v_a, "python", theme="monokai", padding=1), title=f"Version {len(snippets)}"))

            res_b = generate_solution(crossover_prompt)
            v_b = extract_code(res_b)
            snippets.append(v_b)
            p_console.print(Panel(Syntax(v_b, "python", theme="monokai", padding=1), title=f"Version {len(snippets)}"))

            current_parents = [v_a, v_b]
        except Exception as e:
            p_console.print(f"[bold red]Failed crossover in round {round_num}: {e}[/bold red]")
            break

    status_str = (
        f"[bold green]GENERATED ({len(snippets)} versions)[/bold green]"
        if problem_passed
        else "[bold red]FAILED[/bold red]"
    )
    p_console.print(f"\nFinal Problem Status: {status_str}")

    progress.advance(task_id)
    return (
        problem.question_id,
        problem_passed,
        log_stream.getvalue(),
        snippets,
    )


@app.command()
def run(
    llm: Path = typer.Option(Path("configs/llm/gpt-5-mini.yaml"), help="Path to LLM config YAML"),
    count: Optional[int] = typer.Option(None, help="Number of problems to process"),
    difficulty: str = typer.Option("hard", help="Difficulty of problems to load (easy, medium, hard)"),
    version: str = typer.Option("release_v6", help="LCB dataset version"),
    start_date: Optional[str] = typer.Option("2025-03-01", help="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = typer.Option("2025-05-10", help="End date (YYYY-MM-DD)"),
    output_dir: Path = typer.Option(Path("logs/crossover"), help="Directory to save logs"),
    workers: int = typer.Option(8, help="Number of parallel workers"),
    iterations: int = typer.Option(2, help="Number of crossover iterations"),
) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    log_file = output_dir / f"{run_id}_crossover.txt"
    jsonl_file = output_dir / f"{run_id}_solutions.jsonl"
    logger_txt = DirectLogger(log_file)
    jsonl_logger = JsonlLogger(jsonl_file)

    console.print(
        Panel(
            f"Starting Crossover Code Generation Run: [bold cyan]{run_id}[/bold cyan]\n"
            f"Log file: [yellow]{log_file}[/yellow]\n"
            f"JSONL file: [yellow]{jsonl_file}[/yellow]\n"
            f"Workers: [green]{workers}[/green]\n"
            f"Iterations: [green]{iterations}[/green]"
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
        overall_task = progress.add_task("[cyan]Processing problems with crossover...", total=total_problems)

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
                    iterations,
                ): problem
                for i, problem in enumerate(problems)
            }

            for fut in future_to_problem:
                try:
                    q_id, passed, log_block, snippets = fut.result()
                    results.append((q_id, passed, len(snippets)))
                    if passed:
                        solved_problems += 1
                    logger_txt.log_problem_block(log_block)
                    jsonl_logger.log(
                        {
                            "question_id": q_id,
                            "snippet": snippets[-1] if snippets else "",
                            "all_snippets": snippets,
                            "status": None,
                        }
                    )
                except Exception as e:
                    console.print(f"[bold red]Exception in worker: {e}[/bold red]")

    # Final result table
    table = Table(title="Crossover Generation Results")
    table.add_column("Question ID", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Versions", style="magenta")

    for q_id, passed, versions in results:
        status_text = "[green]Generated[/green]" if passed else "[red]Failed[/red]"
        table.add_row(q_id, status_text, str(versions))

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
