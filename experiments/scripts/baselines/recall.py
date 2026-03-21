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

PROMPT_RECALL = """Problem:
{question_content}

Starter Code:
{starter_code}

Please recall a similar programming problem you have seen before that shares the same underlying concepts or requires a similar algorithmic approach. 
1. Briefly state the similar problem.
2. Describe the core algorithm or logic used to solve that similar problem.

Your response should focus entirely on the *similar problem and its algorithm*, acting as a helpful hint for solving the current problem.
"""

PROMPT_GENERATE_CODE = """Original Problem:
{question_content}

Similar Problem and Algorithm (Hint):
{recalled_context}

Starter Code:
{starter_code}

Based on the original problem description and drawing inspiration from the similar problem and algorithm, please provide a complete Python solution.
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
) -> Tuple[str, bool, str, str, str]:
    """Process a single problem using a recall mechanism before generating code."""

    log_stream = io.StringIO()
    p_console = Console(file=log_stream, force_terminal=True, width=120)

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

    recalled_context = ""
    generated_code = ""
    problem_passed = False

    try:
        # Step 1: Recall Similar Problem
        p_console.print("\n[bold cyan]Step 1: Recalling Similar Problem and Algorithm...[/bold cyan]")
        recall_p = PROMPT_RECALL.format(
            question_content=problem.question_content,
            starter_code=problem.starter_code
        )
        recalled_context = generate(recall_p)
        p_console.print(Panel(Syntax(recalled_context, "markdown", theme="monokai", padding=1), title="Recalled Context"))

        # Step 2: Generate Code Using Recall
        p_console.print("\n[bold cyan]Step 2: Generating Code Using Recalled Context...[/bold cyan]")
        generate_p = PROMPT_GENERATE_CODE.format(
            question_content=problem.question_content,
            recalled_context=recalled_context,
            starter_code=problem.starter_code
        )
        res_code = generate(generate_p)
        generated_code = extract_code(res_code)
        p_console.print(Panel(Syntax(generated_code, "python", theme="monokai", padding=1), title="Generated Code"))
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
        generated_code,
        recalled_context,
    )


@app.command()
def run(
    llm: Path = typer.Option(Path("configs/llm/gpt-5-mini.yaml"), help="Path to LLM config YAML"),
    count: Optional[int] = typer.Option(None, help="Number of problems to process"),
    difficulty: str = typer.Option("hard", help="Difficulty of problems to load (easy, medium, hard)"),
    version: str = typer.Option("release_v6", help="LCB dataset version"),
    start_date: Optional[str] = typer.Option("2025-03-01", help="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = typer.Option("2025-05-10", help="End date (YYYY-MM-DD)"),
    output_dir: Path = typer.Option(Path("logs/recall"), help="Directory to save logs"),
    workers: int = typer.Option(8, help="Number of parallel workers"),
) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    log_file = output_dir / f"{run_id}_recall.txt"
    jsonl_file = output_dir / f"{run_id}_solutions.jsonl"
    logger_txt = DirectLogger(log_file)
    jsonl_logger = JsonlLogger(jsonl_file)

    console.print(
        Panel(
            f"Starting Recall Code Generation Run: [bold cyan]{run_id}[/bold cyan]\n"
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
    solved_problems = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task("[cyan]Processing recall workflow...", total=total_problems)

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
                    q_id, passed, log_block, snippet, recalled_ctx = fut.result()
                    results.append((q_id, passed))
                    if passed:
                        solved_problems += 1
                    logger_txt.log_problem_block(log_block)
                    jsonl_logger.log(
                        {
                            "question_id": q_id,
                            "snippet": snippet,
                            "recalled_context": recalled_ctx,
                            "status": None,
                        }
                    )
                except Exception as e:
                    console.print(f"[bold red]Exception in worker: {e}[/bold red]")

    # Final result table
    table = Table(title="Recall Generation Results")
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
