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

DIRECT_PROMPT_TEMPLATE = """Problem:
{question_content}

Starter Code:
{starter_code}

Based on the problem description above, please provide a complete Python solution.
Wrap your code in ```python blocks. Strictly stick to the Starter Code format. 
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
) -> Tuple[str, bool, str, str]:
    """Process a single problem and return results and log buffer."""

    log_stream = io.StringIO()
    p_console = Console(file=log_stream, force_terminal=True, width=120)

    problem_passed = True
    generated_code = ""

    progress.update(
        task_id, description=f"Processing: {problem.question_title[:40]}..."
    )

    p_console.rule(
        f"[bold magenta]Problem {problem_idx + 1}/{total_problems}: {problem.question_title} ({problem.question_id})[/bold magenta]"
    )
    p_console.print(
        Panel(problem.question_content, title="Question Content", border_style="blue")
    )

    # Direct Generation solution
    for sol_attempt in range(3):
        p_console.print(
            f"\n[bold cyan]Generating Python solution (Attempt {sol_attempt + 1}/3)...[/bold cyan]"
        )
        prompt = DIRECT_PROMPT_TEMPLATE.format(
            question_content=problem.question_content,
            starter_code=problem.starter_code,
        )

        try:

            @tenacity.retry(
                stop=tenacity.stop_after_attempt(5),
                wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
                retry=tenacity.retry_if_exception_type(Exception),
            )
            def generate_solution() -> str:
                return str(llm_client.generate(prompt))

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
            p_console.print(f"[bold red]Failed to generate solution: {e}[/bold red]")
            if sol_attempt == 2:
                problem_passed = False
            continue

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
    )


@app.command()
def run(
    llm: Path = typer.Option(
        Path("configs/llm/gpt-5-mini.yaml"), help="Path to LLM config YAML"
    ),
    count: Optional[int] = typer.Option(None, help="Number of problems to process"),
    difficulty: Optional[str] = typer.Option(
        None, help="Difficulty of problems to load (easy, medium, hard)"
    ),
    version: str = typer.Option("release_v6", help="LCB dataset version"),
    start_date: Optional[str] = typer.Option(
        "2025-03-01", help="Start date (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = typer.Option("2025-05-10", help="End date (YYYY-MM-DD)"),
    output_dir: Path = typer.Option(Path("logs/direct"), help="Directory to save logs"),
    workers: int = typer.Option(16, help="Number of parallel workers"),
) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    log_file = output_dir / f"{run_id}_direct.txt"
    jsonl_file = output_dir / f"{run_id}_solutions.jsonl"
    config_file = output_dir / f"{run_id}_config.txt"

    logger = DirectLogger(log_file)
    jsonl_logger = JsonlLogger(jsonl_file)

    # Initial Run Configuration Panel
    run_config_panel = Panel(
        Group(
            f"Starting Direct Code Generation Run: [bold cyan]{run_id}[/bold cyan]",
            f"Log file: [yellow]{log_file}[/yellow]",
            f"JSONL file: [yellow]{jsonl_file}[/yellow]",
            f"Config file: [yellow]{config_file}[/yellow]",
            "",
            f"Workers: [green]{workers}[/green]",
            f"Difficulty: [blue]{difficulty}[/blue]",
            f"Version: [blue]{version}[/blue]",
            f"Start Date: [blue]{start_date}[/blue]",
            f"End Date: [blue]{end_date}[/blue]",
            f"Count: [blue]{count}[/blue]",
        ),
        title="Run Configuration",
        border_style="bold cyan",
    )
    console.print(run_config_panel)

    # Load LLM Config
    with open(llm, "r") as f:
        llm_cfg = yaml.safe_load(f)

    llm_config_panel = Panel(
        Syntax(yaml.dump(llm_cfg), "yaml", theme="monokai", padding=1),
        title=f"LLM Configuration: {llm.name}",
        border_style="bold green",
    )
    console.print(llm_config_panel)

    # Save to Config File
    with open(config_file, "w", encoding="utf-8") as f:
        cfg_console = Console(file=f, force_terminal=True, width=120)
        cfg_console.print(run_config_panel)
        cfg_console.print(llm_config_panel)

    llm_client = create_llm_client(**llm_cfg)
    python_lang = PythonLanguage()

    # Load Dataset
    diff_enum = Difficulty(difficulty.lower()) if difficulty else None
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
                    q_id, passed, log_block, snippet = fut.result()
                    results.append((q_id, passed))
                    if passed:
                        solved_problems += 1
                    logger.log_problem_block(log_block)
                    jsonl_logger.log(
                        {
                            "question_id": q_id,
                            "snippet": snippet,
                            "status": None,
                        }
                    )
                except Exception as e:
                    console.print(f"[bold red]Exception in worker: {e}[/bold red]")

    # LLM Usage Summary
    usage_table = Table(title="LLM Usage Summary", border_style="bold blue")
    usage_table.add_column("Metric", style="cyan")
    usage_table.add_column("Value", style="bold yellow")

    usage_table.add_row("Model", llm_client.model)
    usage_table.add_row("Total Input Tokens", f"{llm_client.total_input_tokens:,}")
    usage_table.add_row("Total Output Tokens", f"{llm_client.total_output_tokens:,}")
    usage_table.add_row("Total Tokens", f"{llm_client.total_tokens:,}")

    usage_panel = Panel(usage_table, border_style="bold blue")
    console.print(usage_panel)

    # Final result table
    table = Table(title="Direct Generation Results")
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
            f"Pass Rate: [bold cyan]{(solved_problems / total_problems) * 100:.2f}%[/bold cyan]"
            if total_problems > 0
            else "N/A",
        ),
        title="Summary",
        border_style="bold green",
    )
    console.print(summary_panel)

    # Capture final tables and summaries for the log and config files
    log_final_stream = io.StringIO()
    log_final_console = Console(file=log_final_stream, force_terminal=True, width=120)
    log_final_console.print(usage_panel)
    log_final_console.print(table)
    log_final_console.print(summary_panel)
    final_output = log_final_stream.getvalue()

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 120 + "\n")
        f.write("FINAL SUMMARY\n")
        f.write("=" * 120 + "\n")
        f.write(final_output)
        f.write(f"\nRun completed at {datetime.now()}\n")

    with open(config_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 120 + "\n")
        f.write("LLM USAGE AND RUN SUMMARY\n")
        f.write("=" * 120 + "\n")
        f.write(final_output)
        f.write(f"\nRun completed at {datetime.now()}\n")


if __name__ == "__main__":
    app()
