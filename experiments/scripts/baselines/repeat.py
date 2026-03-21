import io
import json
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

# Silence noisy loggers
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
for logger_name in ["httpx", "openai", "urllib3"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

logger.remove()
logger.add(RichHandler(console=console), level="WARNING")

# Prompts
REPEAT_PROMPT_TEMPLATE = r"""Problem:
{question_content}

Starter Code:
{starter_code}

---
Please review the problem description once more to ensure all constraints are understood before generating the solution:

Problem:
{question_content}

Based on the problem description above, please provide a complete Python solution.
Wrap your code in ```python blocks. Strictly stick to the Starter Code format. 
"""

class RepeatLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"Repeat Run started at {datetime.now()}\n")

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
) -> Tuple[str, bool, str, str]:
    log_stream = io.StringIO()
    p_console = Console(file=log_stream, force_terminal=True, width=120)

    problem_passed = True
    generated_code = ""

    progress.update(task_id, description=f"Repeat Strategy: {problem.question_title[:40]}...")
    p_console.rule(f"[bold magenta]Problem {problem_idx + 1}/{total_problems}: {problem.question_title} ({problem.question_id})[/bold magenta]")
    
    p_console.print(Panel(problem.question_content, title="Question Content", border_style="blue"))

    p_console.print("\n[bold cyan]Generating Python solution with Repeated Prompt...[/bold cyan]")
    prompt = REPEAT_PROMPT_TEMPLATE.format(
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
        
        # Extract code
        code_match = re.search(r"```python\n(.*?)\n```", solution_response, re.DOTALL)
        extracted_code = code_match.group(1).strip() if code_match else solution_response.strip()

        if python_lang.parser.is_syntax_valid(extracted_code):
            generated_code = extracted_code
            p_console.print(Panel(Syntax(solution_response, "markdown", theme="monokai", padding=1), title="LLM Response"))
        else:
            p_console.print("[bold red]Generated code has invalid syntax.[/bold red]")
            problem_passed = False
            generated_code = extracted_code
    except Exception as e:
        p_console.print(f"[bold red]Failed to generate solution: {e}[/bold red]")
        problem_passed = False

    status_str = "[bold green]GENERATED[/bold green]" if problem_passed else "[bold red]FAILED[/bold red]"
    p_console.print(f"\nFinal Problem Status: {status_str}")

    progress.advance(task_id)
    return (problem.question_id, problem_passed, log_stream.getvalue(), generated_code)

@app.command()
def run(
    llm: Path = typer.Option(Path("configs/llm/gpt-5-mini.yaml")),
    count: Optional[int] = typer.Option(None),
    difficulty: str = typer.Option("hard"),
    version: str = typer.Option("release_v6"),
    start_date: Optional[str] = typer.Option("2025-03-01"),
    end_date: Optional[str] = typer.Option("2025-05-10"),
    output_dir: Path = typer.Option(Path("logs/repeat")),
    workers: int = typer.Option(16),
) -> None:
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    log_file = output_dir / f"{run_id}_repeat.txt"
    jsonl_file = output_dir / f"{run_id}_solutions.jsonl"
    logger_ = RepeatLogger(log_file)
    jsonl_logger = JsonlLogger(jsonl_file)

    console.print(
        Panel(
            f"Starting Repeat-Problem Generation Run: [bold cyan]{run_id}[/bold cyan]\n"
            f"Log file: [yellow]{log_file}[/yellow]\n"
            f"JSONL file: [yellow]{jsonl_file}[/yellow]\n"
            f"Workers: [green]{workers}[/green]"
        )
    )

    with open(llm, "r") as f:
        llm_cfg = yaml.safe_load(f)

    llm_client = create_llm_client(**llm_cfg)
    python_lang = PythonLanguage()

    problems = load_code_generation_dataset(release_version=version, difficulty=Difficulty(difficulty.lower()), start_date=start_date, end_date=end_date)
    if count:
        problems = problems[:count]

    total_problems = len(problems)
    results = []
    solved_problems = 0

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
        overall_task = progress.add_task("[cyan]Processing repeat strategy...", total=total_problems)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            fut_to_p = {executor.submit(process_problem, p, llm_client, python_lang, i, total_problems, progress, overall_task): p for i, p in enumerate(problems)}
            for fut in fut_to_p:
                try:
                    q_id, passed, log_block, snippet = fut.result()
                    results.append((q_id, passed))
                    if passed: solved_problems += 1
                    logger_.log_problem_block(log_block)
                    jsonl_logger.log({"question_id": q_id, "snippet": snippet, "status": None})
                except Exception as e:
                    console.print(f"[bold red]Exception in worker: {e}[/bold red]")

    table = Table(title="Repeat Strategy Results")
    table.add_column("Question ID", style="cyan")
    table.add_column("Status", style="bold")
    for q_id, passed in results:
        table.add_row(q_id, "[green]Generated[/green]" if passed else "[red]Failed[/red]")
    console.print(table)

    console.print(
        Panel(
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
    )

if __name__ == "__main__":
    app()
