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
from coevolution.utils.config import _load_yaml_file
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

DECOMPOSE_PROMPT_TEMPLATE = """Problem:
{question_content}

Starter Code:
{starter_code}

Please decompose this problem into a list of helper functions.
For each helper function, provide:
1. Its name.
2. Its signature (arguments and return type).
3. A comprehensive description of its responsibility. The developer will not know the problem statement.

**Important Instructions**:
- These helper functions must be global-level functions.
- Provide a clear and strict signature for each.
- Your response must follow this format for each helper:
<helper>
Name: [Name]
Signature: [Signature]
Description: [Description]
</helper>
"""

IMPLEMENT_HELPER_PROMPT_TEMPLATE = """Problem:
{question_content}

Helper Function to Implement:
Name: {helper_name}
Signature: {helper_signature}
Description: {helper_description}

Please provide the Python implementation for this helper function.
Wrap your code in ```python blocks.
"""

ASSEMBLY_PROMPT_TEMPLATE = """Problem:
{question_content}

Helper Functions (already implemented):
{helpers_code}

Starter Code:
{starter_code}

Based on the problem description and the helper functions provided above, please provide the implementation for the Starter Code.
Your implementation should use the provided helper functions.

**Important Instructions**:
- Strictly follow the Starter Code format.
- Do NOT re-implement the helper functions inside the code block; assume they are available at the global level.
- Wrap your code in ```python blocks.
"""


class ModularLogger:
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


def parse_decomposition(response: str) -> list[dict[str, str]]:
    helpers = []
    # Match <helper> blocks
    helper_blocks = re.findall(r"<helper>(.*?)</helper>", response, re.DOTALL)
    for block in helper_blocks:
        name_match = re.search(r"Name:\s*(.*)", block)
        sig_match = re.search(r"Signature:\s*(.*)", block)
        desc_match = re.search(r"Description:\s*(.*)", block)
        if name_match and sig_match and desc_match:
            helpers.append(
                {
                    "name": name_match.group(1).strip(),
                    "signature": sig_match.group(1).strip(),
                    "description": desc_match.group(1).strip(),
                }
            )
    return helpers


def process_problem(
    problem: Any,
    llm_client: Any,
    python_lang: PythonLanguage,
    problem_idx: int,
    total_problems: int,
    progress: Progress,
    task_id: Any,
) -> Tuple[str, bool, str, str, list[dict[str, Any]]]:
    """Process a single problem and return results."""

    log_stream = io.StringIO()
    p_console = Console(file=log_stream, force_terminal=True, width=120)

    problem_passed = True
    final_snippet = ""
    helpers_data = []

    progress.update(
        task_id, description=f"Processing: {problem.question_title[:40]}..."
    )

    p_console.rule(
        f"[bold magenta]Problem {problem_idx + 1}/{total_problems}: {problem.question_title} ({problem.question_id})[/bold magenta]"
    )
    p_console.print(
        Panel(problem.question_content, title="Question Content", border_style="blue")
    )

    # Stage 1: Decomposition
    p_console.print("\n[bold cyan]Stage 1: Decomposing Problem...[/bold cyan]")
    decompose_prompt = DECOMPOSE_PROMPT_TEMPLATE.format(
        question_content=problem.question_content,
        starter_code=problem.starter_code,
    )

    try:

        @tenacity.retry(
            stop=tenacity.stop_after_attempt(5),
            wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
            retry=tenacity.retry_if_exception_type(Exception),
        )
        def generate_decomposition() -> str:
            return str(llm_client.generate(decompose_prompt))

        decomposition_response = generate_decomposition()
        helpers = parse_decomposition(decomposition_response)

        p_console.print(
            Panel(
                Syntax(decomposition_response, "markdown", theme="monokai", padding=1),
                title="Decomposition Response",
            )
        )
        p_console.print(f"Parsed {len(helpers)} helper functions.")
    except Exception as e:
        p_console.print(f"[bold red]Failed to decompose: {e}[/bold red]")
        problem_passed = False
        helpers = []

    # Stage 2: Sub-Implementation
    helper_implementations = []
    if problem_passed and helpers:
        p_console.print(
            "\n[bold cyan]Stage 2: Implementing Sub-Functions...[/bold cyan]"
        )
        for helper in helpers:
            p_console.print(f"Implementing: {helper['name']}...")
            impl_prompt = IMPLEMENT_HELPER_PROMPT_TEMPLATE.format(
                question_content=problem.question_content,
                helper_name=helper["name"],
                helper_signature=helper["signature"],
                helper_description=helper["description"],
            )

            try:

                @tenacity.retry(
                    stop=tenacity.stop_after_attempt(5),
                    wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
                    retry=tenacity.retry_if_exception_type(Exception),
                )
                def generate_helper_impl() -> str:
                    return str(llm_client.generate(impl_prompt))

                impl_response = generate_helper_impl()

                # Extract code
                code_match = re.search(
                    r"```python\n(.*?)\n```", impl_response, re.DOTALL
                )
                helper_code = (
                    code_match.group(1).strip() if code_match else impl_response.strip()
                )

                helper_implementations.append(helper_code)
                helpers_data.append({**helper, "code": helper_code})

                p_console.print(f"Implemented {helper['name']}.")
            except Exception as e:
                p_console.print(
                    f"[bold red]Failed to implement {helper['name']}: {e}[/bold red]"
                )
                problem_passed = False
                break

    # Stage 3: Assembly
    if problem_passed:
        p_console.print(
            "\n[bold cyan]Stage 3: Assembling Final Solution...[/bold cyan]"
        )
        helpers_code_flat = "\n\n".join(helper_implementations)
        assembly_prompt = ASSEMBLY_PROMPT_TEMPLATE.format(
            question_content=problem.question_content,
            helpers_code=helpers_code_flat,
            starter_code=problem.starter_code,
        )

        try:

            @tenacity.retry(
                stop=tenacity.stop_after_attempt(5),
                wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
                retry=tenacity.retry_if_exception_type(Exception),
            )
            def generate_assembly() -> str:
                return str(llm_client.generate(assembly_prompt))

            assembly_response = generate_assembly()

            # Extract code
            code_match = re.search(
                r"```python\n(.*?)\n```", assembly_response, re.DOTALL
            )
            main_code = (
                code_match.group(1).strip() if code_match else assembly_response.strip()
            )

            # Final snippet combines helpers and main code
            final_snippet = helpers_code_flat + "\n\n" + main_code

            if python_lang.parser.is_syntax_valid(final_snippet):
                p_console.print(
                    Panel(
                        Syntax(final_snippet, "python", theme="monokai", padding=1),
                        title="Final Assembled Solution (Valid Syntax)",
                    )
                )
            else:
                p_console.print(
                    "[bold red]Generated solution has invalid syntax.[/bold red]"
                )
                problem_passed = False
        except Exception as e:
            p_console.print(f"[bold red]Failed to assemble solution: {e}[/bold red]")
            problem_passed = False

    status_str = (
        "[bold green]PASSED[/bold green]"
        if problem_passed
        else "[bold red]FAILED[/bold red]"
    )
    p_console.print(f"\nFinal Problem Status: {status_str}")

    progress.advance(task_id)
    return (
        problem.question_id,
        problem_passed,
        log_stream.getvalue(),
        final_snippet,
        helpers_data,
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
        Path("logs/modular"), help="Directory to save logs"
    ),
    workers: int = typer.Option(16, help="Number of parallel workers"),
) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    log_file = output_dir / f"{run_id}_modular.txt"
    jsonl_file = output_dir / f"{run_id}_solutions.jsonl"
    logger = ModularLogger(log_file)
    jsonl_logger = JsonlLogger(jsonl_file)

    console.print(
        Panel(
            f"Starting Modular Generation Run: [bold cyan]{run_id}[/bold cyan]\n"
            f"Log file: [yellow]{log_file}[/yellow]\n"
            f"JSONL file: [yellow]{jsonl_file}[/yellow]\n"
            f"Workers: [green]{workers}[/green]"
        )
    )

    # Load LLM Config
    llm_cfg = _load_yaml_file(llm)

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
        overall_task = progress.add_task(
            "[cyan]Processing modular problems...", total=total_problems
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
                    q_id, passed, log_block, snippet, helpers = fut.result()
                    results.append((q_id, passed))
                    if passed:
                        solved_problems += 1
                    logger.log_problem_block(log_block)
                    jsonl_logger.log(
                        {
                            "question_id": q_id,
                            "snippet": snippet,
                            "helpers": helpers,
                            "status": None,
                        }
                    )
                except Exception as e:
                    console.print(f"[bold red]Exception in worker: {e}[/bold red]")

    # Final result table
    table = Table(title="Modular Generation Results")
    table.add_column("Question ID", style="cyan")
    table.add_column("Status", style="bold")

    for q_id, passed in results:
        status_text = "[green]Passed[/green]" if passed else "[red]Failed[/red]"
        table.add_row(q_id, status_text)

    console.print(table)

    summary_panel = Panel(
        Group(
            f"Total Problems: {total_problems}",
            f"Solved: [bold green]{solved_problems}[/bold green]",
            f"Success Rate: [bold yellow]{(solved_problems / total_problems) * 100:.2f}%[/bold yellow]"
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
