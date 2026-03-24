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
from rich.console import Console
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

# Prompts
CONSTRAINT_EXTRACTION_PROMPT = r"""Problem:
{question_content}

Starter Code:
{starter_code}

Please extract the fundamental logical constraints and invariants of this problem. 
Consider: 
- Monotonicity (e.g., is the output always non-decreasing?)
- Conservation (e.g., is the sum of elements preserved?)
- Parity (e.g., must the result be even/odd?)
- Ordering (e.g., are certain elements always before others?)
- Ranges (e.g., upper/lower bounds on intermediate values)

List them clearly.
"""

DERIVE_CONDITIONS_PROMPT = r"""Problem:
{question_content}

Invariants:
{invariants}

Based on these invariants, derive the **necessary and sufficient conditions** that the algorithm must satisfy to be correct. 
Express these as logical rules or mathematical properties.
"""

INVARIANT_PLAN_PROMPT = r"""Problem:
{question_content}

Invariants:
{invariants}

Necessary Conditions:
{conditions}

Starter Code:
{starter_code}

Plan a step-by-step algorithm that is guaranteed to respect all the invariants and satisfy the conditions derived above.
"""

GENERATE_CODE_PROMPT = r"""Problem:
{question_content}

Invariant Plan:
{plan}

Starter Code:
{starter_code}

Implement the Python code based on this plan. Wrap your code in ```python blocks. Strictly follow the starter code format.
"""

SYMBOLIC_VERIFY_PROMPT = r"""Problem:
{question_content}

Invariants:
{invariants}

Implemented Code:
{code}

Perform a symbolic verification of the code. For each invariant, explain whether the code preserves it or if there's a potential violation (e.g., an off-by-one error or a missed boundary condition).
"""

REPAIR_CODE_PROMPT = r"""Problem:
{question_content}

Invariants:
{invariants}

Code:
{code}

Verification Findings:
{verification}

Based on the verification findings, repair the code to ensure it strictly adheres to all invariants. 
Wrap the final repaired code in ```python blocks. Strictly follow the starter code format.
"""


class InvariantLogger:
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
) -> Tuple[str, bool, str, Dict[str, str]]:
    """Process a single problem through 7 stages of invariant-based generation."""
    log_stream = io.StringIO()
    p_console = Console(file=log_stream, force_terminal=True, width=120)

    stages_output = {
        "invariants": "",
        "conditions": "",
        "plan": "",
        "initial_code": "",
        "verification": "",
        "repairs": "",
        "snippet": ""
    }
    problem_passed = True

    progress.update(task_id, description=f"Processing: {problem.question_title[:40]}...")

    p_console.rule(f"[bold magenta]Problem {problem_idx + 1}/{total_problems}: {problem.question_title} ({problem.question_id})[/bold magenta]")
    p_console.print(Panel(problem.question_content, title="Question Content", border_style="blue"))

    def call_llm(prompt: str, stage_name: str) -> str:
        p_console.print(f"\n[bold cyan]Stage: {stage_name}...[/bold cyan]")
        try:
            @tenacity.retry(
                stop=tenacity.stop_after_attempt(5),
                wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
                retry=tenacity.retry_if_exception_type(Exception),
            )
            def generate() -> str:
                return str(llm_client.generate(prompt))
            res = generate()
            p_console.print(Panel(Syntax(res, "markdown", theme="monokai", padding=1), title=stage_name))
            return res
        except Exception as e:
            p_console.print(f"[bold red]Failed {stage_name}: {e}[/bold red]")
            nonlocal problem_passed
            problem_passed = False
            return f"Error: {e}"

    # 1. Extract Invariants
    stages_output["invariants"] = call_llm(
        CONSTRAINT_EXTRACTION_PROMPT.format(question_content=problem.question_content, starter_code=problem.starter_code),
        "Constraint Extraction"
    )

    # 2. Derive Conditions
    if problem_passed:
        stages_output["conditions"] = call_llm(
            DERIVE_CONDITIONS_PROMPT.format(question_content=problem.question_content, invariants=stages_output["invariants"]),
            "Condition Derivation"
        )

    # 3. Plan
    if problem_passed:
        stages_output["plan"] = call_llm(
            INVARIANT_PLAN_PROMPT.format(
                question_content=problem.question_content,
                invariants=stages_output["invariants"],
                conditions=stages_output["conditions"],
                starter_code=problem.starter_code
            ),
            "Invariant-Consistent Planning"
        )

    # 4. Initial Code
    if problem_passed:
        stages_output["initial_code_raw"] = call_llm(
            GENERATE_CODE_PROMPT.format(
                question_content=problem.question_content,
                plan=stages_output["plan"],
                starter_code=problem.starter_code
            ),
            "Code Generation"
        )
        # Extract code
        code_match = re.search(r"```python\n(.*?)\n```", stages_output["initial_code_raw"], re.DOTALL)
        stages_output["initial_code"] = code_match.group(1).strip() if code_match else stages_output["initial_code_raw"].strip()

    # 5. Symbolic Verification
    if problem_passed:
        stages_output["verification"] = call_llm(
            SYMBOLIC_VERIFY_PROMPT.format(
                question_content=problem.question_content,
                invariants=stages_output["invariants"],
                code=stages_output["initial_code"]
            ),
            "Symbolic Verification"
        )

    # 6. Repair
    if problem_passed:
        stages_output["repairs_raw"] = call_llm(
            REPAIR_CODE_PROMPT.format(
                question_content=problem.question_content,
                invariants=stages_output["invariants"],
                code=stages_output["initial_code"],
                verification=stages_output["verification"]
            ),
            "Heuristic Repair"
        )
        code_match = re.search(r"```python\n(.*?)\n```", stages_output["repairs_raw"], re.DOTALL)
        stages_output["snippet"] = code_match.group(1).strip() if code_match else stages_output["repairs_raw"].strip()

        # Final Syntax Validation
        if not python_lang.parser.is_syntax_valid(stages_output["snippet"]):
            p_console.print("[bold red]Final code has invalid syntax.[/bold red]")
            problem_passed = False
    else:
        stages_output["snippet"] = ""

    status_str = "[bold green]PASSED[/bold green]" if problem_passed else "[bold red]FAILED[/bold red]"
    p_console.print(f"\nFinal Problem Status: {status_str}")

    progress.advance(task_id)
    return (
        problem.question_id,
        problem_passed,
        log_stream.getvalue(),
        stages_output
    )


@app.command()
def run(
    llm: Path = typer.Option(Path("configs/llm/gpt-5-mini.yaml"), help="Path to LLM config YAML"),
    count: Optional[int] = typer.Option(None, help="Number of problems to process"),
    difficulty: str = typer.Option("hard", help="Difficulty (easy, medium, hard)"),
    version: str = typer.Option("release_v6", help="LCB version"),
    start_date: Optional[str] = typer.Option("2025-03-01"),
    end_date: Optional[str] = typer.Option("2025-05-10"),
    output_dir: Path = typer.Option(Path("logs/invariant"), help="Log directory"),
    workers: int = typer.Option(16, help="Workers"),
) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    log_file = output_dir / f"{run_id}_invariant.txt"
    jsonl_file = output_dir / f"{run_id}_solutions.jsonl"
    logger = InvariantLogger(log_file)
    jsonl_logger = JsonlLogger(jsonl_file)
    
    console.print(
        Panel(
            f"Starting Invariant Run: [bold cyan]{run_id}[/bold cyan]\n"
            f"Log file: [yellow]{log_file}[/yellow]\n"
            f"JSONL file: [yellow]{jsonl_file}[/yellow]\n"
            f"Workers: [green]{workers}[/green]"
        )
    )

    with open(llm, "r") as f:
        llm_client = create_llm_client(**yaml.safe_load(f))
    python_lang = PythonLanguage()

    problems = load_code_generation_dataset(
        release_version=version,
        difficulty=Difficulty(difficulty.lower()),
        start_date=start_date,
        end_date=end_date,
    )
    if count:
        problems = problems[:count]

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Processing...", total=len(problems))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_problem, p, llm_client, python_lang, i, len(problems), progress, task) for i, p in enumerate(problems)]
            for fut in futures:
                try:
                    q_id, passed, log, output = fut.result()
                    logger.log_problem_block(log)
                    jsonl_logger.log({"question_id": q_id, "snippet": output["snippet"], **output, "status": None})
                except Exception as e:
                    console.print(f"[bold red]Worker Error: {e}[/bold red]")

    console.print("\n[bold green]Run Completed.[/bold green]")


if __name__ == "__main__":
    app()
