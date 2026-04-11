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
from coevolution.utils.config import _load_yaml_file

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
CONSTRAINT_EXTRACTION_PROMPT = r"""Problem:
{question_content}

Starter Code:
{starter_code}

Please extract the core mathematical and logical constraints of this problem. 
Be precise about input ranges, complexity requirements ($O(N)$, $O(N \log N)$, etc.), and fundamental properties (monotonicity, parity, etc.).
"""

EDGE_CASE_PROMPT = r"""Problem:
{question_content}

Constraints:
{constraints}

Identify 5-7 adversarial edge cases that might break a naive implementation. 
Focus on:
- Boundary values (min/max N, empty inputs)
- Special patterns (all same elements, alternating, strictly increasing/decreasing)
- Large inputs that test complexity limits
- Precision issues (if applicable)
- Cases where a greedy approach might fail
"""

PLAN_PROMPT = r"""Problem:
{question_content}

Constraints:
{constraints}

Edge Cases:
{edge_cases}

Starter Code:
{starter_code}

Design a robust algorithmic plan that specifically addresses all Constraints and handles the identified Edge Cases.
"""

DRAFT_CODE_PROMPT = r"""Problem:
{question_content}

Plan:
{plan}

Starter Code:
{starter_code}

Provide a draft Python implementation of the plan. Wrap your code in ```python blocks.
"""

MENTAL_TRACE_PROMPT = r"""Problem:
{question_content}

Draft Code:
{draft_code}

Adversarial Edge Cases:
{edge_cases}

Perform a systematic mental execution of the draft code against the identified edge cases. 
For each edge case, trace the main logic and state whether the code produces the correct result or identifying any potential bugs.
"""

CRITIQUE_PROMPT = r"""Problem:
{question_content}

Draft Code:
{draft_code}

Mental Trace Results:
{trace_results}

Critique the draft code based on the mental trace. Identify specific lines or logic that are flawed. 
Propose concrete self-corrections or optimizations to handle the edge cases robustly.
"""

FINAL_CODE_PROMPT = r"""Problem:
{question_content}

Draft Code:
{draft_code}

Critique & Self-Correction:
{critique}

Starter Code:
{starter_code}

Implement the final, polished Python solution. Incorporate all self-corrections and ensure it strictly follows the starter code format.
Wrap your code in ```python blocks.
"""

class ATDMELogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"ATD-ME Run started at {datetime.now()}\n")

    def log_problem_all(self, block: str) -> None:
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
    idx: int,
    total: int,
    progress: Progress,
    task_id: Any,
) -> Tuple[str, bool, str, dict[str, Any]]:
    log_stream = io.StringIO()
    p_console = Console(file=log_stream, force_terminal=True, width=120)
    
    stages = {
        "constraints": "",
        "edge_cases": "",
        "plan": "",
        "draft_code": "",
        "mental_trace": "",
        "critique": "",
        "snippet": ""
    }
    passed = True

    progress.update(task_id, description=f"ATD-ME: {problem.question_title[:40]}...")
    p_console.rule(f"[bold magenta]Problem {idx+1}/{total}: {problem.question_title} ({problem.question_id})[/bold magenta]")

    def call(prompt_template: str, stage_name: str, **kwargs: Any) -> str:
        p_console.print(f"\n[bold cyan]Stage: {stage_name}...[/bold cyan]")
        prompt = prompt_template.format(question_content=problem.question_content, **kwargs)
        try:
            @tenacity.retry(stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_exponential(1, 2, 10))
            def gen() -> str:
                return str(llm_client.generate(prompt))
            res = gen()
            p_console.print(Panel(Syntax(res, "markdown", theme="monokai", padding=1), title=stage_name))
            return res
        except Exception as e:
            p_console.print(f"[bold red]Failed {stage_name}: {e}[/bold red]")
            nonlocal passed
            passed = False
            return f"Error: {e}"

    stages["constraints"] = call(CONSTRAINT_EXTRACTION_PROMPT, "Constraint Extraction", starter_code=problem.starter_code)
    
    if passed:
        stages["edge_cases"] = call(EDGE_CASE_PROMPT, "Edge Case Generation", constraints=stages["constraints"])
    if passed:
        stages["plan"] = call(PLAN_PROMPT, "Algorithmic Planning", constraints=stages["constraints"], edge_cases=stages["edge_cases"], starter_code=problem.starter_code)
    if passed:
        draft_raw = call(DRAFT_CODE_PROMPT, "Draft Implementation", plan=stages["plan"], starter_code=problem.starter_code)
        match = re.search(r"```python\n(.*?)\n```", draft_raw, re.DOTALL)
        stages["draft_code"] = match.group(1).strip() if match else draft_raw.strip()
    if passed:
        stages["mental_trace"] = call(MENTAL_TRACE_PROMPT, "Mental Trace (Dry Run)", draft_code=stages["draft_code"], edge_cases=stages["edge_cases"])
    if passed:
        stages["critique"] = call(CRITIQUE_PROMPT, "Critique & Self-Correction", draft_code=stages["draft_code"], trace_results=stages["mental_trace"])
    if passed:
        final_raw = call(FINAL_CODE_PROMPT, "Final Implementation", draft_code=stages["draft_code"], critique=stages["critique"], starter_code=problem.starter_code)
        match = re.search(r"```python\n(.*?)\n```", final_raw, re.DOTALL)
        stages["snippet"] = match.group(1).strip() if match else final_raw.strip()
        
        if not python_lang.parser.is_syntax_valid(stages["snippet"]):
            p_console.print("[bold red]Syntax error in final code.[/bold red]")
            passed = False

    status = "[bold green]PASSED[/bold green]" if passed else "[bold red]FAILED[/bold red]"
    p_console.print(f"\nFinal Status: {status}")
    progress.advance(task_id)
    return (problem.question_id, passed, log_stream.getvalue(), stages)

@app.command()
def run(
    llm: Path = typer.Option(Path("configs/llm/gpt-5-mini.yaml")),
    count: Optional[int] = typer.Option(None),
    difficulty: str = typer.Option("hard"),
    version: str = typer.Option("release_v6"),
    start_date: Optional[str] = typer.Option("2025-03-01"),
    end_date: Optional[str] = typer.Option("2025-05-10"),
    output_dir: Path = typer.Option(Path("logs/atd_me")),
    workers: int = typer.Option(16),
) -> None:
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    log_file = output_dir / f"{run_id}_atd_me.txt"
    jsonl_file = output_dir / f"{run_id}_solutions.jsonl"
    logger_ = ATDMELogger(log_file)
    jsonl_logger = JsonlLogger(jsonl_file)

    console.print(
        Panel(
            f"Starting ATD-ME Generation Run: [bold cyan]{run_id}[/bold cyan]\n"
            f"Log file: [yellow]{log_file}[/yellow]\n"
            f"JSONL file: [yellow]{jsonl_file}[/yellow]\n"
            f"Workers: [green]{workers}[/green]"
        )
    )

    llm_cfg = _load_yaml_file(llm)
    client = create_llm_client(**llm_cfg)
    lang = PythonLanguage()

    problems = load_code_generation_dataset(release_version=version, difficulty=Difficulty(difficulty.lower()), start_date=start_date, end_date=end_date)
    if count: problems = problems[:count]
    
    total = len(problems)
    results = []
    solved = 0

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("[cyan]ATD-ME Generation...", total=total)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futs = [executor.submit(process_problem, p, client, lang, i, total, progress, task) for i, p in enumerate(problems)]
            for fut in futs:
                try:
                    q_id, passed, log, out = fut.result()
                    results.append((q_id, passed))
                    if passed: solved += 1
                    logger_.log_problem_all(log)
                    jsonl_logger.log({"question_id": q_id, "snippet": out["snippet"], **out, "status": None})
                except Exception as e:
                    console.print(f"[bold red]Worker Error: {e}[/bold red]")

    table = Table(title="ATD-ME Results")
    table.add_column("ID"); table.add_column("Status")
    for q_id, p in results: table.add_row(q_id, "[green]Pass[/green]" if p else "[red]Fail[/red]")
    console.print(table)
    console.print(
        Panel(
            Group(
                f"Total Problems: {total}",
                f"Solved: [bold green]{solved}[/bold green]",
                f"Pass Rate: [bold cyan]{(solved / total) * 100:.2f}%[/bold cyan]"
                if total > 0
                else "N/A",
            ),
            title="Summary",
            border_style="bold green",
        )
    )

if __name__ == "__main__":
    app()
