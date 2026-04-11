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

import numpy as np
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

from coevolution.core.individual import CodeIndividual, TestIndividual
from coevolution.core.interfaces import OPERATION_INITIAL
from coevolution.core.population import CodePopulation, TestPopulation
from coevolution.dataset.lcb import Difficulty, load_code_generation_dataset
from coevolution.services.execution import ExecutionSystem
from infrastructure.languages.python.adapter import PythonLanguage
from infrastructure.sandbox.types import SandboxConfig
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
DIRECT_PROMPT = r"""Problem:
{question_content}

Starter Code:
{starter_code}

Based on the problem description above, please provide a complete Python solution.
Wrap your code in ```python blocks. Strictly stick to the Starter Code format. 
"""

TRACE_ANALYSIS_PROMPT = r"""Problem:
{question_content}

Starter Code:
{starter_code}

Failed Code:
{code}

Public Test Failures:
{failures}

If the failures appear to be "test case setup" or "import" errors (e.g., `ImportError`, `NameError`), this is almost always because the **Failed Code** does not strictly adhere to the **Starter Code** signature (function name, class name, or parameters).

Choose the first failing test case and perform a step-by-step mental trace of the code's execution on this input. 
Identify exactly which line or logic causes the deviation from the expected output. Show your reasoning.
"""

STRUCTURAL_REPAIR_PROMPT = r"""Problem:
{question_content}

Failed Code:
{code}

Trace Analysis:
{trace}

Public Test Failures:
{failures}

Starter Code:
{starter_code}

Based on the trace analysis, provide a complete, corrected Python implementation. 

**IMPORTANT**: You must strictly follow the **Starter Code** format above. Do NOT change the function or class signature. Any deviation in the signature will cause testing to fail.

Wrap the final repaired code in ```python blocks.
"""

class DirectExecLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"Direct-Exec Run started at {datetime.now()}\n")

    def log_problem(self, block: str) -> None:
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
    execution_system: ExecutionSystem,
    idx: int,
    total: int,
    progress: Progress,
    task_id: Any,
) -> Tuple[str, bool, str, Dict[str, Any]]:
    log_stream = io.StringIO()
    p_console = Console(file=log_stream, force_terminal=True, width=120)
    
    stages = {
        "initial_code": "",
        "execution_feedback": "",
        "trace_analysis": "",
        "structural_repair": "",
        "snippet": ""
    }
    passed = True

    progress.update(task_id, description=f"DirectExec: {problem.question_title[:40]}...")
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

    # 1. Initial Code Generation
    raw = call(DIRECT_PROMPT, "Initial Code Generation", starter_code=problem.starter_code)
    match = re.search(r"```python\n(.*?)\n```", raw, re.DOTALL)
    stages["initial_code"] = match.group(1).strip() if match else raw.strip()
    stages["snippet"] = stages["initial_code"]

    # 2. Execution on Public Tests
    if passed:
        p_console.print("\n[bold cyan]Stage: Public Test Execution...[/bold cyan]")
        
        # Setup Code Individual
        code_ind = CodeIndividual(
            snippet=python_lang.parser.remove_main_block(stages["snippet"]),
            probability=1.0,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )
        code_pop = CodePopulation(individuals=[code_ind], generation=0)

        # Setup Test Population (Public Tests only)
        public_test_functions = [
            python_lang.composer.generate_test_case(
                tc.input, tc.output, problem.starter_code, i + 1
            )
            for i, tc in enumerate(problem.public_test_cases)
        ]
        test_inds = [
            TestIndividual(snippet=f, probability=1.0, creation_op=OPERATION_INITIAL, generation_born=0, parents={})
            for f in public_test_functions
        ]
        test_pop = TestPopulation(individuals=test_inds, generation=0)

        try:
            interaction = execution_system.execute_tests(code_pop, test_pop)
            results = interaction.observation_matrix[0]
            
            if results.sum() < len(public_test_functions):
                # We have failures
                p_console.print(f"[yellow]Failed {len(public_test_functions) - int(results.sum())} public tests.[/yellow]")
                
                # Format Failures for Logging
                failure_table = Table(title="Public Test Failures", show_header=True, header_style="bold red")
                failure_table.add_column("Input", style="cyan")
                failure_table.add_column("Expected Output", style="green")
                failure_table.add_column("Error / Log", style="red")

                exec_results = interaction.execution_results.results[code_ind.id]
                raw_failures_text = []

                for i, test_individual in enumerate(test_inds):
                    run_res = exec_results[test_individual.id]
                    if run_res.status != "passed":
                        tc = problem.public_test_cases[i]
                        failure_table.add_row(str(tc.input), str(tc.output), str(run_res.error_log))
                        raw_failures_text.append(
                            f"Test Case {i+1}:\nInput: {tc.input}\nExpected: {tc.output}\nError/Log: {run_res.error_log}\n"
                        )
                
                p_console.print(failure_table)
                failure_str = "\n".join(raw_failures_text)
                stages["execution_feedback"] = failure_str

                # 3a. Trace Analysis
                stages["trace_analysis"] = call(
                    TRACE_ANALYSIS_PROMPT,
                    "Mental Trace Analysis",
                    code=stages["snippet"],
                    failures=failure_str,
                    starter_code=problem.starter_code
                )

                # 3b. Structural Repair
                repair_raw = call(
                    STRUCTURAL_REPAIR_PROMPT,
                    "Structural Repair",
                    code=stages["snippet"],
                    trace=stages["trace_analysis"],
                    failures=failure_str,
                    starter_code=problem.starter_code
                )
                
                match = re.search(r"```python\n(.*?)\n```", repair_raw, re.DOTALL)
                stages["structural_repair"] = match.group(1).strip() if match else repair_raw.strip()
                stages["snippet"] = stages["structural_repair"]
            else:
                p_console.print("[bold green]All public tests passed![/bold green]")
                stages["execution_feedback"] = "All public tests passed."
        except Exception as e:
            p_console.print(f"[bold red]Execution system error: {e}[/bold red]")

    # Final Syntax Check
    if passed and not python_lang.parser.is_syntax_valid(stages["snippet"]):
        p_console.print("[bold red]Final code has invalid syntax.[/bold red]")
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
    output_dir: Path = typer.Option(Path("logs/direct_exec")),
    workers: int = typer.Option(16),
) -> None:
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    log_file = output_dir / f"{run_id}_direct_exec.txt"
    jsonl_file = output_dir / f"{run_id}_solutions.jsonl"
    logger_ = DirectExecLogger(log_file)
    jsonl_logger = JsonlLogger(jsonl_file)

    console.print(
        Panel(
            f"Starting Direct-Execution Run: [bold cyan]{run_id}[/bold cyan]\n"
            f"Log file: [yellow]{log_file}[/yellow]\n"
            f"JSONL file: [yellow]{jsonl_file}[/yellow]\n"
            f"Workers: [green]{workers}[/green]"
        )
    )

    llm_cfg = _load_yaml_file(llm)
    client = create_llm_client(**llm_cfg)
    python_lang = PythonLanguage()
    
    sandbox_config = SandboxConfig(timeout=30)
    execution_system = ExecutionSystem(
        sandbox_config=sandbox_config,
        composer=python_lang.composer,
        runtime=python_lang.runtime,
        analyzer=python_lang.analyzer,
        enable_multiprocessing=True,
        cpu_workers=workers,
    )

    problems = load_code_generation_dataset(release_version=version, difficulty=Difficulty(difficulty.lower()), start_date=start_date, end_date=end_date)
    if count:
        problems = problems[:count]
    
    total = len(problems)
    results = []
    solved = 0

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Processing Direct-Exec...", total=total)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futs = [executor.submit(process_problem, p, client, python_lang, execution_system, i, total, progress, task) for i, p in enumerate(problems)]
            for fut in futs:
                try:
                    q_id, passed, log, out = fut.result()
                    results.append((q_id, passed))
                    if passed: solved += 1
                    logger_.log_problem(log)
                    jsonl_logger.log({"question_id": q_id, "snippet": out["snippet"], **out, "status": None})
                except Exception as e:
                    console.print(f"[bold red]Worker Error: {e}[/bold red]")

    table = Table(title="Direct-Exec Results")
    table.add_column("ID")
    table.add_column("Status")
    for q_id, p in results:
        table.add_row(q_id, "[green]Pass[/green]" if p else "[red]Fail[/red]")
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
