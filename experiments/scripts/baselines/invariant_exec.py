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
from infrastructure.llm_client.factory import create_llm_client
from coevolution.utils.config import _load_yaml_file
from infrastructure.sandbox.types import SandboxConfig

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

Please extract the fundamental logical constraints and invariants of this problem. 
Consider: 
- Monotonicity (non-decreasing output, etc.)
- Conservation (sum preservation, etc.)
- Parity (even/odd constraints)
- Ordering (relative order of elements)
- Ranges (intermediate value bounds)

List them clearly.
"""

DERIVE_CONDITIONS_PROMPT = r"""Problem:
{question_content}

Invariants:
{invariants}

Based on these invariants, derive the **necessary and sufficient conditions** that the algorithm must satisfy. 
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

MENTAL_REFINEMENT_PROMPT = r"""Problem:
{question_content}

Invariants:
{invariants}

Code:
{code}

Verification Findings:
{verification}

Starter Code:
{starter_code}

Based on the verification findings, repair the code to ensure it strictly adheres to all invariants. 

**IMPORTANT**: You must strictly follow the **Starter Code** format above. Do NOT change the function or class signature. Any deviation in the signature will cause testing to fail.

Wrap the final repaired code in ```python blocks.
"""

TRACE_ANALYSIS_PROMPT = r"""Problem:
{question_content}

Starter Code:
{starter_code}

Invariants:
{invariants}

Failed Code:
{code}

Public Test Failures:
{failures}

If the failures appear to be "test case setup" or "import" errors (e.g., `ImportError`, `NameError`), this is almost always because the **Failed Code** does not strictly adhere to the **Starter Code** signature (function name, class name, or parameters). In such cases, the root cause is a bug in the code implementation (A).

Choose a failing test case and perform a step-by-step mental trace of the code's execution on this input. 
Identify exactly which line or logic causes the deviation from the expected output. Show your reasoning.

Finally, conclude whether the root cause is:
A) A bug in the code implementation (the invariants were correct, but the code didn't follow them, OR the code didn't follow the starter code signature).
B) A bug in the invariants (the invariants themselves were incomplete or wrong, leading to a flawed plan/code).

End your response with exactly one of:
Conclusion: [BUG_IN_CODE]
Conclusion: [BUG_IN_INVARIANTS]
"""

INVARIANT_REFINEMENT_PROMPT = r"""Problem:
{question_content}

Initial Invariants:
{invariants}

Failed Code (based on initial invariants):
{code}

Trace Analysis of Failure:
{trace}

Public Test Failures:
{failures}

The initial invariants were insufficient or incorrect. Based on the trace and failures, refine the logical constraints and invariants of this problem. 
Ensure the new invariants account for the observed failure and provide a more robust foundation for a correct algorithm.
List them clearly.
"""

STRUCTURAL_REPAIR_PROMPT = r"""Problem:
{question_content}

Invariants:
{invariants}

Failed Code:
{code}

Trace Analysis:
{trace}

Public Test Failures:
{failures}

Starter Code:
{starter_code}

Based on the trace analysis, provide a complete, corrected Python implementation. 
Ensure the new logic respects all invariants and passes all test cases.

**IMPORTANT**: You must strictly follow the **Starter Code** format above. Do NOT change the function or class signature. Any deviation in the signature will cause testing to fail.

Wrap the final repaired code in ```python blocks.
"""


class InvariantExecLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"Invariant-Exec Run started at {datetime.now()}\n")

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
        "invariants": "",
        "conditions": "",
        "plan": "",
        "initial_code": "",
        "mental_verification": "",
        "mental_refinement": "",
        "trace_analysis": "",
        "refined_invariants": "",
        "structural_repair": "",
        "execution_feedback": "",
        "snippet": "",
    }
    passed = True

    progress.update(
        task_id, description=f"InvariantExec: {problem.question_title[:40]}..."
    )
    p_console.rule(
        f"[bold magenta]Problem {idx + 1}/{total}: {problem.question_title} ({problem.question_id})[/bold magenta]"
    )

    def call(prompt_template: str, stage_name: str, **kwargs: Any) -> str:
        p_console.print(f"\n[bold cyan]Stage: {stage_name}...[/bold cyan]")
        prompt = prompt_template.format(
            question_content=problem.question_content, **kwargs
        )
        try:

            @tenacity.retry(
                stop=tenacity.stop_after_attempt(5),
                wait=tenacity.wait_exponential(1, 2, 10),
            )
            def gen() -> str:
                return str(llm_client.generate(prompt))

            res = gen()
            p_console.print(
                Panel(
                    Syntax(res, "markdown", theme="monokai", padding=1),
                    title=stage_name,
                )
            )
            return res
        except Exception as e:
            p_console.print(f"[bold red]Failed {stage_name}: {e}[/bold red]")
            nonlocal passed
            passed = False
            return f"Error: {e}"

    # 1. Invariants
    stages["invariants"] = call(
        CONSTRAINT_EXTRACTION_PROMPT,
        "Invariant Extraction",
        starter_code=problem.starter_code,
    )

    # 2. Conditions
    if passed:
        stages["conditions"] = call(
            DERIVE_CONDITIONS_PROMPT,
            "Condition Derivation",
            invariants=stages["invariants"],
        )

    # 3. Plan
    if passed:
        stages["plan"] = call(
            INVARIANT_PLAN_PROMPT,
            "Invariant-Consistent Planning",
            invariants=stages["invariants"],
            conditions=stages["conditions"],
            starter_code=problem.starter_code,
        )

    # 4. Initial Code
    if passed:
        raw = call(
            GENERATE_CODE_PROMPT,
            "Initial Code Generation",
            plan=stages["plan"],
            starter_code=problem.starter_code,
        )
        match = re.search(r"```python\n(.*?)\n```", raw, re.DOTALL)
        stages["initial_code"] = match.group(1).strip() if match else raw.strip()
        stages["snippet"] = stages["initial_code"]

    # 5. Symbolic Verification (Mental check)
    if passed:
        stages["mental_verification"] = call(
            SYMBOLIC_VERIFY_PROMPT,
            "Symbolic Verification",
            invariants=stages["invariants"],
            code=stages["snippet"],
        )

    # 6. Mental Refinement
    if passed:
        refine_raw = call(
            MENTAL_REFINEMENT_PROMPT,
            "Mental Refinement",
            invariants=stages["invariants"],
            code=stages["snippet"],
            verification=stages["mental_verification"],
            starter_code=problem.starter_code
        )
        match = re.search(r"```python\n(.*?)\n```", refine_raw, re.DOTALL)
        stages["mental_refinement"] = (
            match.group(1).strip() if match else refine_raw.strip()
        )
        stages["snippet"] = stages["mental_refinement"]

    # 7. Execution on Public Tests
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
            TestIndividual(
                snippet=f,
                probability=1.0,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
                parents={},
            )
            for f in public_test_functions
        ]
        test_pop = TestPopulation(individuals=test_inds, generation=0)

        try:
            interaction = execution_system.execute_tests(code_pop, test_pop)
            results = interaction.observation_matrix[0]

            if results.sum() < len(public_test_functions):
                # We have failures
                p_console.print(
                    f"[yellow]Failed {len(public_test_functions) - int(results.sum())} public tests.[/yellow]"
                )

                # 1. Format Failures for Logging
                failure_table = Table(
                    title="Public Test Failures",
                    show_header=True,
                    header_style="bold red",
                )
                failure_table.add_column("Input", style="cyan")
                failure_table.add_column("Expected Output", style="green")
                failure_table.add_column("Error / Log", style="red")

                exec_results = interaction.execution_results.results[code_ind.id]
                raw_failures_text = []

                for i, test_individual in enumerate(test_inds):
                    run_res = exec_results[test_individual.id]
                    if run_res.status != "passed":
                        tc = problem.public_test_cases[i]
                        failure_table.add_row(
                            str(tc.input), str(tc.output), str(run_res.error_log)
                        )
                        raw_failures_text.append(
                            f"Test Case {i + 1}:\nInput: {tc.input}\nExpected: {tc.output}\nError/Log: {run_res.error_log}\n"
                        )

                # Print table to log stream
                p_console.print(failure_table)
                failure_str = "\n".join(raw_failures_text)
                stages["execution_feedback"] = failure_str

                # 8a. Trace Analysis
                stages["trace_analysis"] = call(
                    TRACE_ANALYSIS_PROMPT,
                    "Mental Trace Analysis",
                    invariants=stages["invariants"],
                    code=stages["snippet"],
                    failures=failure_str,
                    starter_code=problem.starter_code
                )

                if "[BUG_IN_INVARIANTS]" in stages["trace_analysis"]:
                    p_console.print(
                        "[bold yellow]Root cause: Invariant Failure. Backtracking to Refinement...[/bold yellow]"
                    )

                    # 9a. Invariant Refinement
                    stages["refined_invariants"] = call(
                        INVARIANT_REFINEMENT_PROMPT,
                        "Invariant Refinement",
                        invariants=stages["invariants"],
                        code=stages["snippet"],
                        trace=stages["trace_analysis"],
                        failures=failure_str,
                    )

                    # 9b. Restart derived stages
                    stages["conditions"] = call(
                        DERIVE_CONDITIONS_PROMPT,
                        "Refined Condition Derivation",
                        invariants=stages["refined_invariants"],
                    )

                    stages["plan"] = call(
                        INVARIANT_PLAN_PROMPT,
                        "Refined Invariant Planning",
                        invariants=stages["refined_invariants"],
                        conditions=stages["conditions"],
                        starter_code=problem.starter_code,
                    )

                    # 9c. Structural Repair (Final from refined plan)
                    repair_raw = call(
                        STRUCTURAL_REPAIR_PROMPT,
                        "Structural Repair (Refined)",
                        invariants=stages["refined_invariants"],
                        code=stages["snippet"],
                        trace=stages["trace_analysis"],
                        failures=failure_str,
                        starter_code=problem.starter_code
                    )
                else:
                    p_console.print(
                        "[bold blue]Root cause: Code Implementation Bug. Proceeding with Repair...[/bold blue]"
                    )

                    # 8b. Structural Repair (Code fix only)
                    repair_raw = call(
                        STRUCTURAL_REPAIR_PROMPT,
                        "Structural Repair",
                        invariants=stages["invariants"],
                        code=stages["snippet"],
                        trace=stages["trace_analysis"],
                        failures=failure_str,
                        starter_code=problem.starter_code
                    )

                # Save final snippet from repair
                match = re.search(r"```python\n(.*?)\n```", repair_raw, re.DOTALL)
                stages["structural_repair"] = (
                    match.group(1).strip() if match else repair_raw.strip()
                )
                stages["snippet"] = stages["structural_repair"]
            else:
                p_console.print("[bold green]All public tests passed![/bold green]")
                stages["execution_feedback"] = "All public tests passed."
                # snippet remains the mental_refinement version
        except Exception as e:
            p_console.print(f"[bold red]Execution system error: {e}[/bold red]")
            stages["snippet"] = stages["initial_code"]

    # Final Syntax Check
    if passed and not python_lang.parser.is_syntax_valid(stages["snippet"]):
        p_console.print("[bold red]Final code has invalid syntax.[/bold red]")
        passed = False

    status = (
        "[bold green]PASSED[/bold green]" if passed else "[bold red]FAILED[/bold red]"
    )
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
    output_dir: Path = typer.Option(Path("logs/invariant_exec")),
    workers: int = typer.Option(16),
) -> None:
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    log_file = output_dir / f"{run_id}_invariant_exec.txt"
    jsonl_file = output_dir / f"{run_id}_solutions.jsonl"
    logger_ = InvariantExecLogger(log_file)
    jsonl_logger = JsonlLogger(jsonl_file)

    console.print(
        Panel(
            f"Starting Invariant-Execution Run: [bold cyan]{run_id}[/bold cyan]\n"
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

    problems = load_code_generation_dataset(
        release_version=version,
        difficulty=Difficulty(difficulty.lower()),
        start_date=start_date,
        end_date=end_date,
    )
    if count:
        problems = problems[:count]

    total = len(problems)
    results = []
    solved = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing Invariant-Exec...", total=total)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # We must be careful about concurrency with the execution system if it uses its own multiprocessing
            # But evaluate.py does it similarly.
            # Actually, evaluate.py runs LCB problems sequentially in a loop, but execution system itself is parallel.
            # If I use ThreadPoolExecutor here, I might oversubscribe CPUs.
            # I'll set workers=1 for the outer loop if I want execution system to handle parallelism,
            # OR I'll let the user decide. Usually, for this kind of script, we process problems one by one
            # because each problem has multiple LLM calls.

            futs = [
                executor.submit(
                    process_problem,
                    p,
                    client,
                    python_lang,
                    execution_system,
                    i,
                    total,
                    progress,
                    task,
                )
                for i, p in enumerate(problems)
            ]
            for fut in futs:
                try:
                    q_id, passed, log, out = fut.result()
                    results.append((q_id, passed))
                    if passed:
                        solved += 1
                    logger_.log_problem(log)
                    jsonl_logger.log(
                        {
                            "question_id": q_id,
                            "snippet": out["snippet"],
                            **out,
                            "status": None,
                        }
                    )
                except Exception as e:
                    console.print(f"[bold red]Worker Error: {e}[/bold red]")

    table = Table(title="Invariant-Exec Results")
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
