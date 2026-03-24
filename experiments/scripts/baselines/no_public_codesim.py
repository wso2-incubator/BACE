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

PROMPT_MODIFY_PROBLEM = """Original Problem Description:
{question_content}

Please rewrite this problem description by:
1. Removing all sample examples, test cases, and "Example 1", "Example 2" sections.
2. Replacing those sections with a clear, explicit description of the Input Format and Output Format.
3. Keep the core problem statement, constraints, and instructions intact.

Output only the modified problem description.
"""

PROMPT_PLAN_WITH_RECALL = """You are a programmer tasked with generating appropriate plan to solve a given problem using the python programming language.

## Problem
{problem}

Starter Code:
{starter_code}

**Expected Output:** Your response must be structured as follows:

### Problem Understanding
- Think about the original problem. Develop an initial understanding about the problem.

### Recall Example Problem
Recall a relevant and distinct problems (different from problem mentioned above) and
- Describe it
- Generate python code step by step to solve that problem
- Discuss the algorithm to solve this problem
- Finally generate a planning to solve that problem

### Algorithm to solve the original problem
- Write down the algorithm that is well suited for the original problem
- Give some tutorials to about the algorithm for example:
- How to approach this type of algorithm
- Important things to consider

### Plan
- Write down a detailed, step-by-step plan to solve the **original problem**.

-------**Important Instruction:**
- Strictly follow the instructions.
- Do not generate code.
"""

PROMPT_SIMULATE_PLAN = """You are a programmer tasked with verifying a plan to solve a given problem using the **python** programming language.

## Problem
{problem}

### Plan
{plan}

Starter Code:
{starter_code}

**Expected Output:** Your response must be structured as follows:

### Simulation
- Come up with a concrete, non-trivial sample input that strictly follows the constraints of the problem.
- Apply the plan step by step to get the output for this specific input.
- Trace the values and logic flow clearly.
- Reason whether the output is the intended output for the problem.

### Plan Evaluation
- If the plan is very good and you see no improvements, write **No Need to Modify Plan**.
- Otherwise write **Plan Modification Needed** and provide a critique of the flawed steps.
"""

PROMPT_REFINE_PLAN = """You are a programmer tasked with generating appropriate plan to solve a given problem using the **python** programming language. You already have a wrong plan. Correct it so that it can generate correct code.

## Problem
{problem}

### Plan
{plan}

## Plan Critique
{plan_critique}

Starter Code:
{starter_code}

**Expected Output:** Your response must be structured as follows:

### New Plan
- Write down a detailed, step-by-step modified plan to solve the **original problem**.
- Ensure each step logically follows from the previous one.

-------**Important Instruction:**
- Your response must contain only the plan.
- Do not add any explanation.
- Do not generate code.
- Your plan must strictly stick to the starter code format.
"""

PROMPT_GENERATE_CODE = """You are a programmer tasked with solving a given problem using the **python** programming language. See the plan to solve the plan and implement code to solve it.

## Problem
{problem}

### Plan
{plan}

Starter Code:
{starter_code}
-------**Important Instruction:**
- Do not add any explanation.
- The generated **python** code must be inside a triple backtick (```) code block.
- Strictly stick to the Starter Code format.
"""

PROMPT_SIMULATE_CODE = """You are a programmer tasked with verifying the python code used to solve a given problem.

## Problem
{problem}

Starter Code:
{starter_code}

### Plan
{plan}

### Code
```python
{code}
```

**Expected Output:** Your response must be structured as follows:

### Simulation
- Come up with a concrete, non-trivial sample input that strictly follows the constraints of the problem.
- Trace the execution of the code line-by-line using that sample input.
- Keep track of the values of key variables at each step.
- Reason whether the final output produced is the intended output for the problem.

### Code Evaluation
- If the code is very good and you see no improvements, write **No Code Modifications Needed**.
- Otherwise, write **Code Modification Needed** and describe the flaws or bugs discovered during the trace.
"""

PROMPT_REFINE_CODE = """You are a programmer tasked with correcting python code for a given problem.

## Problem
{problem}

Starter Code:
{starter_code}

### Plan
{plan}

### Code
```python
{code}
```

## Code Critique
{code_critique}

**Expected Output:**
- Provide a corrected and improved complete python solution.
- Wrap your code in ```python blocks. Strictly stick to the Starter Code format.
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
    code_iterations: int,
) -> Tuple[str, bool, str, str, str, List[str], List[str], List[str], List[str]]:
    """Process a single problem with iterative Plan-Sim-Refine and Code-Sim-Refine workflow."""

    log_stream = io.StringIO()
    p_console = Console(file=log_stream, force_terminal=True, width=120)

    plans = []
    snippets = []
    plan_simulations = []
    code_simulations = []

    progress.update(
        task_id, description=f"Processing: {problem.question_title[:40]}..."
    )

    p_console.rule(
        f"[bold magenta]Problem {problem_idx + 1}/{total_problems}: {problem.question_title} ({problem.question_id})[/bold magenta]"
    )
    p_console.print(
        Panel(
            problem.question_content,
            title="Original Question Content",
            border_style="blue",
        )
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

    modified_problem = ""
    problem_passed = False

    try:
        # Step 0: Modify Problem
        p_console.print(
            "\n[bold cyan]Step 0: Removing Examples and Clarifying Format...[/bold cyan]"
        )
        modify_p = PROMPT_MODIFY_PROBLEM.format(
            question_content=problem.question_content
        )
        modified_problem = generate(modify_p)
        p_console.print(
            Panel(
                modified_problem, title="Modified Problem Content", border_style="green"
            )
        )

        # Step 1: Initial Plan with Recall
        p_console.print(
            "\n[bold cyan]Step 1: Generating Initial Plan with Recall...[/bold cyan]"
        )
        plan_p = PROMPT_PLAN_WITH_RECALL.format(
            problem=modified_problem, starter_code=problem.starter_code
        )
        current_plan = generate(plan_p)
        plans.append(current_plan)
        p_console.print(
            Panel(
                Syntax(current_plan, "markdown", theme="monokai", padding=1),
                title="Plan v1",
            )
        )

        # Plan Refinement Loop
        for i in range(plan_iterations):
            p_console.print(
                f"\n[bold cyan]Step 2.{i + 1}: Simulating Plan v{i + 1}...[/bold cyan]"
            )
            sim_p = PROMPT_SIMULATE_PLAN.format(
                problem=modified_problem,
                plan=current_plan,
                starter_code=problem.starter_code,
            )
            sim_report = generate(sim_p)
            plan_simulations.append(sim_report)
            p_console.print(
                Panel(
                    Syntax(sim_report, "markdown", theme="monokai", padding=1),
                    title=f"Plan Simulation Report v{i + 1}",
                )
            )

            if "No Need to Modify Plan" in sim_report:
                p_console.print(
                    "[bold green]Plan verified. Moving to code generation.[/bold green]"
                )
                break

            p_console.print(
                f"\n[bold yellow]Step 3.{i + 1}: Refining Plan...[/bold yellow]"
            )
            refine_plan_p = PROMPT_REFINE_PLAN.format(
                problem=modified_problem,
                plan=current_plan,
                starter_code=problem.starter_code,
                plan_critique=sim_report,
            )
            current_plan = generate(refine_plan_p)
            plans.append(current_plan)
            p_console.print(
                Panel(
                    Syntax(current_plan, "markdown", theme="monokai", padding=1),
                    title=f"Plan v{i + 2}",
                )
            )

        # Step 4: Initial Code Generation
        p_console.print(
            "\n[bold cyan]Step 4: Generating Initial Code from Plan...[/bold cyan]"
        )
        code_p = PROMPT_GENERATE_CODE.format(
            problem=modified_problem,
            plan=current_plan,
            starter_code=problem.starter_code,
        )
        res_code = generate(code_p)
        current_code = extract_code(res_code)
        snippets.append(current_code)
        p_console.print(
            Panel(
                Syntax(current_code, "python", theme="monokai", padding=1),
                title="Code v1",
            )
        )

        # Code Refinement Loop
        for i in range(code_iterations):
            p_console.print(
                f"\n[bold cyan]Step 5.{i + 1}: Simulating Code v{i + 1}...[/bold cyan]"
            )
            sim_code_p = PROMPT_SIMULATE_CODE.format(
                problem=modified_problem,
                plan=current_plan,
                code=current_code,
                starter_code=problem.starter_code,
            )
            sim_report = generate(sim_code_p)
            code_simulations.append(sim_report)
            p_console.print(
                Panel(
                    Syntax(sim_report, "markdown", theme="monokai", padding=1),
                    title=f"Code Simulation Report v{i + 1}",
                )
            )

            if "No Code Modifications Needed" in sim_report:
                p_console.print("[bold green]Code verified.[/bold green]")
                break

            p_console.print(
                f"\n[bold yellow]Step 6.{i + 1}: Refining Code...[/bold yellow]"
            )
            refine_code_p = PROMPT_REFINE_CODE.format(
                problem=modified_problem,
                plan=current_plan,
                code=current_code,
                starter_code=problem.starter_code,
                code_critique=sim_report,
            )
            res_refine = generate(refine_code_p)
            current_code = extract_code(res_refine)
            snippets.append(current_code)
            p_console.print(
                Panel(
                    Syntax(current_code, "python", theme="monokai", padding=1),
                    title=f"Code v{i + 2}",
                )
            )

        problem_passed = True

    except Exception as e:
        p_console.print(f"[bold red]Failed during processing: {e}[/bold red]")

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
        snippets[-1] if snippets else "",
        modified_problem,
        snippets,
        plans,
        plan_simulations,
        code_simulations,
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
        Path("logs/no_public_codesim"), help="Directory to save logs"
    ),
    workers: int = typer.Option(8, help="Number of parallel workers"),
    plan_iterations: int = typer.Option(
        2, help="Max number of plan refinement iterations"
    ),
    code_iterations: int = typer.Option(
        2, help="Max number of code refinement iterations"
    ),
) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    log_file = output_dir / f"{run_id}_no_public_codesim.txt"
    jsonl_file = output_dir / f"{run_id}_solutions.jsonl"
    logger_txt = DirectLogger(log_file)
    jsonl_logger = JsonlLogger(jsonl_file)

    console.print(
        Panel(
            f"Starting No-Public-CodeSim Gen Run: [bold cyan]{run_id}[/bold cyan]\n"
            f"Log file: [yellow]{log_file}[/yellow]\n"
            f"JSONL file: [yellow]{jsonl_file}[/yellow]\n"
            f"Workers: [green]{workers}[/green]\n"
            f"Max Plan Iterations: [green]{plan_iterations}[/green]\n"
            f"Max Code Iterations: [green]{code_iterations}[/green]"
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
        overall_task = progress.add_task(
            "[cyan]Processing no_public_codesim workflow...", total=total_problems
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
                    plan_iterations,
                    code_iterations,
                ): problem
                for i, problem in enumerate(problems)
            }

            for fut in future_to_problem:
                try:
                    (
                        q_id,
                        passed,
                        log_block,
                        snippet,
                        modified_prob,
                        all_snippets,
                        all_plans,
                        all_sim_plans,
                        all_sim_codes,
                    ) = fut.result()
                    results.append((q_id, passed))
                    if passed:
                        solved_problems += 1
                    logger_txt.log_problem_block(log_block)
                    jsonl_logger.log(
                        {
                            "question_id": q_id,
                            "snippet": snippet,
                            "modified_problem": modified_prob,
                            "all_snippets": all_snippets,
                            "all_plans": all_plans,
                            "all_plan_simulations": all_sim_plans,
                            "all_code_simulations": all_sim_codes,
                            "status": None,
                        }
                    )
                except Exception as e:
                    console.print(f"[bold red]Exception in worker: {e}[/bold red]")

    # Final result table
    table = Table(title="No-Public-CodeSim Results")
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
