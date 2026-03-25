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

INITIAL_PLAN_PROMPT_TEMPLATE = """Problem Description:
{question_content}

Starter Code:
{starter_code}

Please provide a detailed step-by-step implementation plan for the problem described above.
Your plan should be logic-oriented and map out how to handle edge cases and the core algorithmic steps.
"""

PLAN_REFINEMENT_PROMPT_TEMPLATE = """Problem Description:
{question_content}

Starter Code:
{starter_code}

Current Implementation Plan:
{previous_plan}

Based on the problem description and the current implementation plan above, please provide a refined and improved implementation plan.
Focus on:
1. Improving the logic for clarity and efficiency.
2. Adding more detail to algorithmic steps.
3. Ensuring all edge cases mentioned in the problem description are addressed.
"""

SOLUTION_PROMPT_TEMPLATE = """Problem Description:
{question_content}

Implementation Plan:
{plan}

Starter Code:
{starter_code}

Based on the problem description and the implementation plan above, please provide a complete Python solution.
Wrap your code in ```python blocks. Strictly stick to the Starter Code format. 
"""

PROMPT_SIMULATE = """Problem Description:
{question_content}

Current Code Solution:
```python
{current_code}
```

Please perform a "mental simulation" (dry run) of the current code. 
1. Come up with a concrete, non-trivial sample input that strictly follows the bounds and logic described in the problem description.
2. Trace the execution of the code line-by-line using that sample input.
3. Keep track of the values of key variables at each step.
4. Conclude whether the code produces the correct output for this specific input, and briefly note any apparent logic flaws or bugs discovered during the trace.
"""

PROMPT_REFINE_PLAN_WITH_SIMULATION = """Problem Description:
{question_content}

Starter Code:
{starter_code}

Current Implementation Plan:
{plan}

Mental Simulation Trace of Current Code:
{simulation_trace}

Based on the mental simulation trace and the problem description, please provide a refined and improved implementation plan.
Focus on:
1. Fixing any bugs or logic errors identified during the mental simulation.
2. Adjusting the algorithmic steps to better handle the edge cases discovered.
3. Improving the overall logic for clarity and efficiency.
"""

CODE_REFINEMENT_PROMPT_TEMPLATE = """Problem Description:
{question_content}

Implementation Plan:
{plan}

Starter Code:
{starter_code}

Current Code Solution:
```python
{current_code}
```

Based on the implementation plan and the current code solution, please provide a refined and improved complete Python solution.
Focus on:
1. Fixing any remaining bugs or edge cases.
2. Enhancing code clarity and style.
3. Ensuring perfect alignment with the implementation plan.

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
    plan_iterations: int,
    sim_iterations: int,
    final_refine_iterations: int,
) -> Tuple[str, bool, str, str, str, List[str], List[str], List[str]]:
    """Process a single problem with example-free modification and simulation-based plan/code refinement loop."""

    log_stream = io.StringIO()
    p_console = Console(file=log_stream, force_terminal=True, width=120)

    plans = []
    snippets = []
    simulations = []

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

    modified_problem = ""
    problem_passed = False

    try:
        # Step 1: Modify Problem
        p_console.print("\n[bold cyan]Step 1: Removing Examples and Clarifying Format...[/bold cyan]")
        modify_p = PROMPT_MODIFY_PROBLEM.format(question_content=problem.question_content)
        modified_problem = generate(modify_p)
        p_console.print(Panel(modified_problem, title="Modified Problem Content", border_style="green"))

        # Step 2: Initial Plan using modified problem
        p_console.print("\n[bold cyan]Step 2: Generating Initial Plan...[/bold cyan]")
        initial_plan_p = INITIAL_PLAN_PROMPT_TEMPLATE.format(
            question_content=modified_problem,
            starter_code=problem.starter_code
        )
        current_plan = generate(initial_plan_p)
        plans.append(current_plan)
        p_console.print(Panel(Syntax(current_plan, "markdown", theme="monokai", padding=1), title="Plan v1"))

        # Step 3: Plan Refinement
        for i in range(plan_iterations):
            p_console.print(f"\n[bold cyan]Step 3.{i+1}: Refining Plan...[/bold cyan]")
            refine_plan_p = PLAN_REFINEMENT_PROMPT_TEMPLATE.format(
                question_content=modified_problem,
                starter_code=problem.starter_code,
                previous_plan=current_plan
            )
            current_plan = generate(refine_plan_p)
            plans.append(current_plan)
            p_console.print(Panel(Syntax(current_plan, "markdown", theme="monokai", padding=1), title=f"Plan v{i+2}"))

        # Step 4: Initial Code Generation
        p_console.print("\n[bold cyan]Step 4: Generating Initial Code from Plan...[/bold cyan]")
        solution_p = SOLUTION_PROMPT_TEMPLATE.format(
            question_content=modified_problem,
            plan=current_plan,
            starter_code=problem.starter_code
        )
        res_code = generate(solution_p)
        current_code = extract_code(res_code)
        snippets.append(current_code)
        p_console.print(Panel(Syntax(current_code, "python", theme="monokai", padding=1), title="Code v1"))

        # Step 5: Simulate and Refine Plan Loop
        for i in range(sim_iterations):
            # 5a: Simulate
            p_console.print(f"\n[bold cyan]Step 5.{i+1}a: Mentally Simulating Code v{i+1}...[/bold cyan]")
            simulate_p = PROMPT_SIMULATE.format(
                question_content=modified_problem,
                current_code=current_code
            )
            simulation_trace = generate(simulate_p)
            simulations.append(simulation_trace)
            p_console.print(Panel(Syntax(simulation_trace, "markdown", theme="monokai", padding=1), title=f"Simulation Trace v{i+1}"))

            # 5b: Refine Plan with Simulation
            p_console.print(f"\n[bold cyan]Step 5.{i+1}b: Refining Plan Based on Simulation...[/bold cyan]")
            refine_plan_sim_p = PROMPT_REFINE_PLAN_WITH_SIMULATION.format(
                question_content=modified_problem,
                starter_code=problem.starter_code,
                plan=current_plan,
                simulation_trace=simulation_trace
            )
            current_plan = generate(refine_plan_sim_p)
            plans.append(current_plan)
            p_console.print(Panel(Syntax(current_plan, "markdown", theme="monokai", padding=1), title=f"Plan (Sim-Refined v{i+1})"))

            # 5c: Generate Code again from refined plan
            p_console.print(f"\n[bold cyan]Step 5.{i+1}c: Generating Code from Refined Plan...[/bold cyan]")
            solution_p = SOLUTION_PROMPT_TEMPLATE.format(
                question_content=modified_problem,
                plan=current_plan,
                starter_code=problem.starter_code
            )
            res_code = generate(solution_p)
            current_code = extract_code(res_code)
            snippets.append(current_code)
            p_console.print(Panel(Syntax(current_code, "python", theme="monokai", padding=1), title=f"Code v{i+2}"))

        # Step 6: Final Code Refinement
        for i in range(final_refine_iterations):
            p_console.print(f"\n[bold cyan]Step 6.{i+1}: Final Code Refinement...[/bold cyan]")
            code_refine_p = CODE_REFINEMENT_PROMPT_TEMPLATE.format(
                question_content=modified_problem,
                plan=current_plan,
                starter_code=problem.starter_code,
                current_code=current_code
            )
            res_refine = generate(code_refine_p)
            current_code = extract_code(res_refine)
            snippets.append(current_code)
            p_console.print(Panel(Syntax(current_code, "python", theme="monokai", padding=1), title=f"Code (Final Refined v{i+1})"))

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
        snippets[-1] if snippets else "",
        modified_problem,
        snippets,
        plans,
        simulations,
    )


@app.command()
def run(
    llm: Path = typer.Option(Path("configs/llm/gpt-5-mini.yaml"), help="Path to LLM config YAML"),
    count: Optional[int] = typer.Option(None, help="Number of problems to process"),
    difficulty: str = typer.Option("hard", help="Difficulty of problems to load (easy, medium, hard)"),
    version: str = typer.Option("release_v6", help="LCB dataset version"),
    start_date: Optional[str] = typer.Option("2025-03-01", help="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = typer.Option("2025-05-10", help="End date (YYYY-MM-DD)"),
    output_dir: Path = typer.Option(Path("logs/no_public_plan_rewrite_simulate"), help="Directory to save logs"),
    workers: int = typer.Option(8, help="Number of parallel workers"),
    plan_iterations: int = typer.Option(2, help="Number of plan refinement iterations"),
    sim_iterations: int = typer.Option(2, help="Number of simulation & code refinement loop iterations"),
    final_refine_iterations: int = typer.Option(1, help="Number of final code refinement iterations"),
) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    log_file = output_dir / f"{run_id}_no_public_plan_rewrite_simulate.txt"
    jsonl_file = output_dir / f"{run_id}_solutions.jsonl"
    config_file = output_dir / f"{run_id}_config.txt"
    
    logger_txt = DirectLogger(log_file)
    jsonl_logger = JsonlLogger(jsonl_file)

    # Initial Run Configuration Panel
    run_config_panel = Panel(
        Group(
            f"Starting No-Public-Plan-Rewrite-Simulate Gen Run: [bold cyan]{run_id}[/bold cyan]",
            f"Log file: [yellow]{log_file}[/yellow]",
            f"JSONL file: [yellow]{jsonl_file}[/yellow]",
            f"Config file: [yellow]{config_file}[/yellow]",
            "",
            f"Workers: [green]{workers}[/green]",
            f"Plan Iterations: [green]{plan_iterations}[/green]",
            f"Simulation Iterations: [green]{sim_iterations}[/green]",
            f"Final Refine Iterations: [green]{final_refine_iterations}[/green]",
            "",
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
        border_style="bold green"
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
    generated_problems = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task("[cyan]Processing no_public_plan_rewrite_simulate workflow...", total=total_problems)

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
                    sim_iterations,
                    final_refine_iterations,
                ): problem
                for i, problem in enumerate(problems)
            }

            for fut in future_to_problem:
                try:
                    q_id, passed, log_block, snippet, modified_prob, all_snippets, all_plans, all_sims = fut.result()
                    results.append((q_id, passed))
                    if passed:
                        generated_problems += 1
                    logger_txt.log_problem_block(log_block)
                    jsonl_logger.log(
                        {
                            "question_id": q_id,
                            "snippet": snippet,
                            "modified_problem": modified_prob,
                            "all_snippets": all_snippets,
                            "all_plans": all_plans,
                            "all_simulations": all_sims,
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
    table = Table(title="No-Public-Plan-Rewrite-Simulate Results")
    table.add_column("Question ID", style="cyan")
    table.add_column("Status", style="bold")

    for q_id, passed in results:
        status_text = "[green]Generated[/green]" if passed else "[red]Failed[/red]"
        table.add_row(q_id, status_text)

    console.print(table)

    summary_panel = Panel(
        Group(
            f"Total Problems: {total_problems}",
            f"Generated: [bold green]{generated_problems}[/bold green]",
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

    # Append to log file
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 120 + "\n")
        f.write("FINAL SUMMARY\n")
        f.write("=" * 120 + "\n")
        f.write(final_output)
        f.write(f"\nRun completed at {datetime.now()}\n")
        
    # Append to config file
    with open(config_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 120 + "\n")
        f.write("LLM USAGE AND RUN SUMMARY\n")
        f.write("=" * 120 + "\n")
        f.write(final_output)
        f.write(f"\nRun completed at {datetime.now()}\n")


if __name__ == "__main__":
    app()
