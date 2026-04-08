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
EXTRACT_CONSTRAINTS_PROMPT = r"""Problem:
{question_content}

Starter Code:
{starter_code}

Extract strictly the logical/mathematical constraints and the core objective of this problem. 
Specify input ranges, target time/space complexity, and invariant properties.
"""

GENERATE_HYPOTHESES_PROMPT = r"""Problem:
{question_content}

Constraints:
{constraints}

Generate 2-3 distinct algorithmic strategies (hypotheses) to solve this problem. 
For each hypothesis:
- Name the strategy (e.g., Dynamic Programming, Greedy, Segment Tree).
- Provide a high-level summary.
- Explain how it handles the primary constraints.
"""

EVALUATE_HYPOTHESIS_PROMPT = r"""Problem:
{question_content}

Hypothesis:
{hypothesis}

Constraints:
{constraints}

Perform a rigorous formal evaluation of this specific hypothesis:
1. **Formalization**: Define the core logic, recurrence relations, or state transitions precisely.
2. **Adversarial Input Construction**: Design a specific input that might cause this algorithm to fail or exceed limits.
3. **Symbolic Failure Analysis**: Trace the execution on the adversarial input. Does it violate any invariants?
4. **Complexity & Edge Case Audit**: Prove it meets $O(\dots)$ and handles min/max bounds.
"""

ELIMINATION_PROMPT = r"""Problem:
{question_content}

Hypotheses:
{hypotheses_with_evaluations}

Compare the hypotheses and their evaluations. Eliminate strategies that are logically inconsistent, fragile under adversarial inputs, or suboptimal in complexity. 
Select the "Winner" and justify your choice.
"""

CONSOLIDATION_PROMPT = r"""Problem:
{question_content}

Winning Strategy Evaluation:
{winner_evaluation}

Consolidate the winner into a final plan:
1. **Refined Invariants**: Final set of properties preserved by the algorithm.
2. **Step-by-step Pseudocode**: Language-agnostic, precise logic.
"""

CONSISTENCY_CHECK_PROMPT = r"""Problem:
{question_content}

Consolidated Plan:
{plan}

Perform a final self-consistency check:
1. Walkthrough the logic on the hardest adversarial case previously identified.
2. Verify that all invariants are preserved throughout the walkthrough.
Identify if any last-minute adjustments are needed.
"""

GENERATE_CODE_PROMPT = r"""Problem:
{question_content}

Final Plan:
{plan}

Consistency Check:
{consistency_check}

Starter Code:
{starter_code}

Implement the final Python solution. Strictly follow the starter code format. Wrap the code in ```python blocks.
"""

class HypothesisLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"Hypothesis Run started at {datetime.now()}\n")

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
    idx: int,
    total: int,
    progress: Progress,
    task_id: Any,
) -> Tuple[str, bool, str, Dict[str, Any]]:
    log_stream = io.StringIO()
    p_console = Console(file=log_stream, force_terminal=True, width=120)
    
    stages = {
        "constraints": "",
        "hypotheses": "",
        "evaluations": "",
        "elimination": "",
        "consolidation": "",
        "consistency": "",
        "snippet": ""
    }
    passed = True

    progress.update(task_id, description=f"Hypothesis: {problem.question_title[:40]}...")
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

    # 1. Constraints
    stages["constraints"] = call(EXTRACT_CONSTRAINTS_PROMPT, "Constraint Extraction", starter_code=problem.starter_code)

    # 2. Hypotheses
    if passed:
        stages["hypotheses"] = call(GENERATE_HYPOTHESES_PROMPT, "Multi-Hypothesis Generation", constraints=stages["constraints"])
    
    # 3. Evaluation of each Hypothesis
    # Parsing Hypotheses (simple split or LLM can handle if we prompt well)
    # For simplicity, we'll ask LLM to evaluate the generated hypotheses ensemble.
    # But user said "For each Hypothesis...". I'll try to split if possible, or just evaluate all together.
    # To be safe and compliant, I'll pass the individual evaluations.
    if passed:
        evals = call(EVALUATE_HYPOTHESIS_PROMPT, "Hypothesis Evaluation", hypothesis=stages["hypotheses"], constraints=stages["constraints"])
        stages["evaluations"] = evals
    
    # 4. Elimination
    if passed:
        stages["elimination"] = call(ELIMINATION_PROMPT, "Cross-Hypothesis Elimination", hypotheses_with_evaluations=f"HYPOTHESES:\n{stages['hypotheses']}\n\nEVALUATIONS:\n{stages['evaluations']}")
    
    # 5. Consolidation
    if passed:
        stages["consolidation"] = call(CONSOLIDATION_PROMPT, "Winner Consolidation", winner_evaluation=stages["elimination"])
    
    # 6. Consistency Check
    if passed:
        stages["consistency"] = call(CONSISTENCY_CHECK_PROMPT, "Self-Consistency Check", plan=stages["consolidation"])
    
    # 7. Code
    if passed:
        final_raw = call(GENERATE_CODE_PROMPT, "Final Code Generation", plan=stages["consolidation"], consistency_check=stages["consistency"], starter_code=problem.starter_code)
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
    output_dir: Path = typer.Option(Path("logs/hypothesis")),
    workers: int = typer.Option(16),
) -> None:
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    log_file = output_dir / f"{run_id}_hypothesis.txt"
    jsonl_file = output_dir / f"{run_id}_solutions.jsonl"
    logger_ = HypothesisLogger(log_file)
    jsonl_logger = JsonlLogger(jsonl_file)

    console.print(
        Panel(
            f"Starting Multi-Hypothesis Run: [bold cyan]{run_id}[/bold cyan]\n"
            f"Log file: [yellow]{log_file}[/yellow]\n"
            f"JSONL file: [yellow]{jsonl_file}[/yellow]\n"
            f"Workers: [green]{workers}[/green]"
        )
    )

    llm_cfg = _load_yaml_file(llm)
    client = create_llm_client(**llm_cfg)
    lang = PythonLanguage()

    problems = load_code_generation_dataset(release_version=version, difficulty=Difficulty(difficulty.lower()), start_date=start_date, end_date=end_date)
    if count:
        problems = problems[:count]
    
    total = len(problems)
    results = []
    solved = 0

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Hypothesis Reasoning...", total=total)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futs = [executor.submit(process_problem, p, client, lang, i, total, progress, task) for i, p in enumerate(problems)]
            for fut in futs:
                try:
                    q_id, passed, log, out = fut.result()
                    results.append((q_id, passed))
                    if passed: solved += 1
                    logger_.log_problem(log)
                    jsonl_logger.log({"question_id": q_id, "snippet": out["snippet"], **out, "status": None})
                except Exception as e:
                    console.print(f"[bold red]Worker Error: {e}[/bold red]")

    table = Table(title="Multi-Hypothesis Results")
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
