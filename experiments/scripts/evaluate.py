import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from coevolution.core.individual import CodeIndividual, TestIndividual
from coevolution.core.interfaces import OPERATION_INITIAL
from coevolution.core.population import CodePopulation, TestPopulation
from coevolution.dataset.lcb import Difficulty, load_code_generation_dataset
from coevolution.services.execution import ExecutionSystem
from infrastructure.languages.python.adapter import PythonLanguage
from infrastructure.sandbox.types import SandboxConfig

app = typer.Typer()
console = Console()


def load_solutions(jsonl_path: Path) -> List[Dict[str, Any]]:
    solutions = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                solutions.append(json.loads(line))
    return solutions


def save_solutions(jsonl_path: Path, solutions: List[Dict[str, Any]]) -> None:
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sol in solutions:
            f.write(json.dumps(sol) + "\n")


@app.command()
def evaluate(
    jsonl_file: Path = typer.Argument(
        ..., help="Path to the JSONL file with generated code snippets"
    ),
    difficulty: Optional[str] = typer.Option(
        None, help="Difficulty of problems in the LCB dataset"
    ),
    version: str = typer.Option("release_v6", help="LCB dataset version"),
    start_date: Optional[str] = typer.Option(
        "2025-03-01", help="Start date (YYYY-MM-DD)"
    ),
    end_date: Optional[str] = typer.Option("2025-05-10", help="End date (YYYY-MM-DD)"),
    workers: int = typer.Option(
        4, help="Number of parallel workers for execution system"
    ),
) -> None:
    """
    Evaluate code snippets from a JSONL file against the LCB dataset.
    The JSONL file should contain 'question_id' and 'snippet' fields.
    """
    if not jsonl_file.exists():
        console.print(f"[bold red]File not found: {jsonl_file}[/bold red]")
        raise typer.Exit(1)

    console.print(f"[bold cyan]Loading solutions from {jsonl_file}[/bold cyan]")
    solutions = load_solutions(jsonl_file)

    console.print("[bold cyan]Loading LCB dataset for evaluation...[/bold cyan]")
    diff_enum = Difficulty(difficulty.lower()) if difficulty else None
    problems = load_code_generation_dataset(
        release_version=version,
        difficulty=diff_enum,
        start_date=start_date,
        end_date=end_date,
    )
    problem_map = {p.question_id: p for p in problems}

    # Setup Execution System Components
    python_lang = PythonLanguage()
    sandbox_config = SandboxConfig(
        timeout=30,
        max_memory_mb=1024 * 2,
    )
    execution_system = ExecutionSystem(
        sandbox_config=sandbox_config,
        composer=python_lang.composer,
        runtime=python_lang.runtime,
        analyzer=python_lang.analyzer,
        enable_multiprocessing=True,
        cpu_workers=workers,
    )

    console.print(
        f"[bold green]Starting evaluation of {len(solutions)} solutions...[/bold green]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Evaluating...", total=len(solutions))

        for sol in solutions:
            # Skip if already evaluated AND error logs are present
            if sol.get("status") is not None and "public_error_logs" in sol:
                progress.advance(task)
                continue

            q_id = sol["question_id"]
            snippet = sol["snippet"]

            # Initialize error logs
            sol["public_error_logs"] = []
            sol["private_error_logs"] = []

            if not snippet:
                sol["status"] = "fail (no code)"
                progress.advance(task)
                continue

            problem = problem_map.get(q_id)
            if not problem:
                sol["status"] = "error (problem not found)"
                progress.advance(task)
                continue

            # 0. Check Syntax
            if not python_lang.parser.is_syntax_valid(snippet):
                sol["status"] = "fail (syntax error)"
                sol["pass_rate"] = "0/0"
                progress.advance(task)
                continue

            # 1. Setup Code Individual
            code_ind = CodeIndividual(
                snippet=python_lang.parser.remove_main_block(snippet),
                probability=1.0,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
            )
            code_pop = CodePopulation(individuals=[code_ind], generation=0)

            # 2. Setup Test Populations
            public_test_functions = [
                python_lang.composer.generate_test_case(
                    tc.input, tc.output, problem.starter_code, idx + 1
                )
                for idx, tc in enumerate(problem.public_test_cases)
            ]
            private_test_functions = [
                python_lang.composer.generate_test_case(
                    tc.input, tc.output, problem.starter_code, idx + 1
                )
                for idx, tc in enumerate(problem.private_test_cases)
            ]

            all_test_individuals = [
                TestIndividual(
                    snippet=func,
                    probability=1.0,
                    creation_op=OPERATION_INITIAL,
                    generation_born=0,
                    parents={},
                )
                for func in public_test_functions + private_test_functions
            ]
            all_tests_pop = TestPopulation(
                individuals=all_test_individuals, generation=0
            )

            # 3. Execute
            try:
                interaction = execution_system.execute_tests(code_pop, all_tests_pop)

                # Check results
                num_public = len(public_test_functions)
                num_private = len(private_test_functions)

                # interaction.observation_matrix is (num_codes, num_tests)
                # Here (1, num_tests)
                results = interaction.observation_matrix[0]

                passed_public = int(results[:num_public].sum()) if num_public > 0 else 0
                passed_private = (
                    int(results[num_public:].sum()) if num_private > 0 else 0
                )

                sol["public_pass_rate"] = f"{passed_public}/{num_public}"
                sol["private_pass_rate"] = f"{passed_private}/{num_private}"

                # Capture error logs for failed tests
                execution_results = interaction.execution_results.results[code_ind.id]
                for idx, test_ind in enumerate(all_test_individuals):
                    test_res = execution_results.get(test_ind.id)
                    if test_res and test_res.status != "passed" and test_res.error_log:
                        if idx < num_public:
                            sol["public_error_logs"].append(test_res.error_log)
                        else:
                            sol["private_error_logs"].append(test_res.error_log)

                # Status ONLY depends on private tests
                if num_private == 0:
                    sol["status"] = "pass"  # Fallback if no private tests
                else:
                    sol["status"] = "pass" if passed_private == num_private else "fail"

                # Log warnings for any timeouts in the execution results
                for t_id, t_res in interaction.execution_results.results[
                    code_ind.id
                ].items():
                    if t_res.status == "error" and t_res.error_log:
                        err_lower = t_res.error_log.lower()
                        if "timeout" in err_lower or "timed out" in err_lower:
                            logger.warning(
                                f"Timeout detected for problem {q_id}, test {t_id}: {t_res.error_log}"
                            )

            except Exception as e:
                logger.error(f"Error evaluating {q_id}: {e}")
                sol["status"] = "fail (error)"
                sol["public_pass_rate"] = "0/0"
                sol["private_pass_rate"] = "0/0"

            progress.advance(task)

    # Save results
    save_solutions(jsonl_file, solutions)
    console.print(
        f"\n[bold green]Evaluation complete! Results updated in {jsonl_file}[/bold green]"
    )

    # Results Table
    results_table = Table(title="Evaluation Results")
    results_table.add_column("Question ID", style="cyan")
    results_table.add_column("Public Pass Rate", style="magenta")
    results_table.add_column("Private Pass Rate", style="green")
    results_table.add_column("Status", style="bold")

    for sol in solutions:
        results_table.add_row(
            str(sol.get("question_id")),
            sol.get("public_pass_rate", "N/A"),
            sol.get("private_pass_rate", "N/A"),
            sol.get("status", "N/A"),
        )
    console.print(results_table)

    # Summary by Difficulty
    stats = {
        Difficulty.EASY: {"pass": 0, "total": 0, "pppf": 0},
        Difficulty.MEDIUM: {"pass": 0, "total": 0, "pppf": 0},
        Difficulty.HARD: {"pass": 0, "total": 0, "pppf": 0},
    }

    pppf_total = 0
    public_pass_total = 0
    prpuf_total = 0
    prpuf_ids = []

    for sol in solutions:
        q_id = sol["question_id"]
        problem = problem_map.get(q_id)
        if problem:
            diff = problem.difficulty
            if diff in stats:
                stats[diff]["total"] += 1

                # Check status
                status = sol.get("status")
                if status == "pass":
                    stats[diff]["pass"] += 1

                # Check PPPF: Passed all public, but not all private
                pub_pass, pub_total = 0, 0
                priv_pass, priv_total = 0, 0

                pub_rate = sol.get("public_pass_rate", "0/0")
                priv_rate = sol.get("private_pass_rate", "0/0")

                if "/" in pub_rate:
                    pub_pass, pub_total = map(int, pub_rate.split("/"))
                if "/" in priv_rate:
                    priv_pass, priv_total = map(int, priv_rate.split("/"))

                if pub_total > 0 and pub_pass == pub_total:
                    public_pass_total += 1
                    if priv_total > 0 and priv_pass < priv_total:
                        stats[diff]["pppf"] += 1
                        pppf_total += 1

                # Check Private Pass, Public Fail: Passed all private, but not all public
                if priv_total > 0 and priv_pass == priv_total:
                    if pub_total > 0 and pub_pass < pub_total:
                        prpuf_total += 1
                        prpuf_ids.append(q_id)

    summary_table = Table(title="Summary by Difficulty")
    summary_table.add_column("Difficulty", style="cyan")
    summary_table.add_column("Pass Rate", style="bold yellow")
    summary_table.add_column("Percentage", style="bold green")
    summary_table.add_column("PPPF Cases", style="bold red")

    total_pass = 0
    total_count = len(solutions)

    for diff in [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]:
        s = stats[diff]
        if s["total"] > 0:
            pass_rate = f"{s['pass']}/{s['total']}"
            percentage = f"{(s['pass'] / s['total']) * 100:.2f}%"
            summary_table.add_row(
                diff.value.capitalize(), pass_rate, percentage, str(s["pppf"])
            )
            total_pass += s["pass"]

    console.print(summary_table)

    overall_percentage = (total_pass / total_count * 100) if total_count > 0 else 0
    pppf_rate = (pppf_total / public_pass_total * 100) if public_pass_total > 0 else 0

    summary_content = [
        f"Overall Pass Rate: [bold green]{total_pass}/{total_count}[/bold green] ([bold cyan]{overall_percentage:.2f}%[/bold cyan])",
        "",
        "[bold red]Public Pass, Private Fail (PPPF) Analysis:[/bold red]",
        f"  Total PPPF Cases: [bold red]{pppf_total}[/bold red]",
        f"  PPPF Rate (of Public Pass): [bold yellow]{pppf_rate:.2f}%[/bold yellow] ({pppf_total}/{public_pass_total})",
        "",
        "[bold yellow]Private Pass, Public Fail Analysis:[/bold yellow]",
        f"  Total Cases: [bold yellow]{prpuf_total}[/bold yellow]",
    ]

    if prpuf_ids:
        summary_content.append(
            f"  [bold red]WARNING:[/bold red] Rare cases found in: {', '.join(map(str, prpuf_ids))}"
        )

    console.print(
        Panel(
            "\n".join(summary_content),
            title="Final Summary",
            border_style="bold green",
        )
    )


if __name__ == "__main__":
    app()
