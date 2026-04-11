import json
import re
from pathlib import Path
from typing import Any, Optional, cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

app = typer.Typer()


def load_json(file_path: Path) -> dict[str, Any]:
    with open(file_path, "r") as f:
        return cast(dict[str, Any], json.load(f))


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


@app.command()
def cross_check(
    index: Optional[int] = typer.Option(None, help="Index of the entry to view."),
    id: Optional[str] = typer.Option(
        None, help="Question ID (e.g., HumanEval/31) to view."
    ),
    count: int = typer.Option(1, help="Number of entries to show starting from index."),
    output: Optional[Path] = typer.Option(None, help="Path to save the output text file.")
) -> None:
    console = Console(record=True, width=150) if output else Console()

    raw_path = Path("/Users/kaushitha/Documents/APR/data/humaneval_bal_raw.json")
    jsonl_path = Path("/Users/kaushitha/Documents/APR/data/humaneval_ballerina.jsonl")

    if not raw_path.exists() or not jsonl_path.exists():
        console.print("[red]Error: Files not found at expected locations.[/red]")
        return

    raw_data = load_json(raw_path)
    jsonl_data = load_jsonl(jsonl_path)

    entries_to_show = []

    if id:
        target_id_num = id.split("/")[-1] if "/" in id else id
        raw_match = next(
            (x for x in raw_data if x.get("id", "").endswith(f"-{target_id_num}")), None
        )
        jsonl_match = next(
            (
                x
                for x in jsonl_data
                if x.get("question_id", "").endswith(f"/{target_id_num}")
            ),
            None,
        )
        if raw_match and jsonl_match:
            entries_to_show.append((raw_match, jsonl_match))
    elif index is not None:
        start = index
        end = min(start + count, len(raw_data), len(jsonl_data))
        for i in range(start, end):
            entries_to_show.append((raw_data[i], jsonl_data[i]))
    else:
        end = min(count, len(raw_data), len(jsonl_data))
        for i in range(0, end):
            entries_to_show.append((raw_data[i], jsonl_data[i]))

    if not entries_to_show:
        console.print("[yellow]No matching entries found.[/yellow]")
        return

    for raw, jsl in entries_to_show:
        console.rule(f"[bold magenta]Comparison: {raw.get('id', 'N/A')} vs {jsl.get('question_id', 'N/A')}[/bold magenta]")

        # --- Section 1: Problem Context ---
        prompt_content = raw.get("prompt", "")
        starter_content = jsl.get("starter_code", "")
        public_tests_content = json.dumps(jsl.get("public_test_cases", []), indent=2)

        context_table = Table(show_header=False, box=None, expand=True)
        context_table.add_row(Panel(Syntax(prompt_content, "ballerina", theme="monokai", padding=1), title="[bold blue]Prompt (Raw Source)[/bold blue]", border_style="blue"))
        context_table.add_row(Panel(Syntax(starter_content, "ballerina", theme="monokai", padding=1), title="[bold yellow]Starter Code (JSONL)[/bold yellow]", border_style="yellow"))
        context_table.add_row(Panel(Syntax(public_tests_content, "json", theme="monokai", padding=1), title="[bold cyan]Public Test Cases (JSONL)[/bold cyan]", border_style="cyan"))
        
        console.print(Panel(context_table, title="[bold white]Problem Context[/bold white]", border_style="white"))

        # --- Section 2: Private Test Interleaving ---
        console.print("\n[bold green]Private Test Case Verification (Interleaved)[/bold green]")
        
        raw_test_str = raw.get("test", "")
        # Split by @test:Config and keep the tag with the block
        blocks_raw = re.split(r'(?=@test:Config)', raw_test_str)
        
        # Filter: only keep blocks that actually contain a test configuration
        raw_test_blocks = [b.strip() for b in blocks_raw if "@test:Config" in b]

        private_cases = jsl.get("private_test_cases", [])
        
        max_len = max(len(raw_test_blocks), len(private_cases))
        
        from rich.console import Group
        for i in range(max_len):
            raw_block = raw_test_blocks[i] if i < len(raw_test_blocks) else None
            jsl_case = private_cases[i] if i < len(private_cases) else None
            
            components = []
            if raw_block:
                components.append(Panel(Syntax(raw_block, "ballerina", theme="monokai", padding=1), 
                                       title="[bold green]Raw Source Test Code[/bold green]", 
                                       border_style="green"))
            else:
                components.append(Panel("[dim italic]No corresponding raw test block found[/dim italic]", 
                                       title="[bold red]Raw Source Test Code[/bold red]", 
                                       border_style="red"))

            if jsl_case:
                components.append(Panel(Syntax(json.dumps(jsl_case, indent=2), "json", theme="monokai", padding=1), 
                                       title="[bold magenta]JSONL Private Test Data[/bold magenta]", 
                                       border_style="magenta"))
            else:
                components.append(Panel("[dim italic]No corresponding JSONL private test data found[/dim italic]", 
                                       title="[bold red]JSONL Private Test Data[/bold red]", 
                                       border_style="red"))

            console.print(Panel(Group(*components), title=f"Private Test Case #{i}", border_style="white"))

        console.print("\n" + "=" * console.width + "\n")

    if output:
        console.save_text(str(output))
        print(f"Exported alignment report to {output}")


if __name__ == "__main__":
    app()
