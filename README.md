# BACE: Bayesian Coevolution of Code and Tests

Coevolution of code and tests to improve LLM-driven code generation via Bayesian updates and evolutionary strategies.

## Installation

### From source (development mode)

```bash
cd /path/to/workspace
uv sync
```

## Quick Start

The main entry point is [main.py](main.py). Run an experiment using a YAML config:

```bash
uv run python main.py run --config configs/experiments/default.yaml
```

Override components or subsets via CLI options:

```bash
# Use a different LLM config
uv run python main.py run --config configs/experiments/default.yaml \
 --llm configs/llm/gpt-4.yaml

# Run a quick test experiment
uv run python main.py run --config configs/experiments/quick-test.yaml

# Validate a config without running (dry-run)
uv run python main.py run --config configs/experiments/default.yaml --dry-run

# List available config types
uv run python main.py list-configs llm
```

## Configuration

Configuration files live under `configs/` and are modular (LLM, profiles, sandbox, schedules, experiments).
See [configs/README.md](configs/README.md) for details and examples on composing configs and overriding values.

## Project Structure (high level)

- **Source:** `src/` — core modules and implementations.
  - `src/coevolution/populations/` — Population types (code, unittest, differential, agent_coder) and their specialized LLM operators.
  - `src/coevolution/strategies/` — Shared infrastructure (breeding, selection, probability).
  - See [src/coevolution/README.md](src/coevolution/README.md) for deeper details.
- **Configs:** `configs/` — modular YAML files used by `main.py`.
- **Data:** `data/` — datasets, generations, and evaluation outputs.
- **Scripts & experiments:** `experiments/` and `scripts/` — helpers and notebooks for running experiments and analyses.
- **Logs & outputs:** `logs/` — run metadata, saved configs, and generated artifacts.

## Running and Debugging

- Use `--dry-run` to validate configuration and view the resolved config without executing experiments.
- The runner saves the resolved config to `logs/run_config.json` within each run directory for reproducibility.

## Resuming Experiments

If an experiment is interrupted (e.g., due to a crash or manually stopping it), you can resume it using the same `run_id`:

```bash
uv run python main.py run --config configs/experiments/default.yaml --run-id <existing_run_id>
```

By default, the `--resume` flag is set to `True`. The system will:

1. **Reuse the existing log directory** without renaming it.
2. **Scan existing problem logs** for a completion marker (`"event": "survived"`).
3. **Skip already completed problems** and jump directly to the next one in the sequence.
4. **Override partial logs** for problems that were interrupted mid-evolution to ensure a clean restart for that specific problem.

To disable this behavior and force a fresh run (renaming the directory if a collision occurs), use `--no-resume`:

```bash
uv run python main.py run --config configs/experiments/default.yaml --run-id <existing_run_id> --no-resume
```

## Where to look next

- Experiment runner: [main.py](main.py)
- Coevolution implementation and API: [src/coevolution/README.md](src/coevolution/README.md)
- Config documentation: [configs/README.md](configs/README.md)

### Developer Guides

- [How to Add a Language](src/infrastructure/languages/HOW_TO_ADD_A_LANGUAGE.md)
- [How to Add an Operator](src/coevolution/populations/ADDING_AN_OPERATOR.md)
- [How to Add a Population](src/coevolution/populations/ADDING_A_POPULATION.md)
