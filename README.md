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
- The runner saves a resolved config and run metadata to `logs/configs/` and `logs/metadata/` respectively for reproducibility.

## Where to look next

- Experiment runner: [main.py](main.py)
- Coevolution implementation and API: [src/coevolution/README.md](src/coevolution/README.md)
- Config documentation: [configs/README.md](configs/README.md)

If you'd like, I can also add a short example workflow or a minimal example config.
