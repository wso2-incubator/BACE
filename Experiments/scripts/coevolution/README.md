# Coevolution Scripts

This directory contains scripts for running coevolution experiments.

## Scripts

### `run_simple_coevolution.py`

A simple demonstration of the CoevolutionOrchestrator that:

- Loads a problem from LiveCodeBench dataset
- Evolves code solutions and test cases together
- Uses Bayesian belief updating to track correctness probabilities
- Applies genetic operators (crossover, mutation, edit) to both populations
- Returns the best code solution and test case after evolution

**Usage:**

```bash
# From the project root
uv run python experiments/scripts/coevolution/run_simple_coevolution.py
```
