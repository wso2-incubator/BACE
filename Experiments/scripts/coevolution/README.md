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

**What it does:**

1. **Loads a problem**: Gets a HARD difficulty problem from LiveCodeBench (release_v5)
2. **Creates initial populations**:
   - 5 code solutions (using LLM)
   - 10 test cases (using LLM)
3. **Runs 3 generations** of coevolution:
   - Executes all code against all tests
   - Updates Bayesian beliefs based on pass/fail results
   - Selects parents and applies genetic operators
   - Preserves elite individuals
4. **Returns results**: Best code solution and test case with their probabilities

**Configuration:**

The script uses conservative parameters for quick testing:

- Small populations (5 code, 10 tests)
- Few generations (3)
- Balanced genetic operator rates

You can modify these in the script to run longer experiments.

**Expected Runtime:**

- ~2-5 minutes for 3 generations (depending on LLM latency)

**Output:**

The script logs detailed information about each step:

- Population initialization
- Test execution results
- Bayesian belief updates
- Genetic operations
- Best individuals from each generation

Final output shows:

- Best code solution with its probability
- Best test case with its probability
