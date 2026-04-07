# Differential Test Profile Configuration

Configuration settings for differential test discovery in coevolutionary testing.

## Overview

The differential profile uses **Differential Evolution Testing (DET)** to discover divergent behavior between functionally equivalent code solutions. It identifies code pairs that produce identical results on existing tests but can be distinguished with carefully crafted inputs.

## Configuration Parameters

### Population Settings

- **`initial_prior`** (default: `0.2`)  
  Initial probability assigned to newly discovered differential tests. Lower than unittest tests because differential tests are generally less reliable indicators of correctness.

- **`initial_population_size`** (default: `0`)  
  Starting population size. Set to `0` for bootstrap mode where the population grows purely from discoveries.

- **`max_population_size`** (default: `100`)  
  Maximum number of differential tests to maintain in the population.

- **`offspring_rate`** (default: `0.5`)  
  Fraction of available capacity to fill with new offspring each generation.

- **`elitism_rate`** (default: `0.3`)  
  Fraction of top-performing tests to preserve across generations.

### Operation Rates

- **`discovery_rate`** (default: `1.0`)  
  Probability of using the discovery operation. Should sum to 1.0 with other operations.

### Bayesian Parameters

- **`alpha`** (default: `0.2`)  
  P(test passes | code correct, test incorrect). Higher than unittest because differential tests have higher false positive rates.

- **`beta`** (default: `0.2`)  
  P(test passes | code incorrect, test correct). Represents test sensitivity.

- **`gamma`** (default: `0.3`)  
  P(test passes | both code and test incorrect). Baseline noise probability.

- **`learning_rate`** (default: `0.025`)  
  Rate at which beliefs are updated based on new observations. Lower than unittest for more cautious updates.

### Performance Settings

- **`llm_workers`** (default: `10`)  
  Number of parallel threads for LLM-based script generation (I/O bound operations).

- **`cpu_workers`** (managed globally)  
  Number of parallel processes for sandbox execution (CPU bound operations). Handled at the infrastructure level via the `COEVOLUTION_WORKERS` environment variable or `sandbox.workers` config in `main.py`.

### Breeding Strategy Parameters

- **`max_pairs_per_group`** (default: `5`)  
  Maximum number of code pairs to attempt from each functionally equivalent group during candidate selection.
  
  **Rationale**: When a functional group contains many equivalent code solutions, attempting all possible pairs (nCr combinations) can be wasteful if the codes are truly equivalent. This parameter limits exploration to the top N pairs (sorted by probability) to prevent excessive computation on groups unlikely to yield divergences.
  
  **Effect**:
  - Lower values (1-3): Faster but may miss divergences in large groups
  - Higher values (7-10): More thorough but potentially slower
  - Recommended: 5 provides good balance between exploration and efficiency

## Example Configuration

```yaml
# Standard Differential Test Profile
initial_prior: 0.2
initial_population_size: 0
max_population_size: 100
offspring_rate: 0.5
elitism_rate: 0.3

discovery_rate: 1.0

alpha: 0.2
beta: 0.2
gamma: 0.3
learning_rate: 0.025

llm_workers: 10

max_pairs_per_group: 5
```

## Usage

Reference this profile in your experiment configuration:

```yaml
profiles:
  differential: configs/profiles/differential/standard.yaml
```

## Notes

- Differential tests start from an empty population and grow through discovery
- Tests are less reliable than unit tests but can find subtle behavioral differences
- The `max_pairs_per_group` parameter is key for controlling computational cost in large functional equivalence groups
