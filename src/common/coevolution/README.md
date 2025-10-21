# Coevolution Module

This module implements Bayesian coevolution algorithms with various selection strategies for evolutionary computation.

## Structure

```
coevolution/
├── __init__.py          # Main module interface
├── bayesian.py          # Bayesian belief updating logic
├── selection.py         # Selection strategies for evolutionary algorithms
└── README.md           # This file
```

## Modules

### `bayesian.py`

Implements Bayesian belief updating for code-test coevolution where both populations evolve simultaneously with mutual evaluation and belief updates.

**Key Classes:**

- `CoevolutionConfig`: Configuration parameters for the coevolutionary algorithm

**Key Functions:**

- `initialize_populations()`: Initialize code and test populations with prior beliefs
- `update_population_beliefs()`: Perform Bayesian updates based on evaluation results
- `run_evaluation()`: Simulate evaluation matrix (for testing)

**Example:**

```python
from common.coevolution import CoevolutionConfig, initialize_populations, update_population_beliefs

# Create configuration
config = CoevolutionConfig(
    initial_code_population_size=10,
    initial_test_population_size=20,
    c0_prior=0.6,
    t0_prior=0.2,
    alpha=0.01,
    beta=0.2,
    gamma=0.2
)

# Initialize populations
code_probs, test_probs = initialize_populations(config)

# After evaluation, update beliefs
updated_code_probs, updated_test_probs = update_population_beliefs(
    code_probs, test_probs, evaluation_matrix, config
)
```

### `selection.py`

Provides various selection strategies for evolutionary algorithms.

**Key Classes:**

- `SelectionStrategy`: Container class for all selection methods

**Available Methods:**

- `binary_tournament`: Selects winner from two random individuals
- `roulette_wheel`: Probability-based selection proportional to fitness
- `rank_selection`: Selection based on fitness rank rather than raw values
- `elitism`: Selects the top N individuals
- `select_parents`: Selects two different parents using any method

**Example:**

```python
from common.coevolution import SelectionStrategy
import numpy as np

population = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
fitness = np.array([0.2, 0.5, 0.8, 0.3])

# Select two parents using binary tournament
parent1, parent2 = SelectionStrategy.select_parents(
    population, fitness, method="binary_tournament"
)

# Get top 2 individuals
elites = SelectionStrategy.elitism(population, fitness, num_elites=2)

# Check available methods
methods = SelectionStrategy.get_available_methods()
print(methods)  # ['binary_tournament', 'roulette_wheel', 'rank_selection']
```

## Adding New Selection Methods

To add a new selection method:

1. Add a static method to `SelectionStrategy` class in `selection.py`:

```python
@staticmethod
def new_method(population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
    """Your new selection method."""
    # Implementation here
    pass
```

2. Register it in the `_SELECTION_METHODS` dictionary:

```python
_SELECTION_METHODS = {
    "binary_tournament": binary_tournament.__func__,
    "roulette_wheel": roulette_wheel.__func__,
    "rank_selection": rank_selection.__func__,
    "new_method": new_method.__func__,  # Add this line
}
```

3. The method will automatically be available through `select_parents()` and `get_available_methods()`.

## Usage in Notebooks

```python
# Import the entire module
import common.coevolution as coevo

# Or import specific components
from common.coevolution import SelectionStrategy, CoevolutionConfig

# Backward compatible import (also works)
from common import SelectionStrategy, CoevolutionConfig
```

## Design Principles

1. **Separation of Concerns**: Bayesian logic and selection strategies are in separate modules
2. **Extensibility**: Easy to add new selection methods via dictionary mapping
3. **Type Safety**: Uses NumPy arrays with type hints
4. **Numerical Stability**: Bayesian updates use log-odds space internally
5. **Backward Compatibility**: Exports through main `common` module

## Future Extensions

Potential areas for expansion:

- `crossover.py`: Crossover operators for genetic algorithms
- `mutation.py`: Mutation operators
- `fitness.py`: Fitness evaluation functions
- `population.py`: Population management utilities
