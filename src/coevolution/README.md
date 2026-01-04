# Coevolution Module

This module implements Bayesian coevolution algorithms with various selection strategies for evolutionary computation.

## Structure

```
coevolution/
├── __init__.py          # Main module interface
├── orchestrator.py      # Main algorithm coordinator
├── bayesian.py          # Bayesian belief updating logic
├── evaluation.py        # Observation matrix generation
├── operators.py         # LLM-based genetic operators
├── population.py        # Population management
├── selection.py         # Selection strategies for evolutionary algorithms
├── config.py            # Configuration parameters
├── README.md           # This file
```

## Quick Start

```python
from coevolution.config import CoevolutionConfig
from coevolution.orchestrator import CoevolutionOrchestrator
from infrastructure.llm_client import LLMClient
from infrastructure.sandbox import SafeCodeSandbox
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

# 1. Configure the algorithm
config = CoevolutionConfig(
    initial_code_population_size=10,
    initial_test_population_size=20,
    num_generations=50,
    code_crossover_rate=0.7,
    code_mutation_rate=0.2,
)

# 2. Initialize dependencies
llm_client = LLMClient(model="gpt-4")
sandbox = SafeCodeSandbox()
problem = CodeGenerationProblem(...)  # Your coding problem

# 3. Create and run orchestrator
orchestrator = CoevolutionOrchestrator(
    config=config,
    problem=problem,
    llm_client=llm_client,
    sandbox=sandbox
)

# 4. Run the algorithm
best_code, best_code_prob, best_test, best_test_prob = orchestrator.run()

print(f"Best code (probability={best_code_prob:.4f}):")
print(best_code)
```

## Modules

### `orchestrator.py` (NEW!)

Main orchestrator class that coordinates the entire coevolution algorithm.

**Key Class:**

- `CoevolutionOrchestrator`: Manages the complete coevolution workflow

**Main Method:**

- `run()`: Execute the full algorithm, returns `(best_code, code_prob, best_test, test_prob)`

**Algorithm Workflow:**

1. Create initial populations (code and test)
2. For each generation:
   - Generate observation matrix (execute code vs tests)
   - Update beliefs using Bayesian updates
   - Select elite individuals
   - Generate offspring using genetic operators
   - Create next generation from elites + offspring
3. Return best code and test

**Internal Methods:**

- `_create_initial_code_population()`: Initialize code population with priors
- `_create_initial_test_population()`: Initialize test population with priors
- `_generate_code_offspring()`: Apply genetic operators to create code offspring
- `_generate_test_offspring()`: Apply genetic operators to create test offspring
- `_create_next_code_generation()`: Combine elites and offspring for next generation
- `_create_next_test_generation()`: Combine elites and offspring for next generation

**Features:**

- Comprehensive logging at all stages
- Automatic population size handling
- Elitism support
- Configurable genetic operator rates
- Safe test method extraction and class rebuilding

### `evaluation.py`

Provides functions to generate observation matrices by executing code populations against test populations in a safe sandbox environment.

**Key Functions:**

- `generate_observation_matrix()`: Execute code vs tests, treating each test block as one unit

### `config.py`

Configuration classes for coevolutionary algorithms.

**Key Classes:**

- `CoevolutionConfig`: Configuration parameters for the coevolutionary algorithm with Bayesian updates

**Configuration Parameters:**

*Bayesian Parameters:*

- `initial_code_population_size`: Size of initial code population (default: 10)
- `initial_test_population_size`: Size of initial test population (default: 20)
- `initial_code_prior`: Prior probability that a code is correct (default: 0.5)
- `initial_test_prior`: Prior probability that a test is correct (default: 0.5)
- `alpha`: P(pass | code correct, test incorrect) - hyperparameter (default: 0.1)
- `beta`: P(pass | code incorrect, test correct) - hyperparameter (default: 0.2)
- `gamma`: P(pass | code incorrect, test incorrect) - hyperparameter (default: 0.5)
- `learning_rate`: Learning rate for belief updates (default: 1.0)
- `use_intermediate_updates`: Use updated code probs when calculating test updates (default: False)

*Evolution Strategy:*

- `num_generations`: Number of evolutionary cycles (default: 50)
- `selection_strategy`: Selection method to use (default: "binary_tournament")

*Code Population Genetic Operators:*

- `code_crossover_rate`: Probability of crossover operation (default: 0.7)
- `code_mutation_rate`: Probability of mutation operation (default: 0.2)
- `code_edit_rate`: Probability of edit operation based on feedback (default: 0.1)
- `code_elitism_count`: Number of elite individuals to preserve (default: 2)
- `code_offspring_size`: Offspring generated per generation (default: None = population_size)

*Test Population Genetic Operators:*

- `test_crossover_rate`: Probability of crossover operation (default: 0.6)
- `test_mutation_rate`: Probability of mutation operation (default: 0.3)
- `test_edit_rate`: Probability of edit operation based on feedback (default: 0.1)

Note: test populations are fixed-size and do not use separate elite/offspring counts
or a separate maximum population size. Control the size of the test population with
`initial_test_population_size` in `CoevolutionConfig`.

*LLM Configuration:*

- `llm_model`: LLM model name for genetic operators (default: "gpt-4")

### `bayesian.py`

Implements Bayesian belief updating for code-test coevolution where both populations evolve simultaneously with mutual evaluation and belief updates.

**Key Functions:**

- `initialize_prior_beliefs(config)`: Initialize prior belief probabilities for code and test populations
- `update_population_beliefs(prior_code_probs, prior_test_probs, observation_matrix, config)`: Perform Bayesian updates based on evaluation results

**Implementation Details:**

- Public-facing functions operate in probability space for convenience
- Internal calculations use log-odds space for numerical stability
- Uses `_NUMERICAL_STABILITY_EPSILON = 1e-9` to prevent log(0) errors

**Example:**

```python
from coevolution.config import CoevolutionConfig
from coevolution.bayesian import initialize_prior_beliefs, update_population_beliefs
from coevolution.evaluation import generate_observation_matrix

# Create configuration
config = CoevolutionConfig(
    # Bayesian parameters
    initial_code_population_size=10,
    initial_test_population_size=20,
    initial_code_prior=0.6,
    initial_test_prior=0.2,
    alpha=0.1,
    beta=0.2,
    gamma=0.5,
    learning_rate=1.0,
    use_intermediate_updates=False,
    # Evolution strategy
    num_generations=50,
    selection_strategy="binary_tournament",
    # Code genetic operators
    code_crossover_rate=0.7,
    code_mutation_rate=0.2,
    code_edit_rate=0.1,
    code_elitism_count=2,
    # Test genetic operators
    test_crossover_rate=0.6,
    test_mutation_rate=0.3,
    test_edit_rate=0.1,
    
    # LLM configuration
    llm_model="gpt-4"
)

# Initialize prior beliefs
code_probs, test_probs = initialize_prior_beliefs(config)

# Create populations (see population.py)
code_population = CodePopulation(code_solutions, code_probs)
test_population = TestPopulation(test_methods, test_probs, test_class_block=test_class)

# Generate observation matrix by executing tests
observation_matrix = generate_observation_matrix(code_population, test_population, sandbox)

# Update beliefs based on test results
updated_code_probs, updated_test_probs = update_population_beliefs(
    code_probs, test_probs, observation_matrix, config
)
```

### `population.py`

Population management classes for evolutionary algorithms.

**Key Classes:**

- `BasePopulation`: Abstract base class for all population types
- `CodePopulation`: Manages code solution populations
- `TestPopulation`: Manages test case populations (with unittest class tracking)

**Key Features:**

- Individuals with associated correctness probabilities
- Generation tracking
- Methods for adding, removing, and replacing individuals
- Top-k selection and iteration support

**Example:**

```python
from coevolution.population import CodePopulation, TestPopulation
import numpy as np

# Create code population
code_solutions = ["def add(a,b): return a+b", "def add(a,b): return a-b"]
code_probs = np.array([0.9, 0.3])
code_pop = CodePopulation(code_solutions, code_probs)

# Create test population (requires full unittest class)
test_methods = ["def test_add_positive(self): ...", "def test_add_negative(self): ..."]
test_probs = np.array([0.8, 0.7])
test_class = "class TestMath(unittest.TestCase):\n    def test_add_positive..."
test_pop = TestPopulation(test_methods, test_probs, test_class_block=test_class)

# Get best individual
best_code, best_prob = code_pop.get_best_individual()

# Iterate over population
for individual, prob in code_pop:
    print(f"Probability: {prob}")
```

### `operators.py`

LLM-based genetic operators for evolutionary algorithms.

**Key Classes:**

- `BaseLLMOperator`: Abstract base class for LLM-based genetic operators
- `CodeOperator`: Operator for evolving code solutions
- `TestOperator`: Operator for evolving test cases

**Genetic Operations:**

- `create_initial_population(size)`: Generate initial population
- `mutate(individual)`: Create variations of an individual
- `crossover(parent1, parent2)`: Combine two parents
- `edit(individual, feedback)`: Fix individual based on feedback

**Design Note:**

- `CodeOperator.create_initial_population()` returns `List[str]` (multiple separate solutions)
- `TestOperator.create_initial_population()` returns `str` (single unittest class with multiple methods)

**Example:**

```python
from coevolution.operators import CodeOperator, TestOperator
from infrastructure.llm_client import LLMClient
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

llm = LLMClient(model="gpt-4")
problem = CodeGenerationProblem(...)

# Code operator
code_op = CodeOperator(llm)
code_op.set_problem(problem)
solutions = code_op.create_initial_population(5)
mutated = code_op.mutate(solutions[0])
child = code_op.crossover(solutions[0], solutions[1])

# Test operator
test_op = TestOperator(llm)
test_op.set_problem(problem)
test_class = test_op.create_initial_population(5)  # Returns single unittest class
```

### `selection.py`

Provides various selection strategies for evolutionary algorithms.

**Key Classes:**

- `SelectionStrategy`: Container class for all selection methods

**Available Methods:**

- `binary_tournament(population)`: Selects winner from two random individuals
- `roulette_wheel(population)`: Probability-based selection proportional to fitness
- `rank_selection(population)`: Selection based on fitness rank rather than raw values
- `random_selection(population)`: Uniform random selection
- `elitism(population, num_elites)`: Selects the top N individuals
- `select_parents(population, method)`: Selects two different parents using any method

**Example:**

```python
from coevolution.selection import SelectionStrategy
from coevolution.population import CodePopulation
import numpy as np

# Create population
code_solutions = ["solution1", "solution2", "solution3", "solution4"]
probabilities = np.array([0.2, 0.5, 0.8, 0.3])
population = CodePopulation(code_solutions, probabilities)

# Select two parents using binary tournament
(parent1, prob1), (parent2, prob2) = SelectionStrategy.select_parents(
    population, method="binary_tournament"
)

# Get top 2 individuals (elitism)
elites = SelectionStrategy.elitism(population, num_elites=2)

# Check available methods
methods = SelectionStrategy.get_available_methods()
print(methods)  # ['binary_tournament', 'roulette_wheel', 'rank_selection', 'random_selection']
```

## Adding New Selection Methods

To add a new selection method:

1. Add a static method to `SelectionStrategy` class in `selection.py`:

```python
@staticmethod
def new_method(population: BasePopulation) -> tuple[str, float]:
    """Your new selection method.
    
    Args:
        population: BasePopulation object containing individuals and probabilities
        
    Returns:
        Tuple of (selected_individual, selected_probability)
    """
    # Implementation here
    pass
```

2. Update the `_get_selection_function()` method to include your new method:

```python
methods: dict[str, Callable[[BasePopulation], tuple[str, float]]] = {
    "binary_tournament": cls.binary_tournament,
    "roulette_wheel": cls.roulette_wheel,
    "rank_selection": cls.rank_selection,
    "random_selection": cls.random_selection,
    "new_method": cls.new_method,  # Add this line
}
```

3. Add the method name to `get_available_methods()`:

```python
return [
    "binary_tournament",
    "roulette_wheel",
    "rank_selection",
    "random_selection",
    "new_method",  # Add this line
]
```

4. The method will now be available through `select_parents(population, method="new_method")`.

## Usage

### Hierarchical Imports (Recommended)

```python
# Import from specific submodules
from coevolution.config import CoevolutionConfig
from coevolution.bayesian import initialize_prior_beliefs, update_population_beliefs
from coevolution.operators import CodeOperator, TestOperator
from coevolution.selection import SelectionStrategy
from coevolution.population import CodePopulation, TestPopulation
from coevolution.evaluation import generate_observation_matrix
```

### Module-Level Import

```python
# Import the entire module
import common.coevolution.bayesian as bayesian
import common.coevolution.selection as selection
```

**Note:** The `__init__.py` uses hierarchical imports only (no re-exports). Always import from the specific submodules as shown above.

## Design Principles

1. **Separation of Concerns**: Bayesian logic and selection strategies are in separate modules
2. **Extensibility**: Easy to add new selection methods via dictionary mapping
3. **Type Safety**: Uses NumPy arrays with type hints
4. **Numerical Stability**: Bayesian updates use log-odds space internally
5. **Backward Compatibility**: Exports through main `common` module

## Complete Workflow Example

```python
from coevolution.config import CoevolutionConfig
from coevolution.bayesian import initialize_prior_beliefs, update_population_beliefs
from coevolution.operators import CodeOperator, TestOperator
from coevolution.population import CodePopulation, TestPopulation
from coevolution.selection import SelectionStrategy
from coevolution.evaluation import generate_observation_matrix
from infrastructure.llm_client import LLMClient
from infrastructure.sandbox import SafeCodeSandbox

# 1. Setup
config = CoevolutionConfig(
    initial_code_population_size=10,
    initial_test_population_size=20,
    initial_code_prior=0.5,
    initial_test_prior=0.5,
    alpha=0.1, 
    beta=0.2, 
    gamma=0.5,
    learning_rate=1.0,
    num_generations=50,
    selection_strategy="binary_tournament",
    code_crossover_rate=0.7,
    code_mutation_rate=0.2,
    code_elitism_count=2,
    test_crossover_rate=0.6,
    test_mutation_rate=0.3,
    llm_model="gpt-4"
)

llm = LLMClient(model="gpt-4")
sandbox = SafeCodeSandbox()

# 2. Initialize operators
code_op = CodeOperator(llm)
test_op = TestOperator(llm)
code_op.set_problem(problem)
test_op.set_problem(problem)

# 3. Create initial populations
code_solutions = code_op.create_initial_population(config.initial_code_population_size)
test_class = test_op.create_initial_population(config.initial_test_population_size)

# 4. Initialize beliefs
code_probs, test_probs = initialize_prior_beliefs(config)

# 5. Create population objects
code_pop = CodePopulation(code_solutions, code_probs)
test_pop = TestPopulation(test_methods, test_probs, test_class_block=test_class)

# 6. Evaluate and update beliefs
observation_matrix = generate_observation_matrix(code_pop, test_pop, sandbox)
updated_code_probs, updated_test_probs = update_population_beliefs(
    code_pop.probabilities, test_pop.probabilities, observation_matrix, config
)

# 7. Apply selection and genetic operations
(parent1, _), (parent2, _) = SelectionStrategy.select_parents(code_pop, method="binary_tournament")
offspring = code_op.crossover(parent1, parent2)
mutated = code_op.mutate(offspring)
```

## Future Extensions

Potential areas for expansion:

- Advanced crossover strategies in operators
- Multi-objective optimization support
- Adaptive hyperparameter tuning (alpha, beta, gamma)
- Parallel evaluation support
- Population diversity metrics
