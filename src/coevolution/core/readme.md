# Core Modules Documentation

This document provides a comprehensive overview of the core modules in the coevolution system, explaining their purpose, functionality, and relationships.

## Overview

The coevolution framework implements a co-evolutionary algorithm where code and test populations evolve together. The core modules define the fundamental abstractions, data structures, and orchestration logic that drive this evolution.

---

## Module Structure

```
src/coevolution/core/
├── individual.py          # Concrete individual implementations
├── population.py          # Concrete population implementations
├── orchestrator.py        # Main orchestration logic
├── mock.py               # Mock implementations for testing
└── interfaces/           # Protocol definitions and abstractions
    ├── base.py           # Base classes for individuals and populations
    ├── types.py          # Type aliases and enums
    ├── data.py           # Data structures and DTOs
    ├── config.py         # Configuration dataclasses
    ├── operators.py      # Operator protocols
    ├── breeding.py       # Breeding strategy protocols
    ├── selection.py      # Selection strategy protocols
    ├── systems.py        # System-level protocols
    ├── context.py        # Context objects
    └── profiles.py       # Profile bundling classes
```

---

## Core Modules

### 1. `individual.py` - Individual Implementations

**Purpose**: Provides concrete implementations of code and test individuals.

**Key Classes**:

#### `CodeIndividual`

- Represents a code solution candidate
- Inherits from `BaseIndividual`
- Maintains unique ID with prefix `C` (e.g., `C0`, `C1`, `C2`)
- Tracks probability (belief in correctness), generation born, creation operation, and parent lineage

#### `TestIndividual`

- Represents a test case candidate
- Inherits from `BaseIndividual`
- Maintains unique ID with prefix `T` (e.g., `T0`, `T1`, `T2`)
- Similar tracking as CodeIndividual but for tests

**Key Features**:

- Auto-incrementing counters for unique IDs
- Immutable core attributes (snippet, creation_op, generation_born)
- Lifecycle logging (creation, parenthood, selection, probability updates, death/survival)

---

### 2. `population.py` - Population Implementations

**Purpose**: Provides concrete implementations for managing collections of individuals.

**Key Classes**:

#### `CodePopulation`

- Manages a collection of `CodeIndividual` objects
- Tracks generation number
- Provides utilities for:
  - Computing average probability
  - Getting best individuals
  - Updating probabilities via Bayesian updates
  - Transitioning to next generation

#### `TestPopulation`

- Manages a collection of `TestIndividual` objects
- Additional feature: Maintains and rebuilds the **test class block**
  - The test class block is the full test file with imports, setup, and helper methods
  - Individual test methods are extracted from this block
  - When population advances, the block is rebuilt with new test methods
- Uses injected `ITestBlockRebuilder` for rebuilding logic

**Key Features**:

- Support for empty populations (enables bootstrapping scenarios like differential testing)
- Generation advancement with logging of kept/added/removed individuals
- Automatic probability updates with delta tracking
- Index-based access to individuals

---

### 3. `orchestrator.py` - Main Orchestration Logic

**Purpose**: Coordinates the entire co-evolutionary algorithm by "wiring up" all components.

**Key Responsibilities**:

#### Initialization

- Creates initial code population from problem starter code
- Creates evolved test populations (e.g., unittest, differential) from configurations
- Creates fixed test populations (public, private) from dataset test cases
- Logs initial state and baseline execution results

#### Main Evolution Loop

The orchestrator runs a phased evolution schedule with configurable rules:

**For Each Phase**:

- **Execute**: Run code against all test populations
  - Evolved tests (unittest, differential, etc.)
  - Public tests (ground truth/anchoring)
  - Returns interaction data (execution results + observation matrices)

- **Update Beliefs**: Bayesian belief updates with specific ordering to prevent bias
  1. **Anchor**: Update code beliefs using public (ground truth) tests
  2. **Calculate**: For each evolved test population:
     - Update test beliefs based on anchored code
     - Update code beliefs based on test priors
  3. **Apply**: Apply all updates simultaneously

- **Evolve**: Generate next generation (if phase allows)
  - **Code Evolution** (if phase.evolve_code):
    - Select code elites using elite selection strategy
    - Breed code offspring using breeding strategy
    - Transition to next generation
  - **Test Evolution** (if phase.evolve_tests):
    - For each evolved test type:
      - Select test elites (typically Pareto-based)
      - Breed test offspring
      - Transition to next generation

- **Log**: Generation summary with population statistics

#### Finalization

- Final execution on all test sets
- Final private test evaluation
- Log surviving individuals and their complete lifecycle

**Design Principles**:

- **Dependency Injection**: All strategies, operators, and systems injected
- **Separation of Concerns**: Orchestrator coordinates but doesn't implement evolution logic
- **Stateless**: Can work on different problems by calling `run()` with different problem instances
- **Multiple Test Populations**: Supports arbitrary number of evolved and fixed test types

**Key Methods**:

- `run(problem)`: Main entry point, returns final populations
- `_initialize_evolution()`: Creates initial populations
- `_execute_all_interactions()`: Runs all test executions
- `_perform_cooperative_updates()`: Bayesian belief updates
- `_produce_next_generation()`: Selection and breeding
- `_finalize_evolution()`: Final evaluation and logging

---

### 4. `interfaces/base.py` - Base Abstractions

**Purpose**: Defines abstract base classes for individuals and populations.

**Key Classes**:

#### `BaseIndividual`

Abstract base class implementing shared logic for all individuals.

**State Management**:

- `snippet`: The code/test content (immutable)
- `probability`: Belief in correctness (mutable, updated via Bayesian updates)
- `creation_op`: Operation that created this individual (immutable)
- `generation_born`: Generation when created (immutable)
- `parents`: Parent lineage grouped by type `{"code": [ids], "test": [ids]}`
- `metadata`: Operation-specific metadata (prompts, reasoning, traces)
- `lifecycle_log`: Structured log of all lifecycle events

**Lifecycle Events**:

- `CREATED`: Individual created
- `BECAME_PARENT`: Used in breeding
- `SELECTED_AS_ELITE`: Survived selection
- `PROBABILITY_UPDATED`: Belief updated
- `DIED`: Removed from population
- `SURVIVED`: Made it to final generation

**Key Methods**:

- `notify_*()`: Record lifecycle events
- `get_complete_record()`: Export full history for logging

#### `BasePopulation[T_Individual]`

Abstract base class implementing shared logic for populations.

**Key Features**:

- Generic over individual type `T`
- Supports iteration, indexing, slicing
- Empty population support for bootstrapping
- Generation tracking and advancement
- Probability management and statistics

**Key Methods**:

- `set_next_generation()`: Replace population and advance generation
- `update_probabilities()`: Bayesian update for all individuals
- `get_best_individual()`: Highest probability individual
- `get_top_k_individuals()`: Top k by probability
- `compute_average_probability()`: Population average

---

### 5. `interfaces/types.py` - Type Definitions

**Purpose**: Central location for type aliases, enums, and constants.

**Key Definitions**:

```python
# Type alias for genetic operations (flexible string-based)
type Operation = str

# Parent lineage tracking
type ParentDict = dict[Literal["code", "test"], list[str]]

# Standard operations (not exhaustive)
OPERATION_INITIAL = "initial"      # Generation 0
OPERATION_CROSSOVER = "crossover"  # Two-parent combination
OPERATION_EDIT = "edit"            # Feedback-driven improvement
OPERATION_REPRODUCTION = "reproduction"  # Elite preservation
OPERATION_MUTATION = "mutation"    # Single-parent variation

# Lifecycle event types
class LifecycleEvent(Enum):
    CREATED = "created"
    BECAME_PARENT = "became_parent"
    SELECTED_AS_ELITE = "selected_as_elite"
    PROBABILITY_UPDATED = "probability_updated"
    DIED = "died"
    SURVIVED = "survived"

# Execution result mappings
type ExecutionResults = dict[str, ExecutionResult]
type InteractionKey = tuple[str, str, str, str]
```

**Design Philosophy**:

- Operations are strings for flexibility (support custom operations)
- Standard operations provided as constants for convenience
- Type safety via type aliases and enums

---

### 6. `interfaces/data.py` - Data Structures

**Purpose**: Domain data structures and Data Transfer Objects (DTOs).

**Key Classes**:

#### `EvaluationResult`

Represents result of a single unit test execution.

- `status`: "passed" | "failed" | "error"
- `error_log`: Error message or stack trace
- `execution_time`: Time taken in seconds

#### `ExecutionResults`

Represents the collected results of executing a code population against tests.

- `results`: Dict mapping `code_id -> {test_id -> EvaluationResult}`

#### `InteractionData`

Captures interaction between code and test populations.

- `execution_results`: `ExecutionResults` instance containing ID-keyed outcomes
- `observation_matrix`: NumPy array with shape `(n_code, n_test)`
  - `observation_matrix[i, j] = 1` if code[i] passed test[j], else 0
  - Ensures strict index alignment

**Why InteractionData?**

- **Atomic Construction**: Executor creates both results and matrix together
- **Guaranteed Alignment**: No split-brain problem between IDs and indices
- **Type Safety**: Encapsulates the relationship between code and test populations

#### `LogEntry`

Structured log entry for individual lifecycle events.

- `generation`: When event occurred
- `event`: Type of lifecycle event
- `details`: Event-specific information

#### `Test`

Dataset test case.

- `input`: Test input
- `output`: Expected output

#### `Problem`

Complete problem specification.

- `question_title`, `question_content`, `question_id`
- `starter_code`: Code scaffold
- `public_test_cases`: Visible test cases
- `private_test_cases`: Hidden evaluation tests

---

### 7. `interfaces/config.py` - Configuration Classes

**Purpose**: Immutable configuration dataclasses for the framework.

**Key Classes**:

#### `BayesianConfig`

Hyperparameters for Bayesian belief updating.

- `alpha`: P(pass | code correct, test incorrect) - false negative rate
- `beta`: P(pass | code incorrect, test correct) - false positive rate
- `gamma`: P(pass | both incorrect) - noise floor
- `learning_rate`: Step size for belief updates (0.0, 1.0]

**Validation**: Ensures all probabilities in valid ranges

#### `OperatorRatesConfig`

Probability distribution over genetic operations.

- `operation_rates`: Dict mapping operation name → selection probability
- Must sum to exactly 1.0 (all breeding is productive)
- No reproduction operation (elite selection handles preservation)

**Example**:

```python
OperatorRatesConfig(
    operation_rates={
        "mutation": 0.4,   # 40% single-parent variation
        "crossover": 0.3,  # 30% two-parent combination
        "edit": 0.3,       # 30% feedback-driven improvement
    }
)
```

#### `PopulationConfig`

Population-specific parameters.

- `initial_prior`: Starting probability for Gen 0
- `initial_population_size`: Size at Gen 0
- `max_population_size`: Maximum size
- `offspring_rate`: Fraction of max_size to breed as offspring

---

### 8. `interfaces/operators.py` - Operator Protocols

**Purpose**: Defines protocols for genetic operators (stateless workers).

**Design Philosophy**: "Smart Strategy, Dumb Operator"

- Operators handle pure string transformation
- No knowledge of domain objects (Individuals, Populations, Matrices)
- Breeding strategies handle all domain complexity

**Key Protocols**:

#### `IOperator`

Protocol for genetic operators.

**Methods**:

- `generate_initial_snippets(InitialInput)`: Create Gen 0 population
  - Returns `(OperatorOutput, context_code)`
  - `context_code` is test class scaffold for test operators
  - Supports empty populations (size=0 returns empty results)

- `apply(BaseOperatorInput)`: Apply genetic operation
  - Takes operation-specific input DTO
  - Returns `OperatorOutput` with results + metadata

- `supported_operations()`: Returns set of operation names this operator can handle

**Input DTOs**:

- `InitialInput`: For Gen 0 creation
- `BaseOperatorInput`: Base class for operation inputs (extensible)

**Output DTOs**:

- `OperatorResult`: Single result (snippet + metadata)
- `OperatorOutput`: List of results

---

### 9. `interfaces/breeding.py` - Breeding Strategies

**Purpose**: Defines protocols for breeding strategies (smart managers).

**Key Protocol**:

#### `IBreedingStrategy[T_self: BaseIndividual]`

Protocol for breeding strategies that bridge domain model and operators.

**Responsibilities**:

1. **Orchestration**: Decide which operations to perform based on rates
2. **Selection**: Select parents and operation-specific context
   - Generic selection via `IParentSelectionStrategy`
   - Specialized selection (e.g., find failing tests for edit operation)
3. **Data Preparation (Unwrapping)**: Extract strings from Individual objects into DTOs
4. **Dispatch**: Call operator with DTOs
5. **Construction (Wrapping)**: Wrap operator results into new Individuals with IDs, probabilities, lineage

**Methods**:

- `initialize_individuals(problem)`: Create Gen 0 population
  - Delegates to operator's `generate_initial_snippets()`
  - Wraps results into Individual objects
  - Returns `(individuals, context_code)`
  - Supports empty populations (size=0)

- `breed(coevolution_context, num_offsprings)`: Generate offspring
  - For each offspring:
    1. Sample operation from rates config
    2. Select context (parents, failing tests, divergent pairs, etc.)
    3. Prepare input DTO with raw strings
    4. Call `operator.apply(dto)`
    5. Calculate probabilities and create new Individuals
  - Returns exactly `num_offsprings` new individuals
  - Supports empty parent populations (bootstrapping)

**Design Pattern**: Strategy pattern with dependency injection

- Strategies are injected into populations
- Operators are injected into strategies
- Complete flexibility without core changes

---

### 10. `interfaces/selection.py` - Selection Strategies

**Purpose**: Defines protocols for elite and parent selection.

**Key Protocols**:

#### `IEliteSelectionStrategy[T: BaseIndividual]`

Select elite individuals to preserve unchanged to next generation.

**Method**:

```python
select_elites(
    population: BasePopulation[T],
    population_config: PopulationConfig,
    coevolution_context: CoevolutionContext,
) -> list[T]
```

**Strategy Examples**:

- **Code**: Multi-objective (probability + test performance)
- **Test**: Pareto front (probability + discrimination ability)
- **Simple**: Top-k by probability only
- **Diversity**: Select diverse behaviors

**Empty State**: Returns `[]` for empty populations

#### `IParentSelectionStrategy[T: BaseIndividual]`

Select parent individuals for breeding.

**Method**:

```python
select_parents(
    population: BasePopulation[T],
    count: int,  # 1 for mutation, 2 for crossover
    coevolution_context: CoevolutionContext,
) -> list[T]
```

**Strategy Examples**:

- **Roulette wheel**: Probability-proportional
- **Tournament**: Best from random subset
- **Rank-based**: Select by rank not raw probability
- **Behavior-based**: Complementary behaviors

---

### 11. `interfaces/systems.py` - System Protocols

**Purpose**: System-level protocols for execution and belief updates.

**Key Protocols**:

#### `IExecutionSystem`

Execute code against tests and return interaction data.

**Method**:

```python
execute_tests(
    code_population: CodePopulation,
    test_population: TestPopulation,
) -> InteractionData
```

**Guarantees**:

- Atomic construction of results and matrix
- Strict index alignment
- Consistency between ID-keyed results and index-based matrix

**Empty State Behavior**:

- Empty tests: Returns shape `(n_code, 0)`
- Empty code: Returns shape `(0, n_test)`

#### `IBeliefUpdater`

Bayesian belief update strategies.

**Methods**:

- `update_code_beliefs(...)`: Update code probabilities based on test results
- `update_test_beliefs(...)`: Update test probabilities based on code results

**Parameters**:

- Prior probabilities for both populations
- Observation matrix
- Update mask matrix (which observations to consider)
- Bayesian config (alpha, beta, gamma, learning_rate)

**Returns**: Updated posterior probabilities

**Empty State**: Identity operation when no evidence (empty matrices)

#### `IInteractionLedger`

Track which code-test pairs have been evaluated.

**Purpose**: Prevent double-counting evidence

- Get masks for new interactions only
- Commit interactions after belief updates
- Support multiple test populations

#### `ITestBlockRebuilder`

Rebuild test class blocks from test methods.

**Method**:

```python
rebuild_test_block(
    old_test_class_block: str,
    new_test_methods: list[str],
) -> str
```

**Purpose**: Maintain test suite structure across generations

---

### 12. `interfaces/context.py` - Context Objects

**Purpose**: Context objects for passing state through the system.

**Key Class**:

#### `CoevolutionContext`

Mutable container for system state at a specific generation.

**Structure**:

```python
@dataclass
class CoevolutionContext:
    problem: Problem
    code_population: CodePopulation  # Mutated during evolution
    test_populations: dict[str, TestPopulation]  # e.g., {"public", "unittest", "differential"}
    interactions: dict[str, InteractionData]  # Same keys as test_populations
```

**Lifecycle**:

1. Context created with populations + new interaction data
2. Populations mutated during belief updates
3. Populations mutated during evolution
4. Context discarded, new one created for next generation

**Design**: Explicit state container enabling functional programming style

- Clear data flow
- Mutations are intentional and tracked
- Easy to test and reason about

---

### 13. `interfaces/profiles.py` - Profile Classes

**Purpose**: Bundle configurations and strategies for populations.

**Key Classes**:

#### `CodeProfile`

Complete configuration for code population.

**Fields**:

- `population_config`: PopulationConfig (sizes, rates)
- `breeding_strategy`: IBreedingStrategy[CodeIndividual]
- `elite_selector`: IEliteSelectionStrategy[CodeIndividual]

**Usage**: One-stop shop for all code evolution components

#### `TestProfile`

Complete configuration for an evolved test population.

**Fields**:

- `population_config`: PopulationConfig
- `breeding_strategy`: IBreedingStrategy[TestIndividual]
- `elite_selector`: IEliteSelectionStrategy[TestIndividual]
- `bayesian_config`: BayesianConfig (reliability parameters)

**Usage**: Different test types can have different configurations

- unittest: High reliability (low alpha)
- differential: Lower reliability (higher alpha)

#### `PublicTestProfile`

Configuration for fixed/ground-truth test populations.

**Fields**:

- `bayesian_config`: BayesianConfig for anchoring updates

**Usage**: Public tests for anchoring, private tests for evaluation

---

### 14. `mock.py` - Mock Implementations

**Purpose**: Provides mock implementations for testing and development.

**Key Components**:

- `MockCodeOperator`: Simple operator that generates/mutates code strings
- `MockTestOperator`: Simple operator for tests
- `MockExecutionSystem`: Simulates test execution
- `MockBeliefUpdater`: Simple Bayesian update implementation
- `MockSelectionStrategy`: Basic selection logic
- `get_mock_problem()`: Factory for test problems

**Usage**:

- Unit testing of orchestrator and strategies
- Integration testing of the full system
- Prototyping new features
- Documentation examples

**Design**: Follows same protocols as real implementations

- Drop-in replacements
- Predictable behavior for testing
- Minimal external dependencies

---

## Architectural Principles

### 1. Separation of Concerns

- **Core**: Domain model and orchestration
- **Strategies**: Algorithm implementations (injected)
- **Operators**: String transformation workers (injected)
- **Systems**: Infrastructure (execution, belief updates)

### 2. Dependency Injection

- All strategies, operators, and systems injected into orchestrator
- No hard-coded implementations
- Complete flexibility and testability

### 3. Protocol-Based Design

- Interfaces defined via Python Protocols
- Duck typing with type safety
- Easy to add new implementations

### 4. Immutability Where Possible

- Core attributes of individuals are immutable
- Configurations are frozen dataclasses
- Only probabilities and populations mutate

### 5. Smart Strategy, Dumb Operator

- Strategies handle domain complexity
- Operators handle pure transformation
- Clear separation of concerns

### 6. Empty State Support

- All components handle empty populations gracefully
- Enables bootstrapping scenarios (e.g., differential testing)
- Identity operations when no evidence

### 7. Explicit Context Passing

- CoevolutionContext makes state flow explicit
- Reduces hidden dependencies
- Easier to test and reason about

---

## High-Level Co-Evolutionary Algorithm

Based on the orchestrator implementation, here is the complete algorithm:

```
ALGORITHM: Co-Evolutionary Code and Test Generation

INPUT:
  - problem: Problem specification with starter code and test cases
  - evo_config: Evolution configuration (num_epochs, schedule)
  - code_profile: Code population configuration and strategies
  - evolved_test_profiles: Dict of test type → test population configuration
  - public_test_profile: Configuration for ground-truth tests
  - execution_system: System for running code against tests
  - bayesian_system: System for belief updates
  - ledger_factory: Factory for interaction tracking

OUTPUT:
  - final_code_population: Evolved code population
  - final_test_populations: Dict of evolved test populations

MAIN PROCEDURE: run(problem)
  
  # ═══════════════════════════════════════════════════════════════
  # PHASE 1: INITIALIZATION (Generation 0)
  # ═══════════════════════════════════════════════════════════════
  
  1. CREATE INITIAL POPULATIONS
     a. code_pop ← initialize_code_population(problem)
        - Use code breeding strategy to generate initial code solutions
        - Population size = code_profile.initial_population_size
        - Initial probability = code_profile.initial_prior
     
     b. For each test_type in evolved_test_types:
        evolved_test_pops[test_type] ← initialize_test_population(test_type, problem)
        - Use test breeding strategy for that type
        - May start empty (size=0) for bootstrapped types like differential
        - Initial probability = test_profile.initial_prior
     
     c. public_pop ← create_fixed_population(problem.public_test_cases)
        - Build test class block from dataset test cases
        - Extract individual test methods
        - Fixed probability = 1.0 (ground truth)
     
     d. private_pop ← create_fixed_population(problem.private_test_cases)
        - Same as public but for private evaluation
        - Not used in evolution, only for final assessment
  
  2. BASELINE EXECUTION
     - Execute code_pop against private_pop
     - Log observation matrix for baseline performance
     - Log generation 0 summary statistics
  
  3. INITIALIZE INTERACTION LEDGER
     - ledger ← ledger_factory()
     - Tracks which code-test pairs have been evaluated
     - Prevents double-counting evidence in belief updates
  
  # ═══════════════════════════════════════════════════════════════
  # PHASE 2: MAIN EVOLUTION LOOP
  # ═══════════════════════════════════════════════════════════════
  
  4. FOR each phase in evo_config.schedule.phases:
     
     LOG "Starting Phase: {phase.name}"
     LOG "  Rules: evolve_code={phase.evolve_code}, evolve_tests={phase.evolve_tests}"
     
     5. FOR generation_num from 0 to phase.duration - 1:
        
        epoch ← current loop iteration index across all phases
        
        LOG "═══ EPOCH {epoch} / {last_epoch} [Phase: {phase.name}] ═══"
        
        # ────────────────────────────────────────────────────────
        # STEP A: EXECUTION
        # ────────────────────────────────────────────────────────
        
        6. EXECUTE ALL TEST INTERACTIONS
           interactions ← {}
           
           a. For each test_type in evolved_test_types:
              test_pop ← evolved_test_pops[test_type]
              interaction ← execution_system.execute_tests(code_pop, test_pop)
              interactions[test_type] ← interaction
              LOG observation_matrix for (code_pop, test_pop, test_type)
           
           b. Execute against public tests (ground truth):
              public_interaction ← execution_system.execute_tests(code_pop, public_pop)
              interactions["public"] ← public_interaction
              LOG observation_matrix for (code_pop, public_pop, "public")
           
           # Each interaction contains:
           #   - execution_results: dict[code_id] → ExecutionResult
           #   - observation_matrix: ndarray[n_code, n_test] with pass/fail
        
        # ────────────────────────────────────────────────────────
        # STEP B: BUILD CONTEXT
        # ────────────────────────────────────────────────────────
        
        7. CREATE COEVOLUTION CONTEXT
           context ← CoevolutionContext(
               problem=problem,
               code_population=code_pop,
               test_populations={**evolved_test_pops, "public": public_pop},
               interactions=interactions
           )
           # Context provides complete system state for strategies
        
        # ────────────────────────────────────────────────────────
        # STEP C: BELIEF UPDATES (Cooperative Bayesian Updates)
        # ────────────────────────────────────────────────────────
        
        8. UPDATE BELIEFS WITH BIAS PREVENTION
           
           # Goal: Prevent confirmation bias by updating in specific order
           # Strategy: Anchor with ground truth first, then co-evolve
           
           a. GET UPDATE MASKS
              - For each population, determine which individuals need updates
              - Mask based on: generation_born and new interactions
              - ledger.get_new_interaction_mask(code_ids, test_ids, test_type, update_type)
           
           b. ANCHOR: Update Code with Public Tests (Ground Truth)
              code_ids ← [ind.id for ind in code_pop]
              public_ids ← [ind.id for ind in public_pop]
              public_obs_matrix ← interactions["public"].observation_matrix
              
              mask_public ← ledger.get_new_interaction_mask(
                  code_ids, public_ids, "public", "CODE"
              )
              
              code_posterior_public ← bayesian_system.update_code_beliefs(
                  prior_code_probs=code_pop.probabilities,
                  prior_test_probs=public_pop.probabilities,
                  observation_matrix=public_obs_matrix,
                  code_update_mask_matrix=mask_public,
                  config=public_test_profile.bayesian_config
              )
              
              code_pop.update_probabilities(code_posterior_public)
              ledger.commit_interactions(code_ids, public_ids, "public", "CODE", mask_public)
           
           c. FOR each test_type in evolved_test_types:
              
              test_pop ← evolved_test_pops[test_type]
              test_ids ← [ind.id for ind in test_pop]
              test_obs_matrix ← interactions[test_type].observation_matrix
              test_profile ← evolved_test_profiles[test_type]
              
              # Get masks for this test type
              test_mask ← ledger.get_new_interaction_mask(
                  code_ids, test_ids, test_type, "TEST"
              )
              code_mask ← ledger.get_new_interaction_mask(
                  code_ids, test_ids, test_type, "CODE"
              )
              
              # CALCULATE: Update tests based on anchored code
              test_posterior ← bayesian_system.update_test_beliefs(
                  prior_code_probs=code_posterior_public,  # Use anchored code!
                  prior_test_probs=test_pop.probabilities,
                  observation_matrix=test_obs_matrix,
                  test_update_mask_matrix=test_mask,
                  config=test_profile.bayesian_config
              )
              
              # CALCULATE: Update code based on test priors
              code_posterior_test ← bayesian_system.update_code_beliefs(
                  prior_code_probs=code_pop.probabilities,
                  prior_test_probs=test_pop.probabilities,  # Use test priors
                  observation_matrix=test_obs_matrix,
                  code_update_mask_matrix=code_mask,
                  config=test_profile.bayesian_config
              )
              
              # APPLY: Update probabilities simultaneously
              test_pop.update_probabilities(test_posterior)
              code_pop.update_probabilities(code_posterior_test)
              
              # COMMIT: Mark these interactions as processed
              ledger.commit_interactions(code_ids, test_ids, test_type, "TEST", test_mask)
              ledger.commit_interactions(code_ids, test_ids, test_type, "CODE", code_mask)
        
        # ────────────────────────────────────────────────────────
        # STEP D: EVOLUTION (Selection + Breeding)
        # ────────────────────────────────────────────────────────
        
        9. IF epoch < last_epoch:  # last_epoch = num_epochs - 1
           
           # Only breed if not the final epoch (last epoch is read-only evaluation).
           # Phase rules control whether code/tests evolve
           
           a. IF phase.evolve_code == TRUE:
              
              LOG "Active Phase: Evolving Code Population"
              
              # i. SELECT CODE ELITES
              code_elites ← code_profile.elite_selector.select_elites(
                  population=code_pop,
                  population_config=code_profile.population_config,
                  coevolution_context=context
              )
              LOG "Selected {len(code_elites)} code elites"
              
              # Notify elites of their selection
              FOR each elite in code_elites:
                  elite.notify_selected_as_elite(generation=code_pop.generation)
              
              # ii. BREED CODE OFFSPRING
              num_code_offspring ← calculate_offspring_count(
                  max_size=code_profile.population_config.max_population_size,
                  offspring_rate=code_profile.population_config.offspring_rate,
                  num_elites=len(code_elites)
              )
              
              code_offspring ← code_profile.breeding_strategy.breed(
                  coevolution_context=context,
                  num_offsprings=num_code_offspring
              )
              LOG "Generated {len(code_offspring)} code offspring"
              
              # iii. TRANSITION TO NEXT GENERATION
              new_code_individuals ← code_elites + code_offspring
              
              # Notify removed individuals about death
              notify_removed_individuals(code_pop, new_code_individuals, "code")
              
              # Advance generation
              code_pop.set_next_generation(new_code_individuals)
              LOG "Code Population → generation {code_pop.generation}"
           
           ELSE:
              LOG "Frozen Phase: Code Population is static this generation"
           
           b. IF phase.evolve_tests == TRUE:
              
              LOG "Active Phase: Evolving Test Populations"
              
              # For each evolved test type (unittest, differential, etc.)
              FOR each test_type in evolved_test_types:
                 
                 LOG "Evolving {test_type} test population"
                 
                 test_pop ← evolved_test_pops[test_type]
                 test_profile ← evolved_test_profiles[test_type]
                 
                 # i. SELECT TEST ELITES (typically Pareto-based)
                 test_elites ← test_profile.elite_selector.select_elites(
                     population=test_pop,
                     population_config=test_profile.population_config,
                     coevolution_context=context
                 )
                 LOG "Selected {len(test_elites)} {test_type} test elites"
                 
                 # Notify elites
                 FOR each elite in test_elites:
                     elite.notify_selected_as_elite(generation=test_pop.generation)
                 
                 # ii. BREED TEST OFFSPRING
                 num_test_offspring ← (
                     test_profile.population_config.max_population_size 
                     - len(test_elites)
                 )
                 
                 test_offspring ← test_profile.breeding_strategy.breed(
                     coevolution_context=context,
                     num_offsprings=num_test_offspring
                 )
                 LOG "Generated {len(test_offspring)} {test_type} test offspring"
                 
                 # iii. TRANSITION TO NEXT GENERATION
                 new_test_individuals ← test_elites + test_offspring
                 
                 # Notify removed individuals
                 notify_removed_individuals(test_pop, new_test_individuals, test_type)
                 
                 # Advance generation (also rebuilds test class block)
                 test_pop.set_next_generation(new_test_individuals)
                 LOG "{test_type} Test Population → generation {test_pop.generation}"
           
           ELSE:
              LOG "Frozen Phase: Test Populations are static this generation"
        
        # ────────────────────────────────────────────────────────
        # STEP E: LOGGING
        # ────────────────────────────────────────────────────────
        
        10. LOG GENERATION SUMMARY
            - Code population statistics (size, avg probability, best individual)
            - For each evolved test type:
              - Test population statistics
              - Pareto front information
            - Generation transition summary
        
        11. INCREMENT epoch
  
  # ═══════════════════════════════════════════════════════════════
  # PHASE 3: FINALIZATION
  # ═══════════════════════════════════════════════════════════════
  
  12. FINAL EXECUTION
      - Execute final code_pop against all evolved test populations
      - Execute final code_pop against public_pop
      - LOG all final observation matrices
  
  13. FINAL EVALUATION (Private Tests)
      private_interaction ← execution_system.execute_tests(code_pop, private_pop)
      LOG observation_matrix for (code_pop, private_pop, "private")
      # This is the true evaluation metric (held-out test set)
  
  14. LOG FINAL SURVIVORS
      - For each individual that survived to the end:
        - Log complete lifecycle record
        - Notify individual.notify_survived(final_generation)
      - Log final population statistics
      - Log best solutions
  
  15. RETURN final_code_pop, evolved_test_pops

END PROCEDURE
```

### Key Algorithmic Features

**1. Phased Evolution Schedule**

- Different phases can have different evolution rules
- Example: Phase 1 (test-first), Phase 2 (code-first), Phase 3 (co-evolution)
- Controlled by `phase.evolve_code` and `phase.evolve_tests` flags

**2. Bias Prevention in Belief Updates**

- **Problem**: Tests and code can mutually reinforce incorrect beliefs
- **Solution**: Anchor code with ground-truth public tests first
- **Order**: Public → Code → Tests → Code (with tests)
- Prevents runaway confirmation bias

**3. Interaction Ledger**

- Tracks which code-test pairs have been evaluated
- Prevents double-counting the same evidence
- Supports incremental updates (only new individuals get new evidence)

**4. Multiple Test Population Types**

- Fixed populations: "public" (anchoring), "private" (evaluation)
- Evolved populations: "unittest", "differential", "property", etc.
- Each evolved type has independent configuration and strategies

**5. Elite Preservation + Offspring Generation**

- Elites: Best individuals preserved unchanged (no reproduction operation)
- Offspring: Generated via genetic operations (mutation, crossover, edit, etc.)
- Population size = len(elites) + len(offspring)

**6. Lifecycle Tracking**

- Every individual logs all lifecycle events
- Complete provenance: parents, operations, probability updates
- Enables post-hoc analysis and debugging

**7. Empty Population Support**

- Populations can start with size=0 (bootstrapping)
- Example: Differential tests start empty, grow as divergent code pairs emerge
- All operations handle empty states gracefully (identity operations)

---

## Evolution Workflow Summary

### Generation 0: Initialization

1. Create initial code population from starter code
2. Create initial test populations (evolved and fixed)
3. Run baseline execution on private tests
4. Log initial state

### Generation 1-N: Main Loop

For each phase in schedule:

1. **Execute**: Run code against all test populations
2. **Update Beliefs**: Cooperative Bayesian updates
   - Anchor code with public tests
   - Update evolved tests based on anchored code
   - Update code based on evolved tests
3. **Evolve** (if phase allows):
   - Select elites
   - Breed offspring
   - Transition to next generation
4. **Log**: Generation summary

### Finalization

1. Final execution on all tests
2. Final private test evaluation
3. Log surviving individuals
4. Return final populations

---

## Key Design Decisions

### Why InteractionData?

- **Problem**: Split-brain between ID-keyed results and index-based matrices
- **Solution**: Atomic construction of both in executor
- **Benefit**: Guaranteed alignment, type safety

### Why Multiple Test Populations?

- **Flexibility**: Different test types with different properties
- **Modularity**: Easy to add new test types
- **Realism**: Real testing uses multiple test strategies

### Why Bayesian Belief Updates?

- **Uncertainty**: Handle unreliable tests and code
- **Adaptivity**: Adjust beliefs based on evidence
- **Theoretical Foundation**: Principled probabilistic reasoning

### Why Phased Evolution?

- **Control**: Different strategies at different stages
- **Experimentation**: A/B test different schedules
- **Realism**: Mimic human development (test-first, code-first, etc.)

### Why Lifecycle Logging?

- **Debugging**: Understand what happened and why
- **Analysis**: Post-hoc analysis of evolution
- **Transparency**: Explain decisions to users

---

## Extension Points

The framework is designed for extensibility:

1. **New Operators**: Implement `IOperator` protocol
2. **New Breeding Strategies**: Implement `IBreedingStrategy[T]`
3. **New Selection Strategies**: Implement `IEliteSelectionStrategy[T]` or `IParentSelectionStrategy[T]`
4. **New Test Types**: Add new profiles and configure in orchestrator
5. **New Belief Update Strategies**: Implement `IBeliefUpdater`
6. **New Execution Systems**: Implement `IExecutionSystem`

All without modifying core modules!

---

## Summary

The core modules provide a flexible, well-architected foundation for co-evolutionary algorithm research. Key strengths:

- **Clean Architecture**: Clear separation of concerns
- **Type Safety**: Protocols and type hints throughout
- **Extensibility**: Plugin-based design
- **Testability**: Dependency injection and mocking
- **Observability**: Comprehensive logging and lifecycle tracking
- **Robustness**: Handles edge cases like empty populations
- **Documentation**: Well-commented code with design rationale

This architecture enables rapid experimentation with different evolutionary strategies, operators, and configurations while maintaining code quality and correctness.
