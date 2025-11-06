# Deprecated Coevolution Implementations

This directory contains **DEPRECATED** implementations that have been replaced by the `core/` module.

**DO NOT USE THESE FILES** - They are kept here temporarily for reference during migration.

## Deprecated Files

⚠️ **All files in this directory represent incorrect implementations.** The old version was fundamentally flawed. Start fresh using `core/interfaces.py` as the specification.

### `orchestrator.py`

- **Replaced by:** `core/orchestrator.py`
- **Reason:** Incorrect orchestrator implementation with wrong architecture
- **Status:** Deprecated as of 2025-01-06

### `population.py`

- **Replaced by:** `core/population.py`
- **Reason:** Incorrect population classes (CodePopulation, TestPopulation) with wrong API
- **Status:** Deprecated as of 2025-01-06

### `reproduction.py`

- **Replaced by:** `core/breeding_strategy.py`
- **Reason:** Incorrect ReproductionStrategy implementation
- **Status:** Deprecated as of 2025-01-06

### `operators.py`

- **To be replaced by:** New implementation following `core/interfaces.py` (ICodeOperator, ITestOperator)
- **Reason:** Incorrect operator implementations
- **Status:** Deprecated as of 2025-01-06

### `selection.py`

- **To be replaced by:** New implementation following `core/interfaces.py` (ISelectionStrategy)
- **Reason:** Incorrect selection strategy implementation
- **Status:** Deprecated as of 2025-01-06

### `bayesian.py`

- **To be replaced by:** New implementation following `core/interfaces.py` (IProbabilityAssigner, IBeliefUpdater)
- **Reason:** Incorrect Bayesian belief update implementation
- **Status:** Deprecated as of 2025-01-06

### `evaluation.py`

- **To be replaced by:** New implementation following `core/interfaces.py` (ICodeTestExecutor, IEvaluator)
- **Reason:** Incorrect evaluation implementation
- **Status:** Deprecated as of 2025-01-06

### `feedback.py`

- **To be replaced by:** New implementation following `core/interfaces.py` (IFeedbackGenerator)
- **Reason:** Incorrect feedback generation implementation
- **Status:** Deprecated as of 2025-01-06

### `config.py`

- **To be replaced by:** New configuration using dataclasses or Pydantic models
- **Reason:** Incorrect configuration approach
- **Status:** Deprecated as of 2025-01-06

## Migration Status

The core module (`src/common/coevolution/core/`) is now the **single source of truth**.

### Already Implemented (in core/)

- ✅ `core/orchestrator.Orchestrator` - Main orchestration logic
- ✅ `core/population.CodePopulation` and `core/population.TestPopulation` - Population management
- ✅ `core/individual.CodeIndividual` and `core/individual.TestIndividual` - Individual representations
- ✅ `core/breeding_strategy.BreedingStrategy` - Generic breeding logic
- ✅ `core/interfaces.py` - Complete Protocol-based interface definitions

### To Be Implemented (from scratch)

Following the interfaces in `core/interfaces.py`:

- ⏳ **Operators**: ICodeOperator, ITestOperator (LLM-based genetic operators)
- ⏳ **Selection**: ISelectionStrategy (fitness-based selection)
- ⏳ **Bayesian**: IProbabilityAssigner, IBeliefUpdater (belief updates)
- ⏳ **Evaluation**: ICodeTestExecutor, IEvaluator (code execution & fitness)
- ⏳ **Feedback**: IFeedbackGenerator (test failure analysis)
- ⏳ **Configuration**: Modern config management (dataclasses/Pydantic)

## Timeline

- **2025-01-06:** All 9 files moved to deprecated/ - old implementation was fundamentally incorrect
- **Current status:** Starting fresh implementations based on `core/interfaces.py`
- **Target removal date:** TBD (after new implementations are complete and tested)
