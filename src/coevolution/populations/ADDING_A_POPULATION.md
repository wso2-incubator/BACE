# Adding a New Population

Each population lives in its own subfolder under `populations/`. Adding one requires only two touches outside the new folder.

---

## Folder Contract

Every population must follow this structure:

```
populations/<name>/
├── __init__.py          # re-exports operators + factory
├── profile.py           # factory function: create_<name>_profile(...)
└── operators/
    ├── __init__.py      # re-exports all operators
    ├── _helpers.py      # optional: private LLM utility mixin
    ├── mutation.py      # optional: <Name>MutationOperator
    ├── crossover.py     # optional: <Name>CrossoverOperator
    ├── edit.py          # optional: <Name>EditOperator
    └── initializer.py  # required: <Name>Initializer
```

For populations with extra helpers (like `differential`), add sibling files:

```
populations/differential/
├── types.py     # shared protocols / dataclasses
├── selector.py  # IFunctionallyEquivalentCodeSelector impl
└── finder.py    # IDifferentialFinder impl
```

---

## Step-by-Step

### 1. Create the folder

```
populations/<name>/
```

### 2. Implement operators

Each operator file inherits from one of the shared base classes in `strategies/llm_base.py`:

| Base class | Use for |
|---|---|
| `BaseLLMOperator[T]` | Mutation, Crossover, Edit operators |
| `BaseLLMInitializer[T]` | Population initializer (Gen 0) |
| `BaseLLMService` | Internal LLM service (non-operator helpers) |

Minimal operator skeleton:

```python
# populations/<name>/operators/mutation.py
from coevolution.strategies.llm_base import BaseLLMOperator, LLMGenerationError, llm_retry
from coevolution.core.individual import <Individual>
from coevolution.core.interfaces import OPERATION_MUTATION, CoevolutionContext

class <Name>MutationOperator(BaseLLMOperator[<Individual>]):
    def operation_name(self) -> str:
        return OPERATION_MUTATION

    @llm_retry((ValueError, LLMGenerationError))
    def execute(self, context: CoevolutionContext) -> list[<Individual>]:
        ...
```

Minimal initializer skeleton:

```python
# populations/<name>/operators/initializer.py
from coevolution.strategies.llm_base import BaseLLMInitializer
from coevolution.core.individual import <Individual>
from coevolution.core.interfaces import Problem

class <Name>Initializer(BaseLLMInitializer[<Individual>]):
    def initialize(self, problem: Problem) -> list[<Individual>]:
        ...
```

### 3. Write the profile factory

```python
# populations/<name>/profile.py
from coevolution.core.interfaces import CodeProfile, PopulationConfig   # or TestProfile
from coevolution.strategies.breeding.breeder import Breeder, RegisteredOperator
from coevolution.strategies.probability.assigner import ProbabilityAssigner
from coevolution.strategies.selection.parent_selection import RouletteWheelParentSelection
from coevolution.strategies.selection.elite import TopKEliteSelector
from .operators.mutation import <Name>MutationOperator
from .operators.initializer import <Name>Initializer

def create_<name>_profile(llm_client, language_adapter, ...) -> CodeProfile:
    pop_config = PopulationConfig(...)
    prob_assigner = ProbabilityAssigner(strategy="min", initial_prior=...)
    parent_selector = RouletteWheelParentSelection()

    mutation_op = <Name>MutationOperator(llm_client, language_adapter, parent_selector, prob_assigner)
    breeder = Breeder(
        registered_operators=[RegisteredOperator(weight=1.0, operator=mutation_op)],
        llm_workers=4,
    )
    initializer = <Name>Initializer(llm=llm_client, language_adapter=language_adapter, pop_config=pop_config)
    elite_selector = TopKEliteSelector()

    return CodeProfile(
        population_config=pop_config,
        breeder=breeder,
        initializer=initializer,
        elite_selector=elite_selector,
    )
```

### 4. Write `__init__.py` files

```python
# populations/<name>/operators/__init__.py
from .mutation import <Name>MutationOperator
from .initializer import <Name>Initializer

__all__ = ["<Name>MutationOperator", "<Name>Initializer"]
```

```python
# populations/<name>/__init__.py
from .profile import create_<name>_profile
from .operators import <Name>MutationOperator, <Name>Initializer

__all__ = ["create_<name>_profile", "<Name>MutationOperator", "<Name>Initializer"]
```

### 5. Register in `factories/__init__.py`

This is the **only file outside the population folder** that needs to change:

```python
# factories/__init__.py  — add one line
from ..populations.<name>.profile import create_<name>_profile
```

And add the name to `__all__`.

---

## Checklist

- [ ] `populations/<name>/operators/_helpers.py` (if needed)
- [ ] `populations/<name>/operators/mutation.py` / `crossover.py` / `edit.py`
- [ ] `populations/<name>/operators/initializer.py`
- [ ] `populations/<name>/operators/__init__.py`
- [ ] `populations/<name>/profile.py`
- [ ] `populations/<name>/__init__.py`
- [ ] `populations/__init__.py` — add `from . import <name>`
- [ ] `factories/__init__.py` — import + add to `__all__`
- [ ] `uv run mypy src/coevolution --ignore-missing-imports` → 0 errors
- [ ] `uv run ruff check src` → All checks passed
