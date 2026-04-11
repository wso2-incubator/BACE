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
from ..registry import registry

@registry.code_factory("<name>")  # or @registry.test_factory("<name>")
def create_<name>_profile(llm_client, language_adapter, ...) -> CodeProfile:
    pop_config = PopulationConfig(...)
    # ... construction logic ...
    return CodeProfile(...)
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

__all__ = ["create_<name>_profile"]
```

### 5. Ensure population is loaded

Update `src/coevolution/populations/__init__.py` to import your new subpackage:

```python
# src/coevolution/populations/__init__.py
from . import <name>, ...
```

That's it! `main.py` will now automatically discover and construct your population if it's present in the experiment configuration.

---

## checklist

- [ ] `populations/<name>/operators/initializer.py`
- [ ] `populations/<name>/profile.py` (with `@registry` decorator)
- [ ] `populations/__init__.py` — add `from . import <name>`
- [ ] `uv run python main.py run --config your_config.yaml --dry-run`
