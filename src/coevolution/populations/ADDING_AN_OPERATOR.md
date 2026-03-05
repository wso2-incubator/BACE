# Adding a New Operator

This guide explains how to add a new evolutionary operator (e.g., a new heuristic mutation, a novel crossover technique) to an **existing** population.

The process is explicit and requires touching **four files**:
1. The new operator implementation.
2. The population's operator `__init__.py`.
3. The population's `profile.py` factory (to wire it).
4. The YAML configuration (to assign its weight).

---

## 1. Write the Implementation

Create your new operator inside `populations/<name>/operators/<new_operation>.py`.

It should inherit from `BaseLLMOperator` (or implement `IOperator` directly if it doesn't use the LLM) and implement the `execute` method to return a list of individuals.

```python
# populations/code/operators/semantic.py
from coevolution.strategies.llm_base import BaseLLMOperator, LLMGenerationError, llm_retry
from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import OPERATION_MUTATION, CoevolutionContext

class SemanticHeuristicOperator(BaseLLMOperator[CodeIndividual]):
    def operation_name(self) -> str:
        return OPERATION_MUTATION

    @llm_retry((ValueError, LLMGenerationError))
    def execute(self, context: CoevolutionContext) -> list[CodeIndividual]:
        # Your logic here...
        return []
```

## 2. Export It (1 line)

Make the operator available to the population package by exporting it in `populations/<name>/operators/__init__.py`:

```python
from .semantic import SemanticHeuristicOperator

__all__ = [
    # ... existing exports ...
    "SemanticHeuristicOperator",
]
```

## 3. Wire It into the Profile Factory

Open `populations/<name>/profile.py`. This is where dependencies are injected and the operator is registered with the Breeder.

1. **Import your operator** at the top.
2. **Add a weight argument** to the `create_<name>_profile` function signature (e.g., `semantic_rate: float = 0.2`).
3. **Update the `total_rate` validation** to ensure all rates sum to exactly `1.0`.
4. **Instantiate it** inside the function.
5. **Register it** in the `Breeder` instantiation.

Example changes in `populations/code/profile.py`:

```python
def create_default_code_profile(
    llm_client: LLMClient,
    language_adapter: ILanguage,
    # ... existing rates ...
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.2,
    edit_rate: float = 0.4,
    semantic_rate: float = 0.2,  # <-- 1. Add your new rate
) -> CodeProfile:

    # 2. Update validation
    total_rate = mutation_rate + crossover_rate + edit_rate + semantic_rate
    if not (0.99 <= total_rate <= 1.01):
        raise ValueError(f"Operation rates must sum to 1.0, got {total_rate:.4f}")

    # ... setup default dependencies ...

    # 3. Instantiate your operator
    semantic_op = SemanticHeuristicOperator(llm_client, language_adapter, parent_selector, prob_assigner)

    breeder = Breeder(
        registered_operators=[
            RegisteredOperator(weight=mutation_rate, operator=mutation_op),
            RegisteredOperator(weight=crossover_rate, operator=crossover_op),
            RegisteredOperator(weight=edit_rate, operator=edit_op),
            # 4. Register your operator
            RegisteredOperator(weight=semantic_rate, operator=semantic_op), 
        ],
        llm_workers=llm_workers,
    )
```

*(Note: If you load these profile arguments via structured dataclasses in `factories/orchestrator.py` or `core/interfaces.py`, ensure you also add `semantic_rate: float` there so it can be parsed from the YAML.)*

## 4. Update the YAML Config 

Finally, update your experiment configuration YAML files to provide the new configuration rate.

```yaml
# config.yaml
code_profile:
  mutation_rate: 0.1
  crossover_rate: 0.1
  edit_rate: 0.6
  semantic_rate: 0.2  # <-- Add here! Ensure the sum is exactly 1.0.
```

## Summary Checklist
- [ ] Operator implemented in `operators/<new_operation>.py`
- [ ] Exported in `operators/__init__.py`
- [ ] Added to `create_<name>_profile` parameters
- [ ] Sum validation updated in `profile.py`
- [ ] Registered inside the `Breeder` instance
- [ ] Configuration exposed to YAML parser (if applicable)
- [ ] Weights assigned in standard `.yaml` config files
