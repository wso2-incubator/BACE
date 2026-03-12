"""Property test population profile factory."""

from __future__ import annotations

from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import BayesianConfig, PopulationConfig, TestProfile
from coevolution.core.interfaces.language import ILanguage
from coevolution.strategies.breeding.breeder import Breeder, RegisteredOperator
from coevolution.strategies.selection.elite import TestDiversityEliteSelector
from infrastructure.llm_client import LLMClient
from infrastructure.sandbox import SandboxConfig

from .evaluator import PropertyTestEvaluator
from .operators.initializer import PropertyTestInitializer
from .operators.noop import NoOpOperator
from .types import IOPairCache


def create_property_test_profile(
    llm_client: LLMClient,
    language_adapter: ILanguage,
    sandbox_config: SandboxConfig,
    # Population parameters
    initial_prior: float = 0.5,
    initial_population_size: int = 10,
    max_population_size: int = 30,
    offspring_rate: float = 0.0,
    elitism_rate: float = 1.0,
    # Bayesian parameters
    alpha: float = 0.05,
    beta: float = 0.3,
    gamma: float = 0.25,
    learning_rate: float = 0.1,
    # Performance
    cpu_workers: int = 4,
    enable_multiprocessing: bool = True,
    num_inputs: int = 20,
) -> TestProfile:
    """Create a complete property test population profile.

    A single ``IOPairCache`` is created here and injected into both the
    initializer and the evaluator — this is the shared bridge:

    * The **initializer** stores the generator script in the cache during
      ``initialize(problem)``.
    * The **evaluator** reads the generator script from the cache on the first
      call to ``execute_tests()`` and populates the IOPair table.

    ``public_test_cases`` is **not** a profile-creation-time argument; it is
    extracted from the ``Problem`` instance at operator runtime, consistent with
    the orchestrator lifecycle (profiles are created once globally before any
    ``Problem`` is loaded).
    """
    pop_config = PopulationConfig(
        initial_prior=initial_prior,
        initial_population_size=initial_population_size,
        max_population_size=max_population_size,
        offspring_rate=offspring_rate,
        elitism_rate=elitism_rate,
    )

    python_parser = language_adapter.parser

    # ── Shared cache (initializer → evaluator bridge) ────────────────────────
    io_pair_cache = IOPairCache()

    # ── Initializer ──────────────────────────────────────────────────────────
    initializer = PropertyTestInitializer(
        llm=llm_client,
        parser=python_parser,
        language_name=language_adapter.language,
        pop_config=pop_config,
        sandbox_config=sandbox_config,
        io_pair_cache=io_pair_cache,
    )

    # ── Breeder (no-op — breeding disabled via offspring_rate=0) ────────────
    breeder: Breeder[TestIndividual] = Breeder(
        registered_operators=[
            RegisteredOperator(weight=0.0, operator=NoOpOperator()),
        ],
    )

    # ── Elite selector ───────────────────────────────────────────────────────
    elite_selector: TestDiversityEliteSelector[TestIndividual] = (
        TestDiversityEliteSelector(test_population_key="property")
    )

    # ── Evaluator (execution_system) ─────────────────────────────────────────
    evaluator = PropertyTestEvaluator(
        io_pair_cache=io_pair_cache,
        sandbox_config=sandbox_config,
        composer=language_adapter.composer,
        runtime=language_adapter.runtime,
        enable_multiprocessing=enable_multiprocessing,
        cpu_workers=cpu_workers,
        num_inputs=num_inputs,
    )

    # ── Bayesian config ──────────────────────────────────────────────────────
    bayesian_config = BayesianConfig(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        learning_rate=learning_rate,
    )

    return TestProfile(
        population_config=pop_config,
        breeder=breeder,
        initializer=initializer,
        elite_selector=elite_selector,
        bayesian_config=bayesian_config,
        execution_system=evaluator,
    )


__all__ = ["create_property_test_profile"]
