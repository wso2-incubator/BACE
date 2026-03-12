"""End-to-end integration tests for the property test population.

Tests the full workflow:
  1. PropertyTestInitializer calls a real LLM (gpt-5-mini) to generate:
       a. A Python input-generator script (stored in IOPairCache)
       b. A set of property test snippets — TestIndividual objects
  2. PropertyTestEvaluator uses the cached generator script to:
       a. Run the generator script → input strings
       b. Execute each code individual against those inputs → IOPairs
       c. Evaluate each property snippet against the IOPairs → observation matrix

The sorting problem is used as the reference scenario because:
  - It has clear, verifiable properties (non-decreasing, length-preserved, ...)
  - A correct implementation (sorted(lst)) should satisfy all sensible properties
  - A buggy identity-return (return lst) will fail on unsorted inputs

Requirements:
  - OPENAI_API_KEY must be set in the environment
  - Run with: pytest tests/integration/test_property_workflow_e2e.py -v -s

Note on fixture scoping: ``initialized_individuals`` is module-scoped so that
the LLM is only called once per test session.  The same ``IOPairCache``
instance is shared between the initializer and evaluator — this mirrors the
runtime behaviour of the coevolutionary loop.
"""

from __future__ import annotations

import os
from typing import TypeAlias

import pytest

from coevolution.core.individual import CodeIndividual, TestIndividual
from coevolution.core.interfaces.data import Problem, Test
from coevolution.core.interfaces.initializer import IPopulationInitializer
from coevolution.core.interfaces.profiles import TestProfile
from coevolution.core.population import CodePopulation, TestPopulation
from coevolution.populations.property.evaluator import PropertyTestEvaluator
from coevolution.populations.property.operators.validator import validate_property_test
from coevolution.populations.property.profile import create_property_test_profile
from coevolution.populations.property.types import IOPairCache
from infrastructure.languages.python import PythonLanguage
from infrastructure.llm_client import create_llm_client
from infrastructure.llm_client.base import LLMClient
from infrastructure.sandbox import SandboxConfig, create_sandbox

# ── Skip guard ────────────────────────────────────────────────────────────────

REQUIRES_OPENAI = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping real-LLM integration tests",
)

# ── Problem definition ────────────────────────────────────────────────────────

SORT_PROBLEM = Problem(
    question_title="sort_list",
    question_content=(
        "Given a list of integers, return the list sorted in non-decreasing order.\n"
        "Example: sort([3, 1, 2]) should return [1, 2, 3]."
    ),
    question_id="integration/sort_list",
    starter_code="def sort(lst: list[int]) -> list[int]:\n    ...\n",
    public_test_cases=[
        Test(input="[3, 1, 2]", output="[1, 2, 3]"),
        Test(input="[5, 5, 1]", output="[1, 5, 5]"),
        Test(input="[42]", output="[42]"),
    ],
    private_test_cases=[],
)

# ── Code snippets under test ──────────────────────────────────────────────────

# Correct: returns a sorted copy of the input
CORRECT_SORT = "def sort(lst):\n    return sorted(lst)\n"

# Buggy: returns the list unchanged — fails on any unsorted input
BUGGY_SORT = "def sort(lst):\n    return lst\n"

# ── LLM response printer ─────────────────────────────────────────────────────


class PrintingLLMClient(LLMClient):
    """Thin proxy that prints every LLM response to stdout as it arrives.

    Run tests with ``-s`` (or ``--capture=no``) to see the output live::

        pytest tests/integration/test_property_workflow_e2e.py -v -s
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner

    def generate(self, prompt: object, **kwargs: object) -> str:
        response = self._inner.generate(prompt, **kwargs)
        print(f"\n{'═' * 72}")
        print("[LLM RESPONSE]")
        print(response)
        print(f"{'═' * 72}\n")
        return response


# ── Helper to build a CodePopulation ─────────────────────────────────────────

IndividualsList: TypeAlias = list[TestIndividual]


def _make_code_pop(snippets: list[str], prob: float = 0.5) -> CodePopulation:
    """Wrap plain code strings into a CodePopulation."""
    individuals = [
        CodeIndividual(
            snippet=s,
            probability=prob,
            creation_op="initial",
            generation_born=0,
        )
        for s in snippets
    ]
    return CodePopulation(individuals)


# ── Module-scoped fixtures ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def llm_client() -> LLMClient:
    """Create the real OpenAI gpt-5-mini client, wrapped to print every response."""
    inner = create_llm_client(
        provider="openai",
        model="gpt-5-mini",
        reasoning_effort="minimal",
    )
    return PrintingLLMClient(inner)


@pytest.fixture(scope="module")
def sandbox_config() -> SandboxConfig:
    return SandboxConfig(language="python", timeout=30)


@pytest.fixture(scope="module")
def property_profile(
    llm_client: LLMClient, sandbox_config: SandboxConfig
) -> TestProfile:
    """Full property test profile (initializer + evaluator sharing an IOPairCache)."""
    return create_property_test_profile(
        llm_client=llm_client,
        language_adapter=PythonLanguage(),
        sandbox_config=sandbox_config,
        initial_population_size=5,
        max_population_size=10,
        num_inputs=5,
        enable_multiprocessing=False,  # simpler for integration tests
    )


@pytest.fixture(scope="module")
def evaluator(property_profile: TestProfile) -> PropertyTestEvaluator:
    """Typed evaluator extracted from the profile — avoids repeated isinstance casts."""
    assert isinstance(property_profile.execution_system, PropertyTestEvaluator)
    return property_profile.execution_system


@pytest.fixture(scope="module")
def initialized_individuals(property_profile: TestProfile) -> IndividualsList:
    """Call initialize() once and reuse the result across tests in this module.

    Side-effect: populates the IOPairCache with the generator script so that
    the evaluator fixtures below can also use the same cache.
    """
    initializer: IPopulationInitializer[TestIndividual] = property_profile.initializer
    individuals = initializer.initialize(SORT_PROBLEM)
    return individuals


# ── Initializer E2E ────────────────────────────────────────────────────────────


@REQUIRES_OPENAI
class TestPropertyInitializerE2E:
    """End-to-end tests for PropertyTestInitializer using a real OpenAI LLM."""

    def test_returns_at_least_one_individual(
        self, initialized_individuals: IndividualsList
    ) -> None:
        """The LLM must generate at least one valid property test individual."""
        assert len(initialized_individuals) >= 1, (
            "Expected at least one property test individual but got 0. "
            "The LLM may have failed to produce any valid property functions."
        )

    def test_all_snippets_define_a_property_function(
        self, initialized_individuals: IndividualsList
    ) -> None:
        """Every returned snippet must follow the 'def property_' naming convention."""
        for ind in initialized_individuals:
            first_line = ind.snippet.lstrip().split("\n")[0]
            assert first_line.startswith("def property_"), (
                f"Snippet does not start with 'def property_'.\n"
                f"First line: {first_line!r}\n"
                f"Full snippet:\n{ind.snippet}"
            )

    def test_all_snippets_accept_input_and_output_parameters(
        self, initialized_individuals: IndividualsList
    ) -> None:
        """Each property function signature must accept 'input' and 'output' parameters."""
        for ind in initialized_individuals:
            defn_line = ind.snippet.lstrip().split("\n")[0]
            assert "input" in defn_line and "output" in defn_line, (
                f"Property function signature missing 'input' or 'output' parameter.\n"
                f"Definition line: {defn_line!r}"
            )

    def test_generator_script_stored_in_io_pair_cache(
        self, evaluator: PropertyTestEvaluator, initialized_individuals: IndividualsList
    ) -> None:
        """The input-generator script must be cached after initialize() returns."""
        # Force the fixture to run (module-scoped, already evaluated)
        _ = initialized_individuals

        cache: IOPairCache = evaluator.io_pair_cache
        script = cache.get_generator_script()

        assert script is not None, (
            "No generator script found in the IOPairCache after initialize(). "
            "The initializer must store the gen-inputs script in the cache."
        )
        assert "generate_inputs" in script, (
            f"Generator script is missing the 'generate_inputs' function.\n"
            f"Script:\n{script}"
        )

    def test_each_property_validates_against_all_public_test_cases(
        self,
        initialized_individuals: IndividualsList,
    ) -> None:
        """Every generated property must pass all public test cases for the sort problem.

        validate_property_test() checks the property against the _expected outputs_
        from the public test cases — a well-formed property must return True for
        every (input, correct_output) pair.
        """
        sandbox = create_sandbox(SandboxConfig(language="python", timeout=10))
        for ind in initialized_individuals:
            result = validate_property_test(
                snippet=ind.snippet,
                public_test_cases=SORT_PROBLEM.public_test_cases,
                sandbox=sandbox,
            )
            assert result is True, (
                f"Property failed validation against public test cases.\n"
                f"Snippet:\n{ind.snippet}"
            )

    def test_individuals_have_valid_initial_probability(
        self, initialized_individuals: IndividualsList
    ) -> None:
        """All returned individuals must have a probability in [0, 1]."""
        for ind in initialized_individuals:
            assert 0.0 <= ind.probability <= 1.0, (
                f"Individual probability {ind.probability} is outside [0, 1].\n"
                f"Snippet:\n{ind.snippet}"
            )


# ── Evaluator E2E ──────────────────────────────────────────────────────────────


@REQUIRES_OPENAI
class TestPropertyEvaluatorE2E:
    """End-to-end tests for PropertyTestEvaluator.

    Depends on ``initialized_individuals`` (module-scoped) to populate the
    IOPairCache so that the evaluator can resolve inputs on its first call.
    """

    def test_observation_matrix_has_correct_shape(
        self,
        initialized_individuals: IndividualsList,
        evaluator: PropertyTestEvaluator,
    ) -> None:
        """The returned observation matrix must have shape (n_codes, n_tests)."""
        code_pop = _make_code_pop([CORRECT_SORT, BUGGY_SORT])
        test_pop = TestPopulation(initialized_individuals)

        result = evaluator.execute_tests(code_pop, test_pop)

        expected_shape = (code_pop.size, test_pop.size)
        assert result.observation_matrix.shape == expected_shape, (
            f"Expected observation matrix shape {expected_shape}, "
            f"got {result.observation_matrix.shape}"
        )

    def test_observation_matrix_contains_only_zeros_and_ones(
        self,
        initialized_individuals: IndividualsList,
        evaluator: PropertyTestEvaluator,
    ) -> None:
        """Observation matrix entries must be binary (0 = fail, 1 = pass)."""
        code_pop = _make_code_pop([CORRECT_SORT, BUGGY_SORT])
        test_pop = TestPopulation(initialized_individuals)

        result = evaluator.execute_tests(code_pop, test_pop)
        matrix = result.observation_matrix

        unique_values = set(matrix.flatten().tolist())
        assert unique_values.issubset({0, 1}), (
            f"Observation matrix must contain only 0 and 1, got values: {unique_values}"
        )

    def test_correct_sort_passes_at_least_one_property(
        self,
        initialized_individuals: IndividualsList,
        evaluator: PropertyTestEvaluator,
    ) -> None:
        """A correct sorting implementation must pass at least one LLM-generated property."""
        code_pop = _make_code_pop([CORRECT_SORT])
        test_pop = TestPopulation(initialized_individuals)

        result = evaluator.execute_tests(code_pop, test_pop)
        correct_passes = int(result.observation_matrix[0].sum())

        assert correct_passes >= 1, (
            f"Correct sort passed 0 out of {test_pop.size} property tests.\n"
            f"Generated properties:\n"
            + "\n--- \n".join(ind.snippet for ind in initialized_individuals)
        )

    def test_correct_sort_passes_more_properties_than_buggy_sort(
        self,
        initialized_individuals: IndividualsList,
        evaluator: PropertyTestEvaluator,
    ) -> None:
        """Correct sort should pass at least as many properties as a buggy no-op sort.

        The buggy identity-return sort will fail any non-trivial property that
        checks ordering (e.g., non-decreasing, ascending, etc.) on the input
        [3, 1, 2] since it is returned unchanged.
        """
        code_pop = _make_code_pop([CORRECT_SORT, BUGGY_SORT])
        test_pop = TestPopulation(initialized_individuals)

        result = evaluator.execute_tests(code_pop, test_pop)
        matrix = result.observation_matrix

        correct_score = int(matrix[0].sum())
        buggy_score = int(matrix[1].sum())

        assert correct_score >= buggy_score, (
            f"Expected correct sort (score={correct_score}) to pass at least as many "
            f"properties as buggy sort (score={buggy_score}).\n"
            f"Observation matrix (row 0 = correct, row 1 = buggy):\n{matrix}"
        )

    def test_execution_results_keyed_by_code_id(
        self,
        initialized_individuals: IndividualsList,
        evaluator: PropertyTestEvaluator,
    ) -> None:
        """execution_results must have one top-level entry per code individual."""
        code_pop = _make_code_pop([CORRECT_SORT, BUGGY_SORT])
        test_pop = TestPopulation(initialized_individuals)

        result = evaluator.execute_tests(code_pop, test_pop)
        code_ids = {ind.id for ind in code_pop}

        assert set(result.execution_results.results.keys()) == code_ids, (
            f"execution_results keys {set(result.execution_results.results)} "
            f"don't match code individual IDs {code_ids}"
        )

    def test_execution_results_nested_by_test_id(
        self,
        initialized_individuals: IndividualsList,
        evaluator: PropertyTestEvaluator,
    ) -> None:
        """For each code individual, execution_results must have one entry per test."""
        code_pop = _make_code_pop([CORRECT_SORT])
        test_pop = TestPopulation(initialized_individuals)

        result = evaluator.execute_tests(code_pop, test_pop)

        for code_id, test_results in result.execution_results.results.items():
            assert len(test_results) == test_pop.size, (
                f"Code {code_id!r}: expected {test_pop.size} test results, "
                f"got {len(test_results)}"
            )


# ── Full pipeline smoke test ───────────────────────────────────────────────────


@REQUIRES_OPENAI
class TestFullPipelineE2E:
    """Smoke test: initializer → evaluator in one shot using a fresh, isolated profile.

    Creates its own IOPairCache (via a new profile instance) so it doesn't
    share state with the module-scoped fixtures above.
    """

    def test_initialize_then_evaluate_end_to_end(
        self, llm_client: LLMClient, sandbox_config: SandboxConfig
    ) -> None:
        """Full pipeline: LLM generates properties, evaluator scores code against them."""
        # Fresh profile — isolated IOPairCache
        profile = create_property_test_profile(
            llm_client=llm_client,
            language_adapter=PythonLanguage(),
            sandbox_config=sandbox_config,
            initial_population_size=3,
            max_population_size=5,
            num_inputs=3,
            enable_multiprocessing=False,
        )

        # Step 1 — initialization (LLM call)
        individuals = profile.initializer.initialize(SORT_PROBLEM)
        assert len(individuals) >= 1, (
            "Initialization returned no property test individuals."
        )

        # Step 2 — evaluation
        assert isinstance(profile.execution_system, PropertyTestEvaluator)
        code_pop = _make_code_pop([CORRECT_SORT])
        test_pop = TestPopulation(individuals)
        result = profile.execution_system.execute_tests(code_pop, test_pop)

        # Structural integrity
        assert result.observation_matrix.shape == (1, len(individuals)), (
            f"Unexpected matrix shape {result.observation_matrix.shape}"
        )
        code_id = list(result.execution_results.results.keys())[0]
        assert len(result.execution_results.results[code_id]) == len(individuals), (
            "Mismatch between execution_results entries and number of test individuals."
        )
