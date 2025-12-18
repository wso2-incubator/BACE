import logging
import random
import time
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from common.coevolution.breeding_strategies.base_breeding import BaseBreedingStrategy
from common.coevolution.core.interfaces import (
    BaseIndividual,
    CoevolutionContext,
    OperatorRatesConfig,
    Problem,
)


# --- 1. Mocks & Test Implementation ---
@pytest.fixture
def caplog(caplog: pytest.LogCaptureFixture):  # type: ignore
    """
    Overwrites the default caplog fixture to intercept Loguru logs.
    """

    class PropagateHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logging.getLogger(record.name).handle(record)

    handler_id: int = logger.add(
        PropagateHandler(),
        format="{message}",
        level="DEBUG",  # or INFO/ERROR depending on what you test
    )
    yield caplog
    logger.remove(handler_id)


@pytest.fixture
def mock_rates_config() -> OperatorRatesConfig:
    """Returns a config with 50/50 split between two ops."""
    return OperatorRatesConfig(operation_rates={"op_fast": 0.5, "op_slow": 0.5})


@pytest.fixture
def mock_context() -> MagicMock:
    """Mock context object."""
    return MagicMock(spec=CoevolutionContext)


class MockIndividual(BaseIndividual):
    """Minimal concrete individual for testing."""

    def __init__(self, id_val: str) -> None:
        self._id = id_val

    @property
    def id(self) -> str:
        return self._id

    def __repr__(self) -> str:
        return f"Mock({self._id})"


class TestStrategy(BaseBreedingStrategy[MockIndividual]):
    """
    Concrete subclass of BaseBreedingStrategy for testing purposes.
    Allows us to inject arbitrary handler functions.
    """

    def __init__(
        self, op_rates_config: OperatorRatesConfig, max_workers: int = 1
    ) -> None:
        super().__init__(op_rates_config, max_workers)
        # Register handlers that we can swap out during tests
        self._strategies = {"op_fast": self._handle_fast, "op_slow": self._handle_slow}

        # Mocks to control handler behavior
        self.mock_fast_return: list[MockIndividual] = []
        self.mock_slow_return: list[MockIndividual] = []
        self.fast_delay: int | float = 0
        self.slow_delay: int | float = 0

    def initialize_individuals(
        self, problem: Problem
    ) -> tuple[list[MockIndividual], None]:
        """Stub implementation for abstract method."""
        return [MockIndividual("init"), MockIndividual("init2")], None

    def _handle_fast(self, context: CoevolutionContext) -> list[MockIndividual]:
        if self.fast_delay > 0:
            time.sleep(self.fast_delay)
        return list(self.mock_fast_return)

    def _handle_slow(self, context: CoevolutionContext) -> list[MockIndividual]:
        if self.slow_delay > 0:
            time.sleep(self.slow_delay)
        return list(self.mock_slow_return)


@pytest.fixture
def strategy(mock_rates_config: OperatorRatesConfig) -> TestStrategy:
    """Default single-threaded strategy."""
    return TestStrategy(mock_rates_config, max_workers=1)


@pytest.fixture
def parallel_strategy(mock_rates_config: OperatorRatesConfig) -> TestStrategy:
    """Parallel strategy with 4 workers."""
    return TestStrategy(mock_rates_config, max_workers=4)


# --- 2. Unit Tests: Selector & Attempt Wrapper ---


def test_operator_selector_distribution(strategy: TestStrategy) -> None:
    """Verify that operators are selected roughly according to weights."""
    random.seed(42)
    counts = {"op_fast": 0, "op_slow": 0}

    for _ in range(1000):
        op = strategy._operator_selector()
        counts[op] += 1

    # With seed 42 and 0.5/0.5, expect roughly 500 each
    assert 450 < counts["op_fast"] < 550
    assert 450 < counts["op_slow"] < 550


def test_attempt_breeding_success(
    strategy: TestStrategy, mock_context: MagicMock
) -> None:
    """Verify _attempt_breeding returns what the handler returns."""
    # Force selector to pick 'op_fast'
    with patch.object(strategy, "_operator_selector", return_value="op_fast"):
        expected = [MockIndividual("1")]
        strategy.mock_fast_return = expected

        result = strategy._attempt_breeding(mock_context)
        assert result == expected


def test_attempt_breeding_exception_handling(
    strategy: TestStrategy, mock_context: MagicMock
) -> None:
    """Verify exceptions in handlers are caught and return empty list."""
    # Inject a handler that raises an error
    strategy._strategies["op_broken"] = MagicMock(side_effect=ValueError("Boom"))

    with patch.object(strategy, "_operator_selector", return_value="op_broken"):
        # Should not raise exception, but log error and return []
        result = strategy._attempt_breeding(mock_context)
        assert result == []


# --- 3. Integration Tests: breed() Workflow ---


def test_breed_single_threaded_success(
    strategy: TestStrategy, mock_context: MagicMock
) -> None:
    """Verify basic single-threaded loop fills the quota."""
    strategy.mock_fast_return = [MockIndividual("A")]  # Yields 1 per call

    # Force only op_fast
    with patch.object(strategy, "_operator_selector", return_value="op_fast"):
        results = strategy.breed(mock_context, num_offsprings=5)

    assert len(results) == 5
    assert all(isinstance(i, MockIndividual) for i in results)


def test_breed_single_threaded_circuit_breaker(
    strategy: TestStrategy, mock_context: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify loop aborts if handlers keep returning empty lists."""
    strategy.mock_fast_return = []  # Yields nothing

    with patch.object(strategy, "_operator_selector", return_value="op_fast"):
        results = strategy.breed(mock_context, num_offsprings=5)

    # Should return empty list (or partial) and log error
    assert len(results) == 0
    assert "Breeding aborted: exceeded" in caplog.text


def test_breed_parallel_success(
    parallel_strategy: TestStrategy, mock_context: MagicMock
) -> None:
    """Verify parallel execution correctly aggregates results."""
    # Each call returns 1 individual
    parallel_strategy.mock_fast_return = [MockIndividual("P")]
    parallel_strategy.mock_slow_return = [MockIndividual("P")]

    results = parallel_strategy.breed(mock_context, num_offsprings=10)
    assert len(results) == 10


def test_breed_parallel_smart_batching(
    parallel_strategy: TestStrategy, mock_context: MagicMock
) -> None:
    """
    Verify that tasks are submitted in batches, not all at once.
    We check this by ensuring we don't over-generate grossly.
    """
    # High Yield: Each call returns 5 individuals!
    parallel_strategy.mock_fast_return = [MockIndividual("HighYield") for _ in range(5)]

    # We want 10.
    # Batch size is max_workers * 2 = 8.
    # 8 tasks * 5 yield = 40 potentials.
    # But early exit should catch it.

    with patch.object(parallel_strategy, "_operator_selector", return_value="op_fast"):
        results = parallel_strategy.breed(mock_context, num_offsprings=10)

    assert len(results) == 10
    # Ideally, we verify that not too many extra calls were made,
    # but exact counts depend on thread scheduling.


def test_breed_parallel_circuit_breaker(
    parallel_strategy: TestStrategy,
    mock_context: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify parallel loop aborts on consecutive batch failures."""
    parallel_strategy.mock_fast_return = []  # Fail
    parallel_strategy.mock_slow_return = []  # Fail

    results = parallel_strategy.breed(mock_context, num_offsprings=10)

    assert len(results) == 0
    assert "ABORTING BREEDING" in caplog.text


def test_breed_parallel_early_exit(
    parallel_strategy: TestStrategy, mock_context: MagicMock
) -> None:
    """
    Verify that if the first task in a batch satisfies the quota,
    we stop processing and trim the result.
    """
    # Yield matches request in 1 shot
    parallel_strategy.mock_fast_return = [MockIndividual("Instant") for _ in range(10)]

    # Simulate slight delay so multiple tasks might get submitted
    parallel_strategy.fast_delay = 0.01

    with patch.object(parallel_strategy, "_operator_selector", return_value="op_fast"):
        results = parallel_strategy.breed(mock_context, num_offsprings=10)

    assert len(results) == 10
    # The key test here is that it didn't crash or hang
