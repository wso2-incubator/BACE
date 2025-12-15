"""
Comprehensive tests for the individual.py module.

This module contains tests for both CodeIndividual and TestIndividual classes,
covering initialization, properties, methods, edge cases, and integration scenarios.
"""

from unittest.mock import MagicMock, patch

import pytest

from common.coevolution.core.individual import CodeIndividual, TestIndividual
from common.coevolution.core.interfaces import (
    OPERATION_CROSSOVER,
    OPERATION_EDIT,
    OPERATION_INITIAL,
    OPERATION_MUTATION,
    OPERATION_REPRODUCTION,
)


class TestBaseIndividualSharedBehavior:
    """Test common behavior inherited from BaseIndividual."""

    def test_code_individual_initialization(self) -> None:
        """Test CodeIndividual initialization with all parameters."""
        individual = CodeIndividual(
            snippet="def foo(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={"code": [], "test": []},
        )

        assert individual.snippet == "def foo(): pass"
        assert individual.probability == 0.5
        assert individual.creation_op == OPERATION_INITIAL
        assert individual.generation_born == 0
        assert individual.parents == {"code": [], "test": []}
        assert len(individual.lifecycle_log) > 0  # Should have birth log entry

    def test_test_individual_initialization(self) -> None:
        """Test TestIndividual initialization with all parameters."""
        individual = TestIndividual(
            snippet="def test_foo(): assert True",
            probability=0.7,
            creation_op=OPERATION_MUTATION,
            generation_born=5,
            parents={"code": [], "test": ["T1"]},
        )

        assert individual.snippet == "def test_foo(): assert True"
        assert individual.probability == 0.7
        assert individual.creation_op == OPERATION_MUTATION
        assert individual.generation_born == 5
        assert individual.parents == {"code": [], "test": ["T1"]}
        assert len(individual.lifecycle_log) > 0

    def test_add_to_log(self) -> None:
        """Test logging mechanism with structured events."""
        individual = CodeIndividual(
            snippet="def bar(): return 42",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={"code": [], "test": []},
        )

        initial_log_length = len(individual.lifecycle_log)

        # Verify initial CREATED event was logged
        assert initial_log_length == 1
        assert individual.lifecycle_log[0].event.value == "created"
        assert individual.lifecycle_log[0].generation == 0
        assert individual.lifecycle_log[0].details["creation_op"] == "initial"

        # Test notify_parent_of logging
        individual.notify_parent_of("C1", OPERATION_CROSSOVER, generation=1)
        individual.notify_selected_as_elite(generation=2)

        assert len(individual.lifecycle_log) == initial_log_length + 2
        assert individual.lifecycle_log[1].event.value == "became_parent"
        assert individual.lifecycle_log[1].details["offspring_id"] == "C1"
        assert individual.lifecycle_log[2].event.value == "selected_as_elite"

    def test_probability_setter_valid_values(self) -> None:
        """Test probability setter with valid values."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        # Test valid probability updates
        individual.probability = 0.0
        assert individual.probability == 0.0

        individual.probability = 1.0
        assert individual.probability == 1.0

        individual.probability = 0.5
        assert individual.probability == 0.5

    def test_probability_setter_invalid_values(self) -> None:
        """Test probability setter with invalid values raises ValueError."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        with pytest.raises(ValueError, match="Probability must be between 0.0 and 1.0"):
            individual.probability = -0.1

        with pytest.raises(ValueError, match="Probability must be between 0.0 and 1.0"):
            individual.probability = 1.1

        with pytest.raises(ValueError, match="Probability must be between 0.0 and 1.0"):
            individual.probability = 2.0

    def test_generation_born_property(self) -> None:
        """Test generation_born property access."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=10,
        )

        assert individual.generation_born == 10

        # Test validation for negative generation
        individual._generation_born = -1
        with pytest.raises(ValueError, match="Generation born cannot be negative"):
            _ = individual.generation_born

    def test_snippet_property(self) -> None:
        """Test snippet property is read-only."""
        snippet_text = "def complex_function():\n    return sum(range(100))"
        individual = CodeIndividual(
            snippet=snippet_text,
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        assert individual.snippet == snippet_text

    def test_creation_op_property(self) -> None:
        """Test creation_op property with different operations."""
        operations: list[str] = [
            OPERATION_INITIAL,
            OPERATION_MUTATION,
            OPERATION_CROSSOVER,
            OPERATION_EDIT,
            OPERATION_REPRODUCTION,
        ]

        for op in operations:
            individual = CodeIndividual(
                snippet="def test(): pass",
                probability=0.5,
                creation_op=op,
                generation_born=0,
            )
            assert individual.creation_op == op

    def test_parent_ids_property(self) -> None:
        """Test parent_ids property with different scenarios."""
        # Initial individual - no parents
        individual1 = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={"code": [], "test": []},
        )
        assert individual1.parents == {"code": [], "test": []}

        # Mutation - one parent
        individual2 = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_MUTATION,
            generation_born=1,
            parents={"code": ["C0"], "test": []},
        )
        assert individual2.parents == {"code": ["C0"], "test": []}

        # Crossover - two parents
        individual3 = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_CROSSOVER,
            generation_born=1,
            parents={"code": ["C0", "C1"], "test": []},
        )
        assert individual3.parents == {"code": ["C0", "C1"], "test": []}


class TestCodeIndividual:
    """Test CodeIndividual-specific functionality."""

    def test_id_format(self) -> None:
        """Test ID follows C-prefix pattern."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        assert individual.id.startswith("C")
        assert individual.id[1:].isdigit()

    def test_unique_ids(self) -> None:
        """Test each CodeIndividual instance has unique ID."""
        individuals = [
            CodeIndividual(
                snippet=f"def test_{i}(): pass",
                probability=0.5,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
            )
            for i in range(10)
        ]

        ids = [ind.id for ind in individuals]
        assert len(ids) == len(set(ids))  # All IDs are unique

    def test_id_persistence(self) -> None:
        """Test ID remains constant throughout individual's lifetime."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        original_id = individual.id

        # Modify probability (one of the mutable properties)
        individual.probability = 0.8

        # Trigger lifecycle events
        individual.notify_parent_of("C1", OPERATION_CROSSOVER, generation=1)

        # ID should remain the same
        assert individual.id == original_id

    def test_repr_format(self) -> None:
        """Test __repr__ string representation."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.75,
            creation_op=OPERATION_CROSSOVER,
            generation_born=3,
            parents={"code": ["C1", "C2"], "test": []},
        )

        repr_str = repr(individual)

        assert "CodeIndividual" in repr_str
        assert f"id={individual.id}" in repr_str
        assert "gen=3" in repr_str
        assert "op=crossover" in repr_str
        assert "prob=0.8" in repr_str or "prob=0.7" in repr_str  # Rounded to 1 decimal

    def test_repr_with_different_operations(self) -> None:
        """Test __repr__ correctly shows different operations."""
        operations: list[str] = [
            OPERATION_INITIAL,
            OPERATION_MUTATION,
            OPERATION_EDIT,
            OPERATION_REPRODUCTION,
        ]

        for op in operations:
            individual = CodeIndividual(
                snippet="def test(): pass",
                probability=0.5,
                creation_op=op,
                generation_born=1,
            )
            repr_str = repr(individual)
            assert f"op={op}" in repr_str

    @patch("common.coevolution.core.individual.logger")
    def test_creation_logs_debug_message(self, mock_logger: MagicMock) -> None:
        """Test that individual creation logs debug message."""
        _ = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        # Verify debug was called
        mock_logger.debug.assert_called()
        # Check the log message contains the repr
        call_args = str(mock_logger.debug.call_args)
        assert "CodeIndividual" in call_args


class TestTestIndividual:
    """Test TestIndividual-specific functionality."""

    def test_id_format(self) -> None:
        """Test ID follows T-prefix pattern."""
        individual = TestIndividual(
            snippet="def test_foo(): assert True",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        assert individual.id.startswith("T")
        assert individual.id[1:].isdigit()

    def test_unique_ids(self) -> None:
        """Test each TestIndividual instance has unique ID."""
        individuals = [
            TestIndividual(
                snippet=f"def test_{i}(): assert True",
                probability=0.5,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
            )
            for i in range(10)
        ]

        ids = [ind.id for ind in individuals]
        assert len(ids) == len(set(ids))  # All IDs are unique

    @patch("common.coevolution.core.individual.logger")
    def test_creation_logs_debug_message(self, mock_logger: MagicMock) -> None:
        """Test that individual creation logs debug message."""
        _ = TestIndividual(
            snippet="def test_foo(): assert True",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        # Verify debug was called
        mock_logger.debug.assert_called()
        call_args = str(mock_logger.debug.call_args)
        assert "TestIndividual" in call_args


class TestIndividualEquality:
    """Test equality comparison between individuals."""

    def test_code_individuals_with_same_id_are_equal(self) -> None:
        """Test two CodeIndividuals with same ID are equal."""
        ind1 = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        # Manually set same ID for testing (normally wouldn't do this)
        ind2 = CodeIndividual(
            snippet="def different(): pass",
            probability=0.8,
            creation_op=OPERATION_MUTATION,
            generation_born=5,
            parents={"code": ["C0"], "test": []},
        )
        ind2._id = ind1.id  # Force same ID

        assert ind1 == ind2

    def test_code_individuals_with_different_ids_are_not_equal(self) -> None:
        """Test two CodeIndividuals with different IDs are not equal."""
        ind1 = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        ind2 = CodeIndividual(
            snippet="def test(): pass",  # Same snippet
            probability=0.5,  # Same probability
            creation_op=OPERATION_INITIAL,  # Same operation
            generation_born=0,  # Same generation
        )

        # Despite everything else being the same, different IDs mean not equal
        assert ind1 != ind2

    def test_test_individuals_with_different_ids_are_not_equal(self) -> None:
        """Test two TestIndividuals with different IDs are not equal."""
        ind1 = TestIndividual(
            snippet="def test_foo(): assert True",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        ind2 = TestIndividual(
            snippet="def test_foo(): assert True",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        assert ind1 != ind2

    def test_code_and_test_individuals_are_not_equal(self) -> None:
        """Test CodeIndividual and TestIndividual are never equal."""
        code_ind = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        test_ind = TestIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        assert code_ind != test_ind

    def test_individual_not_equal_to_non_individual(self) -> None:
        """Test individual is not equal to non-BaseIndividual objects."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        assert individual != "C0"
        assert individual != 42
        assert individual is not None
        assert individual != {"id": "C0"}


class TestIndividualEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_empty_snippet(self) -> None:
        """Test individual with empty code snippet."""
        individual = CodeIndividual(
            snippet="",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        assert individual.snippet == ""
        assert individual.id.startswith("C")

    def test_very_long_snippet(self) -> None:
        """Test individual with very long code snippet."""
        long_snippet = "def test():\n    " + "x = 1\n    " * 1000 + "return x"
        individual = CodeIndividual(
            snippet=long_snippet,
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        assert individual.snippet == long_snippet
        assert len(individual.snippet) > 10000

    def test_probability_boundary_zero(self) -> None:
        """Test probability at boundary 0.0."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.0,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        assert individual.probability == 0.0

    def test_probability_boundary_one(self) -> None:
        """Test probability at boundary 1.0."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=1.0,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        assert individual.probability == 1.0

    def test_large_generation_number(self) -> None:
        """Test individual with large generation number."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_MUTATION,
            generation_born=999999,
            parents={"code": ["C0"], "test": []},
        )

        assert individual.generation_born == 999999

    def test_many_parent_ids(self) -> None:
        """Test individual with many parent IDs (unusual but valid)."""
        many_parents = {"code": [f"C{i}" for i in range(100)], "test": []}
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_CROSSOVER,
            generation_born=1,
            parents=many_parents,
        )

        assert len(individual.parents["code"]) == 100
        assert individual.parents == many_parents

    def test_special_characters_in_snippet(self) -> None:
        """Test snippet with special characters."""
        snippet = 'def test():\n    s = "Hello\\nWorld\\t!"\n    return s'
        individual = CodeIndividual(
            snippet=snippet,
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        assert individual.snippet == snippet

    @pytest.mark.parametrize(
        "probability,should_raise",
        [
            (0.0, False),
            (0.25, False),
            (0.5, False),
            (0.75, False),
            (1.0, False),
            (-0.01, True),
            (-1.0, True),
            (1.01, True),
            (2.0, True),
            (100.0, True),
        ],
    )
    def test_probability_validation_on_setter(
        self, probability: float, should_raise: bool
    ) -> None:
        """Test probability validation via setter with various values."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        if should_raise:
            with pytest.raises(ValueError):
                individual.probability = probability
        else:
            individual.probability = probability
            assert individual.probability == probability

    @pytest.mark.parametrize(
        "probability",
        [-0.01, -1.0, 1.01, 2.0, 100.0],
    )
    def test_probability_validation_on_init(self, probability: float) -> None:
        """
        Test that invalid probabilities are rejected during initialization.

        This test verifies that BaseIndividual.__init__ properly validates
        probability by using the setter, which ensures all invalid probabilities
        are rejected consistently.
        """
        with pytest.raises(ValueError, match="Probability must be between 0.0 and 1.0"):
            _ = CodeIndividual(
                snippet="def test(): pass",
                probability=probability,  # Invalid probability should raise
                creation_op=OPERATION_INITIAL,
                generation_born=0,
            )


class TestIndividualCounters:
    """Test ID counter behavior across multiple instances."""

    def test_code_counter_increments(self) -> None:
        """Test CodeIndividual counter increments correctly."""
        individuals = [
            CodeIndividual(
                snippet=f"def test_{i}(): pass",
                probability=0.5,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
            )
            for i in range(5)
        ]

        ids = [ind.id for ind in individuals]
        # Extract numeric parts
        numbers = [int(id[1:]) for id in ids]

        # Should be sequential (might not start at 0 due to previous tests)
        for i in range(len(numbers) - 1):
            assert numbers[i + 1] == numbers[i] + 1

    def test_test_counter_increments(self) -> None:
        """Test TestIndividual counter increments correctly."""
        individuals = [
            TestIndividual(
                snippet=f"def test_{i}(): assert True",
                probability=0.5,
                creation_op=OPERATION_INITIAL,
                generation_born=0,
            )
            for i in range(5)
        ]

        ids = [ind.id for ind in individuals]
        numbers = [int(id[1:]) for id in ids]

        # Should be sequential
        for i in range(len(numbers) - 1):
            assert numbers[i + 1] == numbers[i] + 1

    def test_code_and_test_counters_are_independent(self) -> None:
        """Test CodeIndividual and TestIndividual have independent counters."""
        code_ind1 = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        test_ind1 = TestIndividual(
            snippet="def test_foo(): assert True",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        code_ind2 = CodeIndividual(
            snippet="def test2(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        test_ind2 = TestIndividual(
            snippet="def test_bar(): assert True",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        # Extract numeric IDs
        code_num1 = int(code_ind1.id[1:])
        code_num2 = int(code_ind2.id[1:])
        test_num1 = int(test_ind1.id[1:])
        test_num2 = int(test_ind2.id[1:])

        # Code individuals should increment independently
        assert code_num2 == code_num1 + 1
        assert test_num2 == test_num1 + 1

        # The actual numbers might be different
        # (i.e., not synchronized)


class TestLifecycleEvents:
    """Test lifecycle event tracking and notifications."""

    def test_notify_died(self) -> None:
        """Test notify_died logs DIED event correctly."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        initial_log_length = len(individual.lifecycle_log)
        individual.notify_died(generation=5)

        assert len(individual.lifecycle_log) == initial_log_length + 1
        died_event = individual.lifecycle_log[-1]
        assert died_event.event.value == "died"
        assert died_event.generation == 5
        assert died_event.details == {}

    def test_notify_survived(self) -> None:
        """Test notify_survived logs SURVIVED event correctly."""
        individual = TestIndividual(
            snippet="def test_foo(): assert True",
            probability=0.7,
            creation_op=OPERATION_MUTATION,
            generation_born=2,
            parents={"code": [], "test": ["T0"]},
        )

        initial_log_length = len(individual.lifecycle_log)
        individual.notify_survived(generation=10)

        assert len(individual.lifecycle_log) == initial_log_length + 1
        survived_event = individual.lifecycle_log[-1]
        assert survived_event.event.value == "survived"
        assert survived_event.generation == 10
        assert survived_event.details == {}

    def test_complete_lifecycle_with_death(self) -> None:
        """Test complete lifecycle ending with death."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        # Lifecycle events
        individual.notify_parent_of("C1", OPERATION_CROSSOVER, generation=1)
        individual.notify_selected_as_elite(generation=2)
        individual.notify_parent_of("C2", OPERATION_MUTATION, generation=3)
        individual.notify_died(generation=4)

        # Verify all events are logged
        assert len(individual.lifecycle_log) == 5  # CREATED + 4 events
        events = [entry.event.value for entry in individual.lifecycle_log]
        assert events == [
            "created",
            "became_parent",
            "selected_as_elite",
            "became_parent",
            "died",
        ]

        # Verify final event is DIED
        assert individual.lifecycle_log[-1].event.value == "died"
        assert individual.lifecycle_log[-1].generation == 4

    def test_complete_lifecycle_with_survival(self) -> None:
        """Test complete lifecycle ending with survival."""
        individual = TestIndividual(
            snippet="def test_bar(): assert True",
            probability=0.6,
            creation_op=OPERATION_EDIT,
            generation_born=1,
            parents={"code": [], "test": ["T0"]},
        )

        # Lifecycle events
        individual.notify_selected_as_elite(generation=2)
        individual.notify_parent_of("T1", OPERATION_REPRODUCTION, generation=3)
        individual.notify_selected_as_elite(generation=4)
        individual.notify_survived(generation=10)

        # Verify all events
        assert len(individual.lifecycle_log) == 5  # CREATED + 4 events
        events = [entry.event.value for entry in individual.lifecycle_log]
        assert events == [
            "created",
            "selected_as_elite",
            "became_parent",
            "selected_as_elite",
            "survived",
        ]

        # Verify final event is SURVIVED
        assert individual.lifecycle_log[-1].event.value == "survived"
        assert individual.lifecycle_log[-1].generation == 10

    def test_lifecycle_events_are_mutually_exclusive(self) -> None:
        """Test that DIED and SURVIVED are mutually exclusive end states."""
        individual1 = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )
        individual1.notify_died(generation=5)

        individual2 = CodeIndividual(
            snippet="def test2(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )
        individual2.notify_survived(generation=10)

        # Check they have different final events
        assert individual1.lifecycle_log[-1].event.value == "died"
        assert individual2.lifecycle_log[-1].event.value == "survived"

        # Individuals should not have both events
        ind1_events = [entry.event.value for entry in individual1.lifecycle_log]
        ind2_events = [entry.event.value for entry in individual2.lifecycle_log]

        assert "died" in ind1_events
        assert "survived" not in ind1_events
        assert "survived" in ind2_events
        assert "died" not in ind2_events


class TestGetCompleteRecord:
    """Test get_complete_record method functionality."""

    def test_get_complete_record_basic(self) -> None:
        """Test get_complete_record returns all basic attributes."""
        individual = CodeIndividual(
            snippet="def foo(): return 42",
            probability=0.75,
            creation_op=OPERATION_MUTATION,
            generation_born=3,
            parents={"code": ["C0"], "test": []},
        )

        record = individual.get_complete_record()

        assert record["id"] == individual.id
        assert record["type"] == "CodeIndividual"
        assert record["snippet"] == "def foo(): return 42"
        assert record["creation_op"] == "mutation"
        assert record["generation_born"] == 3
        assert record["probability"] == 0.75
        assert record["parents"] == {"code": ["C0"], "test": []}

    def test_get_complete_record_with_lifecycle_events(self) -> None:
        """Test get_complete_record includes all lifecycle events."""
        individual = TestIndividual(
            snippet="def test_bar(): assert True",
            probability=0.6,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        # Add lifecycle events
        individual.notify_parent_of("T1", OPERATION_CROSSOVER, generation=1)
        individual.notify_selected_as_elite(generation=2)
        individual.notify_died(generation=3)

        record = individual.get_complete_record()

        # Check lifecycle_events structure
        assert "lifecycle_events" in record
        assert len(record["lifecycle_events"]) == 4  # CREATED + 3 events

        # Verify event structure
        events = record["lifecycle_events"]
        assert all("generation" in event for event in events)
        assert all("event" in event for event in events)
        assert all("details" in event for event in events)

        # Verify event sequence
        event_types = [event["event"] for event in events]
        assert event_types == ["created", "became_parent", "selected_as_elite", "died"]

    def test_get_complete_record_json_serializable(self) -> None:
        """Test that get_complete_record returns JSON-serializable data."""
        import json

        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_CROSSOVER,
            generation_born=2,
            parents={"code": ["C0", "C1"], "test": []},
        )

        individual.notify_selected_as_elite(generation=3)
        individual.notify_survived(generation=10)

        record = individual.get_complete_record()

        # Should be JSON serializable
        try:
            json_str = json.dumps(record)
            reconstructed = json.loads(json_str)
            assert reconstructed["id"] == individual.id
            assert reconstructed["snippet"] == individual.snippet
            assert len(reconstructed["lifecycle_events"]) == 3  # CREATED + 2 events
        except (TypeError, ValueError) as e:
            pytest.fail(f"Record is not JSON serializable: {e}")

    def test_get_complete_record_with_empty_lifecycle(self) -> None:
        """Test get_complete_record when only CREATED event exists."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        record = individual.get_complete_record()

        # Should have only CREATED event
        assert len(record["lifecycle_events"]) == 1
        assert record["lifecycle_events"][0]["event"] == "created"
        assert record["lifecycle_events"][0]["generation"] == 0

    def test_get_complete_record_with_complex_lifecycle(self) -> None:
        """Test get_complete_record with a complex lifecycle history."""
        individual = TestIndividual(
            snippet="def test_complex(): assert True",
            probability=0.7,
            creation_op=OPERATION_EDIT,
            generation_born=1,
            parents={"code": [], "test": ["T0"]},
        )

        # Simulate complex lifecycle
        individual.notify_parent_of("T1", OPERATION_CROSSOVER, generation=2)
        individual.notify_selected_as_elite(generation=3)
        individual.notify_parent_of("T2", OPERATION_MUTATION, generation=4)
        individual.notify_parent_of("T3", OPERATION_REPRODUCTION, generation=5)
        individual.notify_selected_as_elite(generation=6)
        individual.notify_parent_of("T4", OPERATION_EDIT, generation=7)
        individual.notify_survived(generation=10)

        record = individual.get_complete_record()

        # Should have all events
        assert len(record["lifecycle_events"]) == 8  # CREATED + 7 events

        # Verify specific events
        events = record["lifecycle_events"]
        assert events[0]["event"] == "created"
        assert events[1]["event"] == "became_parent"
        assert events[1]["details"]["offspring_id"] == "T1"
        assert events[2]["event"] == "selected_as_elite"
        assert events[-1]["event"] == "survived"
        assert events[-1]["generation"] == 10

    def test_get_complete_record_for_test_individual(self) -> None:
        """Test get_complete_record correctly identifies TestIndividual type."""
        individual = TestIndividual(
            snippet="def test_foo(): assert True",
            probability=0.8,
            creation_op=OPERATION_MUTATION,
            generation_born=5,
            parents={"code": [], "test": ["T4"]},
        )

        record = individual.get_complete_record()

        assert record["type"] == "TestIndividual"
        assert record["id"].startswith("T")

    def test_get_complete_record_immutability(self) -> None:
        """Test that modifying returned record doesn't affect individual."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        record = individual.get_complete_record()
        original_snippet = individual.snippet

        # Modify the record
        record["snippet"] = "def modified(): pass"
        record["probability"] = 0.99

        # Individual should be unchanged
        assert individual.snippet == original_snippet
        assert individual.probability == 0.5

    def test_get_complete_record_with_updated_probability(self) -> None:
        """Test get_complete_record reflects current probability, not initial."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.3,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        # Update probability multiple times
        individual.probability = 0.5
        individual.probability = 0.7

        record = individual.get_complete_record()

        # Should reflect current probability
        assert record["probability"] == 0.7

        # But CREATED event should have initial probability
        created_event = record["lifecycle_events"][0]
        assert created_event["details"]["probability"] == 0.3


class TestIndividualIntegration:
    """Integration tests for realistic scenarios."""

    def test_evolutionary_scenario(self) -> None:
        """Test a realistic evolutionary scenario."""
        # Initial population
        initial = CodeIndividual(
            snippet="def solve(n): return n * 2",
            probability=0.3,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        # Mutation
        mutated = CodeIndividual(
            snippet="def solve(n): return n * 2 + 1",
            probability=0.35,
            creation_op=OPERATION_MUTATION,
            generation_born=1,
            parents={"code": [initial.id], "test": []},
        )

        # Crossover
        another = CodeIndividual(
            snippet="def solve(n): return n ** 2",
            probability=0.4,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
            parents={"code": [], "test": []},
        )

        offspring = CodeIndividual(
            snippet="def solve(n): return (n * 2) ** 2",
            probability=0.5,
            creation_op=OPERATION_CROSSOVER,
            generation_born=2,
            parents={"code": [mutated.id, another.id], "test": []},
        )

        # Verify relationships
        assert initial.generation_born == 0
        assert mutated.generation_born == 1
        assert offspring.generation_born == 2
        assert mutated.parents == {"code": [initial.id], "test": []}
        assert offspring.parents == {"code": [mutated.id, another.id], "test": []}

    def test_probability_updates_through_evolution(self) -> None:
        """Test probability updates throughout evolution."""
        individual = CodeIndividual(
            snippet="def solve(n): return n",
            probability=0.3,
            creation_op=OPERATION_INITIAL,
            generation_born=0,
        )

        # Simulate probability updates through generations
        probabilities = [0.3, 0.4, 0.5, 0.6, 0.7]

        for prob in probabilities:
            individual.probability = prob
            assert individual.probability == prob


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
