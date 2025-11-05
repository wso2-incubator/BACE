"""
Comprehensive tests for the individual.py module.

This module contains tests for both CodeIndividual and TestIndividual classes,
covering initialization, properties, methods, edge cases, and integration scenarios.
"""

from unittest.mock import MagicMock, patch

import pytest

from common.coevolution.core.individual import CodeIndividual, TestIndividual
from common.coevolution.core.interfaces import Operations


class TestBaseIndividualSharedBehavior:
    """Test common behavior inherited from BaseIndividual."""

    def test_code_individual_initialization(self) -> None:
        """Test CodeIndividual initialization with all parameters."""
        individual = CodeIndividual(
            snippet="def foo(): pass",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        assert individual.snippet == "def foo(): pass"
        assert individual.probability == 0.5
        assert individual.creation_op == "initial"
        assert individual.generation_born == 0
        assert individual.parent_ids == []
        assert len(individual.log) > 0  # Should have birth log entry

    def test_test_individual_initialization(self) -> None:
        """Test TestIndividual initialization with all parameters."""
        individual = TestIndividual(
            snippet="def test_foo(): assert True",
            probability=0.7,
            creation_op="mutation",
            generation_born=5,
            parent_ids=["T1"],
        )

        assert individual.snippet == "def test_foo(): assert True"
        assert individual.probability == 0.7
        assert individual.creation_op == "mutation"
        assert individual.generation_born == 5
        assert individual.parent_ids == ["T1"]
        assert len(individual.log) > 0

    def test_add_to_log(self) -> None:
        """Test logging mechanism."""
        individual = CodeIndividual(
            snippet="def bar(): return 42",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        initial_log_length = len(individual.log)
        individual.add_to_log("Probability updated to 0.8")
        individual.add_to_log("Selected for crossover")

        assert len(individual.log) == initial_log_length + 2
        assert "Probability updated to 0.8" in individual.log
        assert "Selected for crossover" in individual.log

    def test_probability_setter_valid_values(self) -> None:
        """Test probability setter with valid values."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
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
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
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
            creation_op="initial",
            generation_born=10,
            parent_ids=[],
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
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        assert individual.snippet == snippet_text

    def test_creation_op_property(self) -> None:
        """Test creation_op property with different operations."""
        operations: list[Operations] = [
            "initial",
            "mutation",
            "crossover",
            "edit",
            "reproduction",
        ]

        for op in operations:
            individual = CodeIndividual(
                snippet="def test(): pass",
                probability=0.5,
                creation_op=op,
                generation_born=0,
                parent_ids=[],
            )
            assert individual.creation_op == op

    def test_parent_ids_property(self) -> None:
        """Test parent_ids property with different scenarios."""
        # Initial individual - no parents
        individual1 = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )
        assert individual1.parent_ids == []

        # Mutation - one parent
        individual2 = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op="mutation",
            generation_born=1,
            parent_ids=["C0"],
        )
        assert individual2.parent_ids == ["C0"]

        # Crossover - two parents
        individual3 = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op="crossover",
            generation_born=1,
            parent_ids=["C0", "C1"],
        )
        assert individual3.parent_ids == ["C0", "C1"]


class TestCodeIndividual:
    """Test CodeIndividual-specific functionality."""

    def test_id_format(self) -> None:
        """Test ID follows C-prefix pattern."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        assert individual.id.startswith("C")
        assert individual.id[1:].isdigit()

    def test_unique_ids(self) -> None:
        """Test each CodeIndividual instance has unique ID."""
        individuals = [
            CodeIndividual(
                snippet=f"def test_{i}(): pass",
                probability=0.5,
                creation_op="initial",
                generation_born=0,
                parent_ids=[],
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
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        original_id = individual.id

        # Modify various properties
        individual.probability = 0.8
        individual.add_to_log("Modified")

        # ID should remain the same
        assert individual.id == original_id

    def test_repr_format(self) -> None:
        """Test __repr__ string representation."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.75,
            creation_op="crossover",
            generation_born=3,
            parent_ids=["C1", "C2"],
        )

        repr_str = repr(individual)

        assert "CodeIndividual" in repr_str
        assert f"id={individual.id}" in repr_str
        assert "gen=3" in repr_str
        assert "op=crossover" in repr_str
        assert "prob=0.8" in repr_str or "prob=0.7" in repr_str  # Rounded to 1 decimal

    def test_repr_with_different_operations(self) -> None:
        """Test __repr__ correctly shows different operations."""
        operations: list[Operations] = ["initial", "mutation", "edit", "reproduction"]

        for op in operations:
            individual = CodeIndividual(
                snippet="def test(): pass",
                probability=0.5,
                creation_op=op,
                generation_born=0,
                parent_ids=[],
            )
            repr_str = repr(individual)
            assert f"op={op}" in repr_str

    @patch("common.coevolution.core.individual.logger")
    def test_creation_logs_debug_message(self, mock_logger: MagicMock) -> None:
        """Test that individual creation logs debug message."""
        _ = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
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
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        assert individual.id.startswith("T")
        assert individual.id[1:].isdigit()

    def test_unique_ids(self) -> None:
        """Test each TestIndividual instance has unique ID."""
        individuals = [
            TestIndividual(
                snippet=f"def test_{i}(): assert True",
                probability=0.5,
                creation_op="initial",
                generation_born=0,
                parent_ids=[],
            )
            for i in range(10)
        ]

        ids = [ind.id for ind in individuals]
        assert len(ids) == len(set(ids))  # All IDs are unique

    def test_discrimination_initial_state(self) -> None:
        """Test discrimination is initially None."""
        individual = TestIndividual(
            snippet="def test_foo(): assert True",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        assert individual.discrimination is None

    @patch("common.coevolution.core.individual.logger")
    def test_discrimination_getter_warns_when_unset(
        self, mock_logger: MagicMock
    ) -> None:
        """Test accessing discrimination warns when not set."""
        individual = TestIndividual(
            snippet="def test_foo(): assert True",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        # Access discrimination (should warn)
        result = individual.discrimination

        assert result is None
        mock_logger.warning.assert_called()
        call_args = str(mock_logger.warning.call_args)
        assert "discrimination" in call_args.lower()

    def test_discrimination_setter(self) -> None:
        """Test discrimination setter with valid float values."""
        individual = TestIndividual(
            snippet="def test_foo(): assert True",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        # Set discrimination
        individual.discrimination = 0.8
        assert individual.discrimination == 0.8

        # Update discrimination
        individual.discrimination = 0.3
        assert individual.discrimination == 0.3

        # Set to zero
        individual.discrimination = 0.0
        assert individual.discrimination == 0.0

    def test_discrimination_can_be_set_to_none(self) -> None:
        """Test discrimination can be explicitly set to None."""
        individual = TestIndividual(
            snippet="def test_foo(): assert True",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        # Set to a value
        individual.discrimination = 0.7
        assert individual.discrimination == 0.7

        # Reset to None
        individual.discrimination = None
        assert individual.discrimination is None

    def test_repr_with_discrimination_unset(self) -> None:
        """Test __repr__ shows 'None' when discrimination is not set."""
        individual = TestIndividual(
            snippet="def test_foo(): assert True",
            probability=0.6,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        repr_str = repr(individual)

        assert "TestIndividual" in repr_str
        assert f"id={individual.id}" in repr_str
        assert "gen=0" in repr_str
        assert "prob=0.6" in repr_str
        assert "disc=None" in repr_str

    def test_repr_with_discrimination_set(self) -> None:
        """Test __repr__ shows discrimination value when set."""
        individual = TestIndividual(
            snippet="def test_foo(): assert True",
            probability=0.6,
            creation_op="mutation",
            generation_born=2,
            parent_ids=["T0"],
        )

        individual.discrimination = 0.85

        repr_str = repr(individual)

        assert "TestIndividual" in repr_str
        assert f"id={individual.id}" in repr_str
        assert "gen=2" in repr_str
        # Note: TestIndividual.__repr__ doesn't show creation_op
        assert "disc=0.8" in repr_str or "disc=0.9" in repr_str  # Rounded to 1 decimal

    @patch("common.coevolution.core.individual.logger")
    def test_discrimination_setter_logs_trace(self, mock_logger: MagicMock) -> None:
        """Test discrimination setter logs trace message."""
        individual = TestIndividual(
            snippet="def test_foo(): assert True",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        individual.discrimination = 0.75

        mock_logger.trace.assert_called()
        call_args = str(mock_logger.trace.call_args)
        assert "discrimination" in call_args.lower()

    @patch("common.coevolution.core.individual.logger")
    def test_creation_logs_debug_message(self, mock_logger: MagicMock) -> None:
        """Test that individual creation logs debug message."""
        _ = TestIndividual(
            snippet="def test_foo(): assert True",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
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
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        # Manually set same ID for testing (normally wouldn't do this)
        ind2 = CodeIndividual(
            snippet="def different(): pass",
            probability=0.8,
            creation_op="mutation",
            generation_born=5,
            parent_ids=["C0"],
        )
        ind2._id = ind1.id  # Force same ID

        assert ind1 == ind2

    def test_code_individuals_with_different_ids_are_not_equal(self) -> None:
        """Test two CodeIndividuals with different IDs are not equal."""
        ind1 = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        ind2 = CodeIndividual(
            snippet="def test(): pass",  # Same snippet
            probability=0.5,  # Same probability
            creation_op="initial",  # Same operation
            generation_born=0,  # Same generation
            parent_ids=[],  # Same parents
        )

        # Despite everything else being the same, different IDs mean not equal
        assert ind1 != ind2

    def test_test_individuals_with_different_ids_are_not_equal(self) -> None:
        """Test two TestIndividuals with different IDs are not equal."""
        ind1 = TestIndividual(
            snippet="def test_foo(): assert True",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        ind2 = TestIndividual(
            snippet="def test_foo(): assert True",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        assert ind1 != ind2

    def test_code_and_test_individuals_are_not_equal(self) -> None:
        """Test CodeIndividual and TestIndividual are never equal."""
        code_ind = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        test_ind = TestIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        assert code_ind != test_ind

    def test_individual_not_equal_to_non_individual(self) -> None:
        """Test individual is not equal to non-BaseIndividual objects."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
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
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        assert individual.snippet == ""
        assert individual.id.startswith("C")

    def test_very_long_snippet(self) -> None:
        """Test individual with very long code snippet."""
        long_snippet = "def test():\n    " + "x = 1\n    " * 1000 + "return x"
        individual = CodeIndividual(
            snippet=long_snippet,
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        assert individual.snippet == long_snippet
        assert len(individual.snippet) > 10000

    def test_probability_boundary_zero(self) -> None:
        """Test probability at boundary 0.0."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.0,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        assert individual.probability == 0.0

    def test_probability_boundary_one(self) -> None:
        """Test probability at boundary 1.0."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=1.0,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        assert individual.probability == 1.0

    def test_large_generation_number(self) -> None:
        """Test individual with large generation number."""
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op="mutation",
            generation_born=999999,
            parent_ids=["C0"],
        )

        assert individual.generation_born == 999999

    def test_many_parent_ids(self) -> None:
        """Test individual with many parent IDs (unusual but valid)."""
        many_parents = [f"C{i}" for i in range(100)]
        individual = CodeIndividual(
            snippet="def test(): pass",
            probability=0.5,
            creation_op="crossover",
            generation_born=1,
            parent_ids=many_parents,
        )

        assert len(individual.parent_ids) == 100
        assert individual.parent_ids == many_parents

    def test_special_characters_in_snippet(self) -> None:
        """Test snippet with special characters."""
        snippet = 'def test():\n    s = "Hello\\nWorld\\t!"\n    return s'
        individual = CodeIndividual(
            snippet=snippet,
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
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
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
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
                creation_op="initial",
                generation_born=0,
                parent_ids=[],
            )


class TestIndividualCounters:
    """Test ID counter behavior across multiple instances."""

    def test_code_counter_increments(self) -> None:
        """Test CodeIndividual counter increments correctly."""
        individuals = [
            CodeIndividual(
                snippet=f"def test_{i}(): pass",
                probability=0.5,
                creation_op="initial",
                generation_born=0,
                parent_ids=[],
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
                creation_op="initial",
                generation_born=0,
                parent_ids=[],
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
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        test_ind1 = TestIndividual(
            snippet="def test_foo(): assert True",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        code_ind2 = CodeIndividual(
            snippet="def test2(): pass",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        test_ind2 = TestIndividual(
            snippet="def test_bar(): assert True",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
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


class TestIndividualIntegration:
    """Integration tests for realistic scenarios."""

    def test_evolutionary_scenario(self) -> None:
        """Test a realistic evolutionary scenario."""
        # Initial population
        initial = CodeIndividual(
            snippet="def solve(n): return n * 2",
            probability=0.3,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        # Mutation
        mutated = CodeIndividual(
            snippet="def solve(n): return n * 2 + 1",
            probability=0.35,
            creation_op="mutation",
            generation_born=1,
            parent_ids=[initial.id],
        )

        # Crossover
        another = CodeIndividual(
            snippet="def solve(n): return n ** 2",
            probability=0.4,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        offspring = CodeIndividual(
            snippet="def solve(n): return (n * 2) ** 2",
            probability=0.5,
            creation_op="crossover",
            generation_born=2,
            parent_ids=[mutated.id, another.id],
        )

        # Verify relationships
        assert initial.generation_born == 0
        assert mutated.generation_born == 1
        assert offspring.generation_born == 2
        assert mutated.parent_ids == [initial.id]
        assert offspring.parent_ids == [mutated.id, another.id]

    def test_test_evolution_with_discrimination(self) -> None:
        """Test test individual evolution with discrimination tracking."""
        # Initial test
        test1 = TestIndividual(
            snippet="def test_positive(self): assert solve(5) > 0",
            probability=0.5,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        # Set discrimination
        test1.discrimination = 0.8

        # Evolved test
        test2 = TestIndividual(
            snippet="def test_positive(self): assert solve(5) == 10",
            probability=0.6,
            creation_op="edit",
            generation_born=1,
            parent_ids=[test1.id],
        )

        # Updated discrimination
        test2.discrimination = 0.9

        # Verify evolution
        assert test1.discrimination == 0.8
        assert test2.discrimination == 0.9
        assert test2.parent_ids == [test1.id]
        assert test2.probability > test1.probability

    def test_probability_updates_through_evolution(self) -> None:
        """Test probability updates throughout evolution."""
        individual = CodeIndividual(
            snippet="def solve(n): return n",
            probability=0.3,
            creation_op="initial",
            generation_born=0,
            parent_ids=[],
        )

        # Simulate probability updates through generations
        probabilities = [0.3, 0.4, 0.5, 0.6, 0.7]

        for prob in probabilities:
            individual.probability = prob
            assert individual.probability == prob


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
