import numpy as np
from loguru import logger

from coevolution.core.individual import CodeIndividual, TestIndividual
from coevolution.core.interfaces import CoevolutionContext

# Type alias matching the interface definition
type TestPopulationType = str


class FailingTestSelector:
    """Helper class to select failing tests for code individuals."""

    @staticmethod
    def _rank_selection(probabilities: list[float]) -> int:
        """Select an index based on rank selection strategy."""
        if not probabilities:
            raise ValueError("No probabilities provided for rank selection")

        # Create pairs of (original_index, probability)
        indexed_probs = list(enumerate(probabilities))

        # Sort by probability descending (Higher prob = Higher rank)
        ranked_items = sorted(indexed_probs, key=lambda x: x[1], reverse=True)

        # Assign descending weights based on Rank (n, n-1, ..., 1)
        n = len(ranked_items)
        weights = np.arange(n, 0, -1, dtype=float)
        selection_probs = weights / float(weights.sum())

        # Select a RANK index based on weights
        selected_rank_index = np.random.choice(range(n), p=selection_probs)

        # Return the original index stored in that rank position
        return int(ranked_items[selected_rank_index][0])

    @staticmethod
    def select_k_failing_tests(
        coevolution_context: CoevolutionContext,
        code_individual: CodeIndividual,
        k: int = 10,
    ) -> list[tuple[TestIndividual, TestPopulationType]]:
        """Select up to k failing tests for the given code individual.

        Aggregates failing tests from all test populations and uses rank selection
        to pick up to k tests, favoring tests with higher belief (probability).
        If fewer than k failing tests exist, returns all available failing tests.

        Args:
            coevolution_context: Current coevolution context with populations and interactions.
            code_individual: The code individual for which to select failing tests.
            k: Maximum number of failing tests to select (default: 10).

        Returns:
            A list of tuples (selected_test_individual, test_population_type).
            Empty list if no failing tests are found.
        """
        # List of candidates: (TestIndividual, TestPopulationType)
        candidates: list[tuple[TestIndividual, TestPopulationType]] = []

        # Iterate over all test populations (Unit, Differential, Public, etc.)
        for test_type, test_pop in coevolution_context.test_populations.items():
            if test_type not in coevolution_context.interactions:
                logger.warning(f"No interaction data for test population '{test_type}'")
                continue

            interaction = coevolution_context.interactions[test_type]
            if code_individual.id not in interaction.execution_results:
                logger.warning(
                    f"No execution results for code individual '{code_individual.id}' "
                    f"in test population '{test_type}'"
                )
                continue

            test_results = interaction.execution_results[code_individual.id]

            # Get tests that failed against this code individual using the execution results
            for test_ind in test_pop:
                if test_results[test_ind.id].status in ["failed", "error"]:
                    candidates.append((test_ind, test_type))

        if not candidates:
            return []

        # Limit to available tests if fewer than k exist
        num_to_select = min(k, len(candidates))

        # Extract probabilities from the TestIndividual objects
        probabilities = [ind.probability for ind, _ in candidates]

        # Select k tests using rank selection (with replacement prevention)
        selected_tests: list[tuple[TestIndividual, TestPopulationType]] = []
        remaining_candidates = candidates.copy()
        remaining_probabilities = probabilities.copy()

        for _ in range(num_to_select):
            if not remaining_candidates:
                break

            selected_idx = FailingTestSelector._rank_selection(remaining_probabilities)
            selected_tests.append(remaining_candidates[selected_idx])

            # Remove selected item to prevent duplicates
            remaining_candidates.pop(selected_idx)
            remaining_probabilities.pop(selected_idx)

        return selected_tests
