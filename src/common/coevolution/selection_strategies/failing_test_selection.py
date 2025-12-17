import numpy as np
from loguru import logger

from common.coevolution.core.individual import CodeIndividual, TestIndividual
from common.coevolution.core.interfaces import CoevolutionContext

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
    def select_failing_test(
        coevolution_context: CoevolutionContext,
        code_individual: CodeIndividual,
    ) -> tuple[TestIndividual, TestPopulationType] | None:
        """Select a failing test for the given code individual.

        Aggregates failing tests from all test populations and uses rank selection
        to pick one, favoring tests with higher belief (probability).

        Args:
            coevolution_context: Current coevolution context with populations and interactions.
            code_individual: The code individual for which to select a failing test.

        Returns:
            A tuple of (selected_test_individual, test_population_type) if a failing test is found,
            otherwise None.
        """
        # 1. Get Code Index
        code_index = coevolution_context.code_population.get_index_of_individual(
            code_individual
        )
        if code_index == -1:
            logger.warning(
                f"CodeIndividual {code_individual.id} not found in population."
            )
            return None

        # List of candidates: (TestIndividual, TestPopulationType)
        candidates: list[tuple[TestIndividual, TestPopulationType]] = []

        # 2. Iterate over all test populations (Unit, Differential, Public, etc.)
        for test_type, test_pop in coevolution_context.test_populations.items():
            if test_type not in coevolution_context.interactions:
                logger.warning(f"No interaction data for test population '{test_type}'")
                continue

            interaction = coevolution_context.interactions[test_type]

            # Get tests that failed against this code individual using the observation matrix
            for test_idx, test_individual in enumerate(test_pop.individuals):
                if interaction.observation_matrix[code_index][test_idx] == 0:
                    candidates.append((test_individual, test_type))

        if not candidates:
            return None

        # 3. Rank Selection
        # Extract probabilities from the TestIndividual objects
        probabilities = [ind.probability for ind, _ in candidates]
        selected_idx = FailingTestSelector._rank_selection(probabilities)

        return candidates[selected_idx]
