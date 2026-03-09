"""FunctionallyEqSelector — groups code individuals by identical behavior vectors."""

from __future__ import annotations

import numpy as np
from loguru import logger

from coevolution.core.interfaces import CoevolutionContext
from .types import FunctionallyEquivGroup, IFunctionallyEquivalentCodeSelector


class FunctionallyEqSelector(IFunctionallyEquivalentCodeSelector):
    """Selects functionally equivalent code snippets based on identical behavior vectors."""

    def select_functionally_equivalent_codes(
        self,
        coevolution_context: CoevolutionContext,
    ) -> list[FunctionallyEquivGroup]:
        """
        Select groups of functionally equivalent code snippets.

        Strategy:
        1. Stack all observation matrices horizontally to create a 'behavior signature'
           for each code individual (Row = Code, Col = All Tests Combined).
        2. Use np.unique to group rows that are identical.
        3. Map these groups back to CodeIndividual objects and identify their passing tests.
        """
        func_eq_groups: list[FunctionallyEquivGroup] = []
        code_pop = coevolution_context.code_population

        if code_pop.size == 0:
            logger.warning(
                "No code individuals to evaluate for functional equivalence."
            )
            return []

        matrices_to_stack = []
        test_types = sorted(coevolution_context.interactions.keys())

        for t_type in test_types:
            interaction = coevolution_context.interactions[t_type]
            obs_matrix = interaction.observation_matrix

            if obs_matrix.shape[0] != code_pop.size:
                logger.error(
                    f"Mismatch in matrix rows for '{t_type}'. "
                    f"Expected {code_pop.size}, got {obs_matrix.shape[0]}."
                )
                return []
            matrices_to_stack.append(obs_matrix)

        if not matrices_to_stack:
            return [
                FunctionallyEquivGroup(
                    code_individuals=list(code_pop.individuals),
                    passing_test_individuals={},
                )
            ]

        combined_matrix = np.hstack(matrices_to_stack)
        _, inverse_indices = np.unique(combined_matrix, axis=0, return_inverse=True)

        groups_map: dict[int, list[int]] = {}
        for code_idx, pattern_id in enumerate(inverse_indices):
            groups_map.setdefault(pattern_id, []).append(code_idx)

        for pattern_id, code_indices in groups_map.items():
            group_codes = [code_pop[i] for i in code_indices]
            representative_idx = code_indices[0]
            passing_tests_map = {}

            for t_type in test_types:
                interaction = coevolution_context.interactions[t_type]
                row = interaction.observation_matrix[representative_idx]
                passing_indices = np.where(row == 1)[0]

                if passing_indices.size > 0:
                    test_pop = coevolution_context.test_populations[t_type]
                    passing_tests_map[t_type] = [test_pop[i] for i in passing_indices]

            func_eq_groups.append(
                FunctionallyEquivGroup(
                    code_individuals=group_codes,
                    passing_test_individuals=passing_tests_map,
                )
            )

        return func_eq_groups


__all__ = ["FunctionallyEqSelector"]
