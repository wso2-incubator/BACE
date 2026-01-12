# coevolution/core/interfaces/context.py
"""
Context objects for passing state through the coevolution system.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .data import InteractionData, Problem

if TYPE_CHECKING:
    from ..population import CodePopulation, TestPopulation


@dataclass
class CoevolutionContext:
    """
    Mutable container for the coevolution system state at a specific generation.

    Holds references to populations and their current interactions. Populations
    are mutated during the evolution cycle (belief updates, generation transitions).
    A new context is created at each generation with fresh interaction data.

    Design principles:
    - One code population (primary population being evolved)
    - Multiple test populations with different roles/types
    - Each test population has interaction data with code population
    - Populations are mutated in-place during evolution
    - Context is reconstructed each generation with new interactions
    - Problem context flows through the system via this container

    Lifecycle within a generation:
        1. Context created with populations + new interaction data
        2. Populations mutated during belief updates
        3. Populations mutated during evolution (if not final generation)
        4. Context discarded, new one created for next generation

    Example:
        context = CoevolutionContext(
            problem=problem,
            code_population=code_pop,  # Will be mutated
            test_populations={
                "public": public_tests,
                "unittest": generated_tests,
                "differential": diff_tests,
            },
            interactions={
                "public": InteractionData(exec_results_1, obs_matrix_1),
                "unittest": InteractionData(exec_results_2, obs_matrix_2),
                "differential": InteractionData(exec_results_3, obs_matrix_3),
            }
        )

        # Mutations happen here
        update_beliefs(context)  # Mutates populations
        evolve_populations(context)  # Mutates populations

    Note: current_generation is accessible via code_population.generation
    """

    problem: Problem
    code_population: "CodePopulation"
    test_populations: dict[
        str, "TestPopulation"
    ]  # Keys: "public", "unittest", "differential", "property", etc.
    interactions: dict[str, InteractionData]  # Same keys as test_populations
