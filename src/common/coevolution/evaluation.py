"""
Evaluation engine for code-test coevolution.

This module provides functions to generate observation matrices by executing
code populations against test populations in a safe sandbox environment.
"""

import numpy as np
from loguru import logger

from common.code_preprocessing import builders
from common.coevolution.population import CodePopulation, TestPopulation
from common.sandbox import SafeCodeSandbox


def generate_observation_matrix(
    code_population: CodePopulation,
    test_population: TestPopulation,
    sandbox: SafeCodeSandbox,
) -> np.ndarray:
    """
    Generates an observation matrix indicating which code snippets pass which tests.

    Args:
        code_population: Population of code snippets
        test_population: Population of test case snippets

    Returns:
        Numpy array observation matrix with shape (num_code, num_tests)
        where entry (i, j) is 1 if code i passes test j, else 0.
    """
    observation_matrix = np.zeros(
        (code_population.size, test_population.size), dtype=int
    )

    logger.info(
        f"Generating observation matrix: {code_population.size} code × {test_population.size} tests "
        f"= {code_population.size * test_population.size} evaluations"
    )

    for code_idx, (code_snippet, _) in enumerate(code_population):
        logger.debug(f"Evaluating code snippet {code_idx + 1}/{code_population.size}")

        script = builders.build_test_script_for_lcb(
            code_snippet, test_population.test_class_block
        )
        test_results = sandbox.execute_test_script(script)

        for test_idx, test in enumerate(test_results.test_results):
            logger.trace(
                f"Code {code_idx}, Test {test_idx} ({test.name}): {test.status}"
            )

            if test.status == "passed":
                observation_matrix[code_idx, test_idx] = 1
            elif test.status == "failed":
                observation_matrix[code_idx, test_idx] = 0  # failed
            else:
                observation_matrix[code_idx, test_idx] = (
                    0  # Let's treat errors as failures for now
                )

    logger.debug("Completed generating observation matrix.")
    logger.trace(f"Observation Matrix:\n{observation_matrix}")
    return observation_matrix
