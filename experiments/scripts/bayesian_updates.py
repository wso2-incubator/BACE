"""
Mock script to demonstrate Bayesian updates for code correctness based on test results.
"""

import random

## If your problem grows to have more variables and dependencies (e.g., different types of bugs, multiple code modules,
# tests that depend on each other), writing the logic by hand becomes complicated. pgmpy is a library for
# Probabilistic Graphical Models (PGMs) and is excellent for these cases.

#


def bayesian_update_hypothesis_with_evidence(
    prior_h_p: float,
    evidence_given_hythesis_p: float,
    evidence_given_not_hythesis_p: float,
) -> float:
    """
    Updates the belief of code correctness based on a single passing test.
    Usual baysian update rules written as:
    P(C/E1 )=P(E1 /C)⋅P(C)/(P(E1 /C)⋅P(C)+P(E1 ∣ not C)⋅P( not C)))
    can be written as:
    P(C/E1 )=1 / (1 +  P(E1 ∣ not C)⋅P( not C)/P(E1 ∣ C)⋅P(C)))

    """
    ratio: float = (
        evidence_given_not_hythesis_p
        * (1 - prior_h_p)
        / (evidence_given_hythesis_p * prior_h_p)
    )
    posterior_h_p: float = 1 / (1 + ratio)
    return posterior_h_p


def update_belif_test_results(
    ci_correct_prior: float, test_correctness_p: float, observation: bool
) -> float:
    if observation:
        test_passed_given_correct_code = test_correctness_p
        test_passed_given_incorrect_code = 1 - test_correctness_p
        ci_correct_posterior = bayesian_update_hypothesis_with_evidence(
            prior_h_p=ci_correct_prior,
            evidence_given_hythesis_p=test_passed_given_correct_code,
            evidence_given_not_hythesis_p=test_passed_given_incorrect_code,
        )
    else:
        test_failed_given_correct_code = 1 - test_correctness_p
        test_failed_given_incorrect_code = test_correctness_p
        ci_correct_posterior = bayesian_update_hypothesis_with_evidence(
            prior_h_p=ci_correct_prior,
            evidence_given_hythesis_p=test_failed_given_correct_code,
            evidence_given_not_hythesis_p=test_failed_given_incorrect_code,
        )
    return ci_correct_posterior


def scenario_1() -> None:
    ci_correct_prior: float = 0.6

    observations: list[bool] = [False, False, False]
    test_prior: float = 0.40
    test_correctness_p_list: list[float] = [test_prior, test_prior, test_prior]

    for i, observation in enumerate(observations):
        ci_correct_posterior: float = update_belif_test_results(
            ci_correct_prior, test_correctness_p_list[i], observation
        )
        ci_correct_prior = ci_correct_posterior
        print(f"After Test {i} passes, belief P(Correct) = {ci_correct_prior:.3f}")


def scenario_2() -> None:
    ci_correct_prior: float = 0.6

    # create a list of observations with random True or False
    observations_count: int = 10
    observations: list[bool] = [
        random.choice([True, False]) for _ in range(observations_count)
    ]
    test_correctness_p_list: list[float] = [
        random.uniform(0.1, 0.7) for _ in range(observations_count)
    ]

    for i, observation in enumerate(
        observations
    ):  # fixed bug: observation, i -> i, observation
        print(f"Observation {i} is {observation}")
        ci_correct_posterior = update_belif_test_results(
            ci_correct_prior, test_correctness_p_list[i], observation
        )
        ci_correct_prior = ci_correct_posterior
        print(f"After Test {i} passes, belief P(Correct) = {ci_correct_prior:.3f}")


print("Scenario 1: All tests pass with same reliability")
scenario_1()


print("\nScenario 2: Random test results with varying reliability")
scenario_2()
