import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Add src to path so we can import coevolution
# Assuming experiments/scripts/bayesian/run_simulation.py
# ROOT is ../../../
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../src"))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from coevolution.services.bayesian import BayesianSystem
except ImportError:
    # Fallback if the path setup didn't work as expected or running from different cwd
    # Try adding just "src" if running from root
    sys.path.append(os.path.abspath("src"))
    from coevolution.services.bayesian import BayesianSystem


# Define a Config class that mimics BayesianConfig but allows 0.0 values
@dataclass
class SimConfig:
    alpha: float
    beta: float
    gamma: float
    learning_rate: float = 1.0


def run_simulation():
    # Parameters
    alpha = 0.0
    gamma = 0.0

    # Ranges
    code_priors = [round(x * 0.1, 1) for x in range(1, 10)]  # 0.1 to 0.9
    betas = [round(x * 0.1, 1) for x in range(0, 6)]  # 0.0 to 0.5 inclusive
    learning_rates = [0.05] + [
        round(x * 0.1, 1) for x in range(1, 11)
    ]  # 0.05, 0.1 to 1.0

    # We will run the same pattern of scenarios for public test counts 1..4.
    # For a given n tests create scenarios for k passes (first k pass) where k in 0..n.

    results = []

    # Configure Pandas display
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.float_format", "{:.4f}".format)

    print(f"Running Bayesian Simulation")
    print(f"Alpha: {alpha}, Gamma: {gamma}")
    print(f"Public Tests: 4 (Prior Probability = 1.0)")
    print("-" * 60)

    for num_tests in range(1, 5):
        # Build scenarios for this number of public tests
        scenarios = {
            f"{k}P {num_tests - k}F": ([1] * k) + ([0] * (num_tests - k))
            for k in range(0, num_tests + 1)
        }

        for code_prior in code_priors:
            for beta in betas:
                for lr in learning_rates:
                    config = SimConfig(
                        alpha=alpha, beta=beta, gamma=gamma, learning_rate=lr
                    )

                    # For each scenario
                    for scenario_name, obs_list in scenarios.items():
                        # Setup Inputs
                        # 1. Code Priors (1 element)
                        prior_code_probs = np.array([code_prior])

                        # 2. Test Priors (num_tests elements, fixed at 1.0)
                        prior_test_probs = BayesianSystem.initialize_beliefs(
                            num_tests, 1.0
                        )

                        # 3. Observation Matrix (1 code x num_tests)
                        observation_matrix = np.array([obs_list])

                        # 4. Mask Matrix (All True/1)
                        mask_matrix = np.ones((1, num_tests))

                        # Perform Update
                        posterior_code_probs = BayesianSystem.update_code_beliefs(
                            prior_code_probs=prior_code_probs,
                            prior_test_probs=prior_test_probs,
                            observation_matrix=observation_matrix,
                            code_update_mask_matrix=mask_matrix,
                            config=config,
                        )

                        posterior = posterior_code_probs[0]

                        results.append(
                            {
                                "NumTests": num_tests,
                                "Prior": code_prior,
                                "Beta": beta,
                                "LearningRate": lr,
                                "Scenario": scenario_name,
                                "Posterior": posterior,
                                "Change": posterior - code_prior,
                            }
                        )

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    output_dir = os.path.join(current_dir, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_path = os.path.join(output_dir, "bayesian_simulation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Display Pivot Table for better readability
    # Pivot: Index=Prior, Beta; Columns=Scenario; Values=Posterior
    pivot_df = df.pivot_table(
        index=["Prior", "Beta", "LearningRate"], columns="Scenario", values="Posterior"
    )

    print("\nSimulation Results (Posterior Probabilities):")
    print(pivot_df)


if __name__ == "__main__":
    run_simulation()
