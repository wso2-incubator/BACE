import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Ensure src is on path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../src"))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from coevolution.services.bayesian import BayesianSystem
except ImportError:
    sys.path.append(os.path.abspath("src"))
    from coevolution.services.bayesian import BayesianSystem


@dataclass
class SimConfig:
    alpha: float
    beta: float
    gamma: float
    learning_rate: float = 1.0


def run_iterative_updates():
    # Fixed parameters
    beta = 0.2
    num_tests = 4
    iterations = 5

    # Sweep parameters
    priors = [round(x * 0.1, 1) for x in range(1, 10)]  # 0.1 to 0.9
    learning_rates = [0.05] + [
        round(x * 0.1, 1) for x in range(1, 11)
    ]  # 0.05, 0.1 to 1.0

    all_records = []

    for prior in priors:
        for lr in learning_rates:
            config = SimConfig(alpha=0.0, beta=beta, gamma=0.0, learning_rate=lr)

            # Initialize
            prior_code_probs = np.array([prior])
            prior_test_probs = BayesianSystem.initialize_beliefs(num_tests, 1.0)
            observation_matrix = np.ones((1, num_tests), dtype=int)  # 4 pass, 0 fail
            mask_matrix = np.ones((1, num_tests), dtype=int)

            for it in range(1, iterations + 1):
                posterior_code_probs = BayesianSystem.update_code_beliefs(
                    prior_code_probs=prior_code_probs,
                    prior_test_probs=prior_test_probs,
                    observation_matrix=observation_matrix,
                    code_update_mask_matrix=mask_matrix,
                    config=config,
                )

                prior_val = float(prior_code_probs[0])
                post_val = float(posterior_code_probs[0])
                delta = post_val - prior_val

                all_records.append(
                    {
                        "Prior": prior,
                        "LearningRate": lr,
                        "Iteration": it,
                        "PriorVal": prior_val,
                        "Posterior": post_val,
                        "Delta": delta,
                        "Beta": beta,
                    }
                )

                # Set prior <- posterior for next iteration
                prior_code_probs = posterior_code_probs

    # Save and print
    df = pd.DataFrame(all_records)

    out_dir = os.path.join(current_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "iterative_update_sweep_results.csv")
    df.to_csv(out_path, index=False)

    pd.set_option("display.float_format", "{:.6f}".format)
    print("Iterative Bayesian Updates Sweep (prior <- posterior each step):\n")
    print(df.to_string(index=False))
    print(f"\nSaved results to: {out_path}")

    return df


if __name__ == "__main__":
    run_iterative_updates()
