"""
Script to test contains_starter_code function with real LCB dataset starter codes.
"""

from common.code_preprocessing.analysis import contains_starter_code
from common.coevolution.lcb_dataset import Difficulty, load_code_generation_dataset


def main() -> None:
    """Load dataset and analyze starter codes."""
    print("Loading LCB dataset...")

    # Load a small sample of the dataset
    problems = load_code_generation_dataset(
        difficulty=Difficulty.EASY, start_date="2024-01-01", end_date="2024-01-31"
    )

    print(f"\nLoaded {len(problems)} problems")
    print("\n" + "=" * 80)
    print("STARTER CODE EXAMPLES:")
    print("=" * 80)

    # Examine first 10 starter codes
    for i, problem in enumerate(problems[:10], 1):
        print(f"\n--- Problem {i}: {problem.question_title} ---")
        print(f"Platform: {problem.platform.value}")
        print(f"Difficulty: {problem.difficulty.value}")
        print(f"\nStarter Code ({len(problem.starter_code)} chars):")
        print("-" * 40)
        print(problem.starter_code)
        print("-" * 40)

        # Test with a simple complete solution
        complete_code = problem.starter_code + "\n    return None"
        result = contains_starter_code(complete_code, problem.starter_code)
        print(f"✓ Self-test passed: {result}")

    # Look for edge cases
    print("\n" + "=" * 80)
    print("EDGE CASE ANALYSIS:")
    print("=" * 80)

    # Find problems with different starter code patterns
    empty_starters = [
        p for p in problems if not p.starter_code or p.starter_code == "pass"
    ]
    very_short = [p for p in problems if len(p.starter_code) < 50]
    very_long = [p for p in problems if len(p.starter_code) > 500]
    multi_function = [p for p in problems if p.starter_code.count("def ") > 1]

    print(f"\nEmpty/pass starters: {len(empty_starters)}")
    print(f"Very short (<50 chars): {len(very_short)}")
    print(f"Very long (>500 chars): {len(very_long)}")
    print(f"Multiple functions: {len(multi_function)}")

    if multi_function:
        print("\nExample multi-function starter:")
        print("-" * 40)
        print(multi_function[0].starter_code)
        print("-" * 40)


if __name__ == "__main__":
    main()
