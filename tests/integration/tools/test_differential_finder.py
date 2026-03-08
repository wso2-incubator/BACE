# Integration test for DifferentialFinder using the real SafeCodeSandbox.
from coevolution.populations.differential.finder import DifferentialFinder
from infrastructure.sandbox import SandboxConfig


def test_integration_differential_finder_real_execution() -> None:
    """
    Integration test using the real SafeCodeSandbox.

    Scenario:
    - Code A (Correct): Adds two numbers normally.
    - Code B (Buggy): Adds two numbers but caps the result at 10.
    - Generator: Produces inputs that cover both behaviors.
    """

    # 1. Setup Snippets
    # We define simple functions. The compose_lcb_output_script will wrap these.
    code_a_snippet = """
def solve(x, y):
    return x + y
"""

    code_b_snippet = """
def solve(x, y):
    res = x + y
    if res > 10:
        return 10
    return res
"""

    # 2. Setup Generator Script
    # This script must print a valid Python list of dictionaries to stdout.
    # We generate 4 specific cases:
    # Case 1: 2 + 3 = 5   (Diff: No)
    # Case 2: 5 + 5 = 10  (Diff: No)
    # Case 3: 6 + 5 = 11  (Diff: Yes -> A=11, B=10)
    # Case 4: 10 + 10 = 20 (Diff: Yes -> A=20, B=10)
    input_generator_script = """
def generate_test_inputs(num_inputs):
    inputs = [
        {"x": 2, "y": 3},
        {"x": 5, "y": 5},
        {"x": 6, "y": 5},
        {"x": 10, "y": 10}
    ]
    print(inputs)

if __name__ == "__main__":
    generate_test_inputs(4)
"""

    # 3. Initialize SandboxConfig & Finder
    # Use specialized config for differential testing:
    # - Aggressive timeout (5s) for fast input execution
    # - Increased memory (200MB) for complex data structures
    # - Larger output buffer (50KB) for output comparisons
    sandbox_config = SandboxConfig(timeout=5, max_memory_mb=200, max_output_size=50_000)
    finder = DifferentialFinder(
        sandbox_config=sandbox_config, enable_multiprocessing=True, cpu_workers=4
    )

    # 4. Execute
    results = finder.find_differential(
        code_a_snippet=code_a_snippet,
        code_b_snippet=code_b_snippet,
        input_generator_script=input_generator_script,
        limit=10,
    )

    # 5. Assertions
    # We expect exactly 2 differentials (Case 3 and Case 4)
    assert len(results) == 2, f"Expected 2 divergences, found {len(results)}"

    # Check First Divergence (Case 3: 6+5)
    div_1 = results[0]
    assert div_1.input_data == {"x": 6, "y": 5}
    assert div_1.output_a == "11"
    assert div_1.output_b == "10"

    # Check Second Divergence (Case 4: 10+10)
    div_2 = results[1]
    assert div_2.input_data == {"x": 10, "y": 10}
    assert div_2.output_a == "20"
    assert div_2.output_b == "10"


def test_integration_complex_list_processing() -> None:
    """
    Scenario 2: List Processing & Conditional Logic

    - Code A (Correct): Standard ascending sort.
    - Code B (Buggy): Sorts ascending for small lists (len <= 3),
      but switches to DESCENDING sort for larger lists (simulating a complex logic bug).
    - Generator: Produces lists of varying lengths (2, 3, 4, 5).
    """

    # 1. Setup Snippets
    code_a_snippet = """
def solve(nums):
    return sorted(nums)
"""

    code_b_snippet = """
def solve(nums):
    # Logic bug: "optimization" that accidentally reverses large lists
    if len(nums) > 3:
        return sorted(nums, reverse=True)
    return sorted(nums)
"""

    # 2. Setup Generator
    # Cases:
    # 1. [3, 1] -> Both [1, 3] (Match)
    # 2. [5, 2, 8] -> Both [2, 5, 8] (Match)
    # 3. [1, 2, 3, 4] -> A=[1,2,3,4], B=[4,3,2,1] (Diff)
    # 4. [10, 5, 1, 9, 2] -> A=[1,2,5,9,10], B=[10,9,5,2,1] (Diff)
    input_generator_script = """
def gen(n):
    inputs = [
        {"nums": [3, 1]},
        {"nums": [5, 2, 8]},
        {"nums": [1, 2, 3, 4]},
        {"nums": [10, 5, 1, 9, 2]}
    ]
    print(inputs)

if __name__ == "__main__":
    gen(0)
"""

    # 3. Execute
    sandbox_config = SandboxConfig(timeout=5, max_memory_mb=200, max_output_size=50_000)
    finder = DifferentialFinder(
        sandbox_config=sandbox_config, enable_multiprocessing=True, cpu_workers=4
    )

    results = finder.find_differential(
        code_a_snippet, code_b_snippet, input_generator_script
    )

    # 4. Assertions
    assert len(results) == 2, f"Expected 2 list divergences, found {len(results)}"

    # Validate Divergence 1 (List length 4)
    res1 = results[0]
    assert res1.input_data == {"nums": [1, 2, 3, 4]}
    # Note: Sandbox outputs are strings
    assert res1.output_a == "[1, 2, 3, 4]"
    assert res1.output_b == "[4, 3, 2, 1]"


def test_integration_string_logic_and_crashes() -> None:
    """
    Scenario 3: String Logic & Runtime Errors

    - Code A (Correct): Robust Palindrome check (Case-insensitive, handles empty strings).
    - Code B (Buggy):
        1. Logical Bug: Case-sensitive (fails 'Racecar').
        2. Runtime Bug: Crashes on empty string input.
    - Generator: Produces mixed case, palindromes, and empty strings.
    """

    # 1. Setup Snippets
    code_a_snippet = """
def solve(s):
    # Correct robust implementation
    cleaned = s.lower()
    return cleaned == cleaned[::-1]
"""

    code_b_snippet = """
def solve(s):
    # Bug 1: Explicitly crashes on empty string (Runtime Error)
    if len(s) == 0:
        raise ValueError("Empty string input not handled")
    
    # Bug 2: Case sensitive comparison (Logical Error)
    return s == s[::-1]
"""

    # 2. Setup Generator
    # Cases:
    # 1. "aba" -> True, True (Match)
    # 2. "hello" -> False, False (Match)
    # 3. "Racecar" -> True (A), False (B) (Diff - Logic)
    # 4. "" -> True (A), Error (B) (Diff - Crash)
    input_generator_script = """
def gen(n):
    inputs = [
        {"s": "aba"},
        {"s": "hello"},
        {"s": "Racecar"},
        {"s": ""}
    ]
    print(inputs)

if __name__ == "__main__":
    gen(0)
"""

    # 3. Execute
    sandbox_config = SandboxConfig(timeout=5, max_memory_mb=200, max_output_size=50_000)
    finder = DifferentialFinder(
        sandbox_config=sandbox_config, enable_multiprocessing=True, cpu_workers=4
    )

    results = finder.find_differential(
        code_a_snippet, code_b_snippet, input_generator_script
    )

    # 4. Assertions
    # The DifferentialFinder skips test inputs that cause execution errors
    # (returns None from _generate_output), so we only get the logical divergence
    assert len(results) == 1, (
        f"Expected 1 divergence (crash skipped), found {len(results)}"
    )

    # Check Logical Divergence ("Racecar")
    res_logic = results[0]
    assert res_logic.input_data == {"s": "Racecar"}
    assert res_logic.output_a == "True"
    assert res_logic.output_b == "False"

    # Note: The empty string case ("") causes code_b to crash, so it's skipped
    # by the finder and not reported as a divergence
