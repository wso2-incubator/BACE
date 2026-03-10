"""
Path-related utilities for the coevolution project.
"""

def sanitize_problem_id(problem_id: str) -> str:
    """
    Replaces characters that would cause issues in file paths (like slashes)
    with underscores.
    """
    return problem_id.replace("/", "_").replace("\\", "_")
