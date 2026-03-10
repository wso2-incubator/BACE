"""
Path-related utilities for the coevolution project.
"""

def sanitize_id(id_str: str) -> str:
    """
    Replaces characters that would cause issues in file paths (like slashes)
    with underscores.
    """
    return id_str.replace("/", "_").replace("\\", "_")
