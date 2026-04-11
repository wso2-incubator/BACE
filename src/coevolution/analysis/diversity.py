import numpy as np
import pandas as pd


def get_equivalent_code(matrices: dict[str, list[pd.DataFrame]]) -> list[list[str]]:
    """
    Identify groups of functionally equivalent code snippets based on identical behavior vectors.
    This is only for the LAST generation.

    Args:
        matrices: A dictionary mapping test type (e.g., 'unittest') to a list of DataFrames,
                  each representing a snapshot of that test type for a specific generation.

    Returns:
        A list of lists, where each inner list contains the IDs of code individuals
        that behave identically across all provided test matrices.
    """
    # 1. Filter and Collect Matrices
    final_matrices = []
    for test_type, matrix_list in matrices.items():
        if test_type == "private":
            continue
        final_matrices.append(matrix_list[-1])  # Get the last generation matrix

    if not final_matrices:
        return []

    # 2. Concatenate Horizontal (Join behavior vectors)
    # join='outer' ensures we include all code IDs, even if they missed a specific test block.
    # We fill NaNs with -1 to treat "missing test" as a distinct behavior.
    combined_matrix = pd.concat(final_matrices, axis=1, join="outer").fillna(-1)

    if combined_matrix.empty:
        return []

    # 3. Identify Unique Vectors
    combined_array = combined_matrix.values
    # axis=0 finds unique rows (behaviors)
    # return_inverse gives us the indices to reconstruct the groups
    _, indices = np.unique(combined_array, axis=0, return_inverse=True)

    # 4. Group IDs by Behavior Pattern
    equiv_groups: dict[int, list[str]] = {}

    # Get the actual Code IDs from the index
    code_ids = combined_matrix.index.tolist()

    for code_pos, pattern_id in enumerate(indices):
        if pattern_id not in equiv_groups:
            equiv_groups[pattern_id] = []

        # FIX: Use the actual ID from the dataframe index, not the integer position
        actual_id = code_ids[code_pos]
        equiv_groups[pattern_id].append(actual_id)

    return list(equiv_groups.values())


__all__ = ["get_equivalent_code"]
