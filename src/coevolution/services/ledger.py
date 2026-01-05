import numpy as np

from ..core.interfaces import IInteractionLedger, InteractionKey


class InteractionLedger(IInteractionLedger):
    """
    Concrete implementation of the interaction ledger.
    """

    def __init__(self) -> None:
        self._history: set[InteractionKey] = set()

    def get_new_interaction_mask(
        self, code_ids: list[str], test_ids: list[str], test_type: str, target: str
    ) -> np.ndarray:
        n_code = len(code_ids)
        n_test = len(test_ids)
        mask = np.zeros((n_code, n_test), dtype=int)

        for r, c_id in enumerate(code_ids):
            for c, t_id in enumerate(test_ids):
                key = (c_id, t_id, test_type, target)
                if key not in self._history:
                    mask[r, c] = 1
        return mask

    def commit_interactions(
        self,
        code_ids: list[str],
        test_ids: list[str],
        test_type: str,
        target: str,
        mask: np.ndarray,
    ) -> None:
        rows, cols = np.where(mask == 1)
        for r, c in zip(rows, cols):
            self._history.add((code_ids[r], test_ids[c], test_type, target))
