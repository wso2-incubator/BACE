import unittest

import numpy as np

from coevolution.services.ledger import InteractionLedger


class TestInteractionLedger(unittest.TestCase):
    def setUp(self) -> None:
        """Create a fresh ledger before each test."""
        self.ledger = InteractionLedger()

        # Common test data
        self.code_ids = ["c1", "c2"]
        self.test_ids = ["t1", "t2", "t3"]
        self.test_type = "unittest"
        self.target = "CODE"

    def test_initialization(self) -> None:
        """Ensure the ledger starts with an empty history."""
        self.assertEqual(len(self.ledger._history), 0)

    def test_mask_all_new(self) -> None:
        """
        Scenario: No interactions have occurred yet.
        Expected: Mask should be all 1s (shape: n_code x n_test).
        """
        mask = self.ledger.get_new_interaction_mask(
            self.code_ids, self.test_ids, self.test_type, self.target
        )

        expected_shape = (2, 3)
        self.assertEqual(mask.shape, expected_shape)
        # Assert all elements are 1
        np.testing.assert_array_equal(mask, np.ones(expected_shape, dtype=int))

    def test_commit_updates_history(self) -> None:
        """
        Scenario: Commit a full mask of interactions.
        Expected: History set should contain all committed tuples.
        """
        # Create a full mask (all 1s)
        mask = np.ones((len(self.code_ids), len(self.test_ids)), dtype=int)

        self.ledger.commit_interactions(
            self.code_ids, self.test_ids, self.test_type, self.target, mask
        )

        # We expect 2 * 3 = 6 entries in history
        self.assertEqual(len(self.ledger._history), 6)

        # Verify a specific key exists: (CodeID, TestID, Type, Target)
        expected_key = ("c1", "t1", self.test_type, self.target)
        self.assertIn(expected_key, self.ledger._history)

    def test_get_mask_after_commit(self) -> None:
        """
        Scenario: Commit interactions, then ask for the mask again with the same IDs.
        Expected: Mask should be all 0s (seen).
        """
        # 1. Commit everything
        mask_initial = np.ones((2, 3), dtype=int)
        self.ledger.commit_interactions(
            self.code_ids, self.test_ids, self.test_type, self.target, mask_initial
        )

        # 2. Get mask again
        mask_new = self.ledger.get_new_interaction_mask(
            self.code_ids, self.test_ids, self.test_type, self.target
        )

        # 3. Assert all 0s
        np.testing.assert_array_equal(mask_new, np.zeros((2, 3), dtype=int))

    def test_partial_commit(self) -> None:
        """
        Scenario: Only specific interactions are committed (e.g., Code1 passed Test1, others failed/skipped).
        Expected: Only the committed cells should be 0 in the future; others remain 1.
        """
        # Mask where only (c0, t0) is processed
        partial_mask = np.zeros((2, 3), dtype=int)
        partial_mask[0, 0] = 1  # c1, t1

        self.ledger.commit_interactions(
            self.code_ids, self.test_ids, self.test_type, self.target, partial_mask
        )

        # Get fresh mask
        new_mask = self.ledger.get_new_interaction_mask(
            self.code_ids, self.test_ids, self.test_type, self.target
        )

        # (0,0) should now be 0 (seen)
        self.assertEqual(new_mask[0, 0], 0)

        # (0,1) was NOT committed, so it should still be 1 (new)
        self.assertEqual(new_mask[0, 1], 1)

    def test_different_targets_are_independent(self) -> None:
        """
        Scenario: "CODE" update committed. Now check "TEST" update for same IDs.
        Expected: "TEST" update should still be seen as new (all 1s).
        """
        # Commit CODE update
        mask = np.ones((2, 3), dtype=int)
        self.ledger.commit_interactions(
            self.code_ids, self.test_ids, self.test_type, "CODE", mask
        )

        # Check TEST mask
        mask_test = self.ledger.get_new_interaction_mask(
            self.code_ids, self.test_ids, self.test_type, "TEST"
        )

        # Should be all 1s (independent history)
        np.testing.assert_array_equal(mask_test, np.ones((2, 3), dtype=int))

    def test_different_test_types_are_independent(self) -> None:
        """
        Scenario: "unittest" committed. Now check "public" for same IDs.
        Expected: "public" update should be seen as new.
        """
        mask = np.ones((2, 3), dtype=int)
        self.ledger.commit_interactions(
            self.code_ids, self.test_ids, "unittest", self.target, mask
        )

        # Check PUBLIC mask
        mask_public = self.ledger.get_new_interaction_mask(
            self.code_ids, self.test_ids, "public", self.target
        )

        np.testing.assert_array_equal(mask_public, np.ones((2, 3), dtype=int))

    def test_empty_lists(self) -> None:
        """
        Scenario: IDs lists are empty.
        Expected: Returns empty numpy array of shape (0, 0).
        """
        mask = self.ledger.get_new_interaction_mask([], [], "type", "CODE")
        self.assertEqual(mask.shape, (0, 0))
        self.assertEqual(mask.size, 0)

    def test_id_order_sensitivity(self) -> None:
        """
        Scenario: Query with shuffled IDs.
        Expected: The ledger correctly maps (ID -> History) regardless of list index position.
        """
        # 1. Commit (c1, t1)
        mask = np.zeros((1, 1), dtype=int)
        mask[0, 0] = 1
        self.ledger.commit_interactions(
            ["c1"], ["t1"], self.test_type, self.target, mask
        )

        # 2. Query with [c2, c1] and [t1]
        # Matrix shape will be (2, 1).
        # Row 0 (c2) -> New (1)
        # Row 1 (c1) -> Seen (0)
        new_mask = self.ledger.get_new_interaction_mask(
            ["c2", "c1"], ["t1"], self.test_type, self.target
        )

        self.assertEqual(new_mask[0, 0], 1)  # c2, t1 (New)
        self.assertEqual(new_mask[1, 0], 0)  # c1, t1 (Seen)


if __name__ == "__main__":
    unittest.main()
