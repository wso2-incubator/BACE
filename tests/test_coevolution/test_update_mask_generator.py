import numpy as np
import pytest

from common.coevolution.bayesian_system import UpdateMaskGenerator


class TestUpdateMaskGenerator:
    @pytest.fixture
    def generator(self) -> type[UpdateMaskGenerator]:
        return UpdateMaskGenerator

    # 1. LOGIC TESTS
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "upd_gen, oth_gen, curr_gen, expected_row",
        [
            # Case 1: Both are OLD (born < current) -> False
            ([1], [1], 2, [False]),
            # Case 2: Updating is NEW (born == current), Other is OLD -> True
            ([2], [1], 2, [True]),
            # Case 3: Updating is OLD, Other is NEW (born == current) -> True
            ([1], [2], 2, [True]),
            # Case 4: Both are NEW -> True
            ([2], [2], 2, [True]),
        ],
    )
    def test_core_logic_single_values(
        self,
        generator: type[UpdateMaskGenerator],
        upd_gen: list[int],
        oth_gen: list[int],
        curr_gen: int,
        expected_row: list[bool],
    ) -> None:
        """Test the fundamental boolean logic on 1x1 matrices."""
        result = generator._get_update_mask_from_populations(upd_gen, oth_gen, curr_gen)
        assert result.shape == (1, 1)
        assert result[0, 0] == expected_row[0]

    def test_complex_matrix_logic(self, generator: UpdateMaskGenerator) -> None:
        """
        Test a complex scenario:
        Current Gen: 10
        Updating (Rows): [Old(9), New(10)]
        Other (Cols):    [Old(9), New(10), Old(8)]
        """
        current_gen = 10
        updating = [9, 10]
        other = [9, 10, 8]

        # Expected Logic:
        # Row 0 (Old): False unless Col is New
        # Row 1 (New): Always True (because self is new)

        # Matrix construction:
        # (9,9)->F, (9,10)->T, (9,8)->F
        # (10,9)->T, (10,10)->T, (10,8)->T
        expected = np.array([[False, True, False], [True, True, True]])

        result = generator._get_update_mask_from_populations(
            updating, other, current_gen
        )

        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == bool

    # 2. SHAPE AND STRUCTURE TESTS
    # -------------------------------------------------------------------------
    def test_empty_lists(self, generator: type[UpdateMaskGenerator]) -> None:
        """Should handle empty input lists gracefully."""
        result = generator._get_update_mask_from_populations([], [], 5)
        assert result.shape == (0, 0)

        # One dimension empty
        result = generator._get_update_mask_from_populations([1, 2], [], 5)
        assert result.shape == (2, 0)

    def test_asymmetric_shapes(self, generator: type[UpdateMaskGenerator]) -> None:
        """Ensure MxN shape is preserved correctly."""
        upd = [1] * 5  # 5 rows
        oth = [1] * 3  # 3 cols
        result = generator._get_update_mask_from_populations(upd, oth, 5)
        assert result.shape == (5, 3)

    # 3. VALIDATION AND ERROR HANDLING
    # -------------------------------------------------------------------------
    def test_raises_on_negative_generations_in_list(
        self, generator: type[UpdateMaskGenerator]
    ) -> None:
        with pytest.raises(ValueError, match="must be non-negative integers"):
            generator._get_update_mask_from_populations([-1], [1], 1)

        with pytest.raises(ValueError, match="must be non-negative integers"):
            generator._get_update_mask_from_populations([1], [1, -5], 1)

    def test_raises_on_negative_current_generation(
        self, generator: type[UpdateMaskGenerator]
    ) -> None:
        with pytest.raises(
            ValueError, match="Current generation must be a non-negative"
        ):
            generator._get_update_mask_from_populations([1], [1], -1)

    def test_raises_on_future_born_dates(
        self, generator: type[UpdateMaskGenerator]
    ) -> None:
        """Test that individuals cannot be born AFTER the current generation."""
        current = 5
        future_born = 6

        # Error in updating list
        with pytest.raises(ValueError, match="must be greater than or equal to"):
            generator._get_update_mask_from_populations([future_born], [1], current)

        # Error in other list
        with pytest.raises(ValueError, match="must be greater than or equal to"):
            generator._get_update_mask_from_populations([1], [future_born], current)

    # 4. PUBLIC WRAPPER METHODS
    # -------------------------------------------------------------------------
    def test_get_code_update_mask_generation(
        self, generator: type[UpdateMaskGenerator]
    ) -> None:
        """Should behave exactly like the internal method."""
        upd = [1, 2]
        oth = [1, 2]
        curr = 2

        internal = generator._get_update_mask_from_populations(upd, oth, curr)
        public = generator.get_code_update_mask_generation(upd, oth, curr)

        np.testing.assert_array_equal(public, internal)

    def test_get_test_update_mask_generation_transposes(
        self, generator: type[UpdateMaskGenerator]
    ) -> None:
        """
        Should behave like the internal method but TRANSPOSED.
        Scenario:
        Upd (Test Inds): 2 items
        Oth (Code Inds): 3 items
        Internal shape: (2, 3)
        Expected Output shape: (3, 2)
        """
        # Setup specifically so the matrix is NOT symmetric,
        # making the transpose obvious.
        upd = [0, 1]  # Length 2
        oth = [0, 1, 1]  # Length 3
        curr = 1

        # Direct internal call
        internal = generator._get_update_mask_from_populations(upd, oth, curr)
        assert internal.shape == (2, 3)

        # Public wrapper call
        result = generator.get_test_update_mask_generation(upd, oth, curr)

        # Verify Transpose
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result, internal.T)

    # 5. EDGE CASES & ADDITIONAL SCENARIOS
    # -------------------------------------------------------------------------
    def test_generation_zero_boundary(
        self, generator: type[UpdateMaskGenerator]
    ) -> None:
        """Test logic when everything happens at generation 0."""
        # If born at 0 and current is 0, they are 'NEW', so mask should be True
        upd = [0, 0]
        oth = [0, 0]
        curr = 0

        result = generator._get_update_mask_from_populations(upd, oth, curr)
        assert np.all(result)  # All should be True

    def test_large_generation_gap(self, generator: type[UpdateMaskGenerator]) -> None:
        """Test where current generation is significantly larger than born dates."""
        # All born at 0, current is 100. All should be considered 'OLD'.
        # Logic: Old updates Old -> False
        upd = [0, 0]
        oth = [0, 0]
        curr = 100

        result = generator._get_update_mask_from_populations(upd, oth, curr)
        assert not np.any(result)  # All should be False

    def test_all_true_scenario(self, generator: type[UpdateMaskGenerator]) -> None:
        """Scenario where everyone is 'new', resulting in a full True mask."""
        upd = [5, 5, 5]
        oth = [5, 5]
        curr = 5
        result = generator._get_update_mask_from_populations(upd, oth, curr)
        assert np.all(result)

    def test_single_row_single_col_shapes(
        self, generator: type[UpdateMaskGenerator]
    ) -> None:
        """Test 1xN and Nx1 shapes specifically."""
        # 1x3
        res_row = generator._get_update_mask_from_populations([2], [1, 2, 1], 2)
        assert res_row.shape == (1, 3)

        # 3x1
        res_col = generator._get_update_mask_from_populations([1, 2, 1], [2], 2)
        assert res_col.shape == (3, 1)
