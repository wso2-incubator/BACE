from unittest.mock import MagicMock, patch

import pytest

from coevolution.populations.property.evaluator import _property_eval_worker


def test_property_eval_worker_sorting_and_early_break():
    # Setup mock sandbox
    mock_sandbox = MagicMock()

    # Define IOPairs with different lengths
    import json
    pairs = [
        {"inputdata": json.dumps({"inputdata": "long_input_data_string"}), "output": "output1"},
        {"inputdata": json.dumps({"inputdata": "short"}), "output": "output2"},
        {"inputdata": json.dumps({"inputdata": "medium_input"}), "output": "output3"},
    ]

    # Sort them as the evaluator would before passing to worker
    sorted_pairs = sorted(pairs, key=lambda p: len(str(p["inputdata"])))
    assert "short" in sorted_pairs[0]["inputdata"]
    assert "medium_input" in sorted_pairs[1]["inputdata"]
    assert "long_input_data_string" in sorted_pairs[2]["inputdata"]

    # Mock execute_code to fail on the SECOND shortest input ("medium_input")
    def side_effect(script, runtime):
        # We check if the script contains the medium_input
        if "medium_input" in script:
            # Simulate a failure (result_line != "True")
            result = MagicMock()
            result.error = None
            result.output = "False"
            return result
        else:
            # Simulate success
            result = MagicMock()
            result.error = None
            result.output = "True"
            return result

    mock_sandbox.execute_code.side_effect = side_effect

    with patch(
        "coevolution.populations.property.evaluator.create_sandbox",
        return_value=mock_sandbox,
    ):
        # We need to mock compose_property_test_script to include inputdata so side_effect can see it
        with patch(
            "coevolution.populations.property.evaluator.compose_property_test_script",
            side_effect=lambda s, i, o: f"test {s} with {i} -> {o}",
        ):
            code_id, test_id, result = _property_eval_worker(
                ("C1", "T1", "prop_snippet", sorted_pairs, MagicMock())
            )

            # 1. Verify it failed
            assert result.status == "failed"

            # 2. Verify execute_code was called exactly TWICE (short success, medium failure, then break)
            # It should NOT call for "long_input_data_string"
            assert mock_sandbox.execute_code.call_count == 2

            # 3. Verify the order of calls
            calls = mock_sandbox.execute_code.call_args_list
            assert "short" in calls[0][0][0]
            assert "medium_input" in calls[1][0][0]


if __name__ == "__main__":
    pytest.main([__file__])

if __name__ == "__main__":
    pytest.main([__file__])
