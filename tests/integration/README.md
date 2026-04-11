# Integration Tests

This directory contains integration tests that verify the complete coevolution system using mock implementations.

## Running Tests

### Option 1: Run with pytest (Recommended for CI/CD)

Run all integration tests:

```bash
pytest tests/integration/ -v
```

Run a specific test:

```bash
pytest tests/integration/test_orchestrator_integration.py -v
```

Run with more detailed output:

```bash
pytest tests/integration/test_orchestrator_integration.py -v -s
```

### Option 2: Run as standalone script (For manual testing/debugging)

Run the integration test directly:

```bash
uv run python tests/integration/test_orchestrator_integration.py
```

This mode provides more detailed logging output and is useful for:

- Understanding the algorithm flow
- Debugging issues
- Verifying the architecture works end-to-end

## What's Tested

### `test_orchestrator_integration.py`

Tests the complete orchestrator run with mock components:

- ✅ Orchestrator runs without errors for configured generations
- ✅ Both code and test populations are created and evolved
- ✅ Final populations contain individuals
- ✅ Best individuals can be identified
- ✅ Pareto front is computed for test population
- ✅ Population size constraints are respected
- ✅ Probability calculations are valid (in range [0, 1])

## Benefits

### pytest Integration Benefits

- Automated testing in CI/CD pipelines
- Generates test reports
- Can be combined with coverage tools
- Integrates with IDE test runners
- Provides clear pass/fail feedback

### Standalone Script Benefits

- More verbose logging output
- Easier to debug
- Shows generation-by-generation progress
- Displays detailed results at the end
- Useful for understanding the algorithm

## Adding New Integration Tests

To add new integration tests:

1. Create a new file following the naming convention: `test_<feature>_integration.py`
2. Use pytest fixtures for common setup
3. Provide both pytest test functions and a `main()` for standalone execution
4. Add clear docstrings explaining what is tested
5. Include assertions to verify expected behavior

Example structure:

```python
"""Integration test for <feature>."""

import pytest

@pytest.fixture
def setup_fixture():
    """Provide test setup."""
    return setup_data()

def test_feature(setup_fixture):
    """Test the feature end-to-end."""
    result = run_feature(setup_fixture)
    assert result is not None
    # Add more assertions

def main():
    """Run as standalone script."""
    setup = setup_fixture()
    test_feature(setup)
    print("Test completed!")

if __name__ == "__main__":
    main()
```
