import pytest

from infrastructure.languages.python import PythonLanguage


@pytest.fixture
def adapter() -> PythonLanguage:
    """Create PythonLanguage instance for testing."""
    return PythonLanguage()


class TestLanguageProperty:
    def test_returns_python(self, adapter: PythonLanguage) -> None:
        assert adapter.language == "python"


class TestExtractCodeBlocks:
    def test_extracts_single_python_block(self, adapter: PythonLanguage) -> None:
        response = """Here's the solution:
```python
def add(a, b):
    return a + b
```"""
        blocks = adapter.extract_code_blocks(response)
        assert len(blocks) == 1
        assert "def add" in blocks[0]

    def test_extracts_capital_python_block(self, adapter: PythonLanguage) -> None:
        response = "```Python\ndef foo(): pass\n```"
        blocks = adapter.extract_code_blocks(response)
        assert len(blocks) == 1
        assert "def foo" in blocks[0]

    def test_extracts_multiple_blocks(self, adapter: PythonLanguage) -> None:
        response = """```python
def solution_a():
    return 1
```

```python
def solution_b():
    return 2
```"""
        blocks = adapter.extract_code_blocks(response)
        assert len(blocks) == 2
        assert "solution_a" in blocks[0]
        assert "solution_b" in blocks[1]

    def test_falls_back_to_raw_response_when_valid(
        self, adapter: PythonLanguage
    ) -> None:
        response = "def foo():\n    return 42"
        blocks = adapter.extract_code_blocks(response)
        assert len(blocks) == 1
        assert "def foo" in blocks[0]

    def test_returns_empty_list_when_no_valid_code(
        self, adapter: PythonLanguage
    ) -> None:
        response = "This is just plain text with no code."
        blocks = adapter.extract_code_blocks(response)
        assert blocks == []

    def test_skips_blocks_with_syntax_errors(self, adapter: PythonLanguage) -> None:
        response = """```python
def valid():
    return 1
```
```python
def broken(
```"""
        blocks = adapter.extract_code_blocks(response)
        assert len(blocks) == 1
        assert "def valid" in blocks[0]
