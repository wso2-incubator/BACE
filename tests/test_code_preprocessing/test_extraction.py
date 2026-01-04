"""Tests for code_preprocessing.extraction module."""

import pytest

from infrastructure.code_preprocessing.exceptions import CodeParsingError
from infrastructure.code_preprocessing.extraction import (
    extract_all_code_blocks_from_response,
    extract_code_block_from_response,
    extract_code_structure,
    extract_function_name_from_problem,
)


class TestExtractFunctionNameFromProblem:
    """Test extract_function_name_from_problem function."""

    def test_extracts_simple_function_name(self) -> None:
        problem = "def calculate(x, y):\n    pass"
        assert extract_function_name_from_problem(problem) == "calculate"

    def test_extracts_function_with_type_hints(self) -> None:
        problem = "def process_data(items: list[str]) -> int:\n    pass"
        assert extract_function_name_from_problem(problem) == "process_data"

    def test_returns_empty_string_when_no_function(self) -> None:
        problem = "This is just a description\nwith no function definition"
        assert extract_function_name_from_problem(problem) == ""

    def test_extracts_first_function_when_multiple(self) -> None:
        problem = "def foo():\n    pass\ndef bar():\n    pass"
        assert extract_function_name_from_problem(problem) == "foo"

    def test_handles_indented_function(self) -> None:
        problem = "    def indented_func():\n        pass"
        assert extract_function_name_from_problem(problem) == "indented_func"


class TestExtractCodeBlockFromResponse:
    """Test extract_code_block_from_response function."""

    def test_extracts_python_code_block(self) -> None:
        response = "Here's the solution:\n```python\ndef foo(): pass\n```"
        assert extract_code_block_from_response(response) == "def foo(): pass"

    def test_extracts_with_capital_python(self) -> None:
        response = "```Python\nprint('hello')\n```"
        assert extract_code_block_from_response(response) == "print('hello')"

    def test_raises_error_when_no_code_block(self) -> None:
        response = "This has no code blocks at all"
        with pytest.raises(CodeParsingError, match="No Python code block found"):
            extract_code_block_from_response(response)

    def test_extracts_first_block_when_multiple(self) -> None:
        response = "```python\nfirst\n```\nsome text\n```python\nsecond\n```"
        assert extract_code_block_from_response(response) == "first"

    def test_handles_multiline_code(self) -> None:
        response = """```python
def add(a, b):
    return a + b

result = add(1, 2)
```"""
        expected = "def add(a, b):\n    return a + b\n\nresult = add(1, 2)"
        assert extract_code_block_from_response(response) == expected

    def test_strips_whitespace(self) -> None:
        response = "```python\n\n  def foo(): pass  \n\n```"
        assert extract_code_block_from_response(response) == "def foo(): pass"


class TestExtractAllCodeBlocksFromResponse:
    """Test extract_all_code_blocks_from_response function."""

    def test_extracts_multiple_code_blocks(self) -> None:
        response = "```python\nfirst\n```\ntext\n```python\nsecond\n```"
        blocks = extract_all_code_blocks_from_response(response)
        assert len(blocks) == 2
        assert blocks[0] == "first"
        assert blocks[1] == "second"

    def test_extracts_single_block(self) -> None:
        response = '```python\nprint("hi")\n```'
        blocks = extract_all_code_blocks_from_response(response)
        assert len(blocks) == 1
        assert blocks[0] == 'print("hi")'

    def test_raises_error_when_no_blocks(self) -> None:
        response = "No code here"
        with pytest.raises(CodeParsingError, match="No Python code blocks found"):
            extract_all_code_blocks_from_response(response)

    def test_handles_mixed_case_python(self) -> None:
        response = "```Python\nfirst\n```\n```python\nsecond\n```"
        blocks = extract_all_code_blocks_from_response(response)
        assert len(blocks) == 2


class TestExtractCodeStructure:
    """Test extract_code_structure function."""

    def test_extracts_imports(self) -> None:
        code = "import os\nimport sys\n\ndef foo(): pass"
        structure = extract_code_structure(code)
        assert "import os" in structure["import_lines"]
        assert "import sys" in structure["import_lines"]

    def test_extracts_function_definitions(self) -> None:
        code = "def foo():\n    pass\n\ndef bar():\n    pass"
        structure = extract_code_structure(code)
        assert "foo" in structure["function_definitions"]
        assert "bar" in structure["function_definitions"]
        assert structure["function_definitions"]["foo"] == (0, 1)
        assert structure["function_definitions"]["bar"] == (3, 4)

    def test_extracts_class_definitions(self) -> None:
        code = "class Foo:\n    pass\n\nclass Bar:\n    pass"
        structure = extract_code_structure(code)
        assert "Foo" in structure["class_definitions"]
        assert "Bar" in structure["class_definitions"]

    def test_removes_duplicate_imports(self) -> None:
        code = "import os\nimport os\ndef foo(): pass"
        structure = extract_code_structure(code)
        assert structure["import_lines"].count("import os") == 1

    def test_handles_from_imports(self) -> None:
        code = "from typing import List\nimport os"
        structure = extract_code_structure(code)
        assert "from typing import List" in structure["import_lines"]
        assert "import os" in structure["import_lines"]

    def test_raises_error_on_syntax_error(self) -> None:
        code = "def foo(\n    invalid syntax"
        with pytest.raises(CodeParsingError, match="Failed to parse code"):
            extract_code_structure(code)

    def test_handles_empty_code(self) -> None:
        code = ""
        structure = extract_code_structure(code)
        assert structure["import_lines"] == []
        assert structure["function_definitions"] == {}
        assert structure["class_definitions"] == {}

    def test_structure_type_matches_codestructure(self) -> None:
        code = "import os\ndef foo(): pass\nclass Bar: pass"
        structure = extract_code_structure(code)
        # TypedDict check - should have all required keys
        assert "import_lines" in structure
        assert "function_definitions" in structure
        assert "class_definitions" in structure
        assert isinstance(structure["import_lines"], list)
        assert isinstance(structure["function_definitions"], dict)
        assert isinstance(structure["class_definitions"], dict)
