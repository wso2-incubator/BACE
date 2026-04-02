import pytest

from coevolution.core.interfaces.language import LanguageParsingError
from infrastructure.languages.python.ast import remove_main_block


def test_remove_main_block_preserves_comments() -> None:
    code_with_comments = """
# Top-level comment
def foo(x):
    \"\"\"Function docstring.\"\"\"
    # Inline comment
    return x + 1

if __name__ == "__main__":
    # Main block comment
    print(foo(5))
"""
    result = remove_main_block(code_with_comments)

    # Assertions
    assert "# Top-level comment" in result
    assert '"""Function docstring."""' in result
    assert "# Inline comment" in result
    assert "def foo(x):" in result

    # The main block itself should be gone
    assert 'if __name__ == "__main__":' not in result
    assert "# Main block comment" not in result
    assert "print(foo(5))" not in result


def test_remove_main_block_no_change_if_no_main() -> None:
    code_no_main = """
# Only functions here
def bar():
    # Comment in bar
    pass
"""
    result = remove_main_block(code_no_main)

    # It should be exactly the same (except maybe leading/trailing whitespace)
    assert result.strip() == code_no_main.strip()
    assert "# Only functions here" in result
    assert "# Comment in bar" in result


def test_remove_main_block_handles_reversed_comparison() -> None:
    code_reversed = """
def baz():
    return 42

if "__main__" == __name__:
    baz()
"""
    result = remove_main_block(code_reversed)
    assert 'if "__main__" == __name__:' not in result
    assert "def baz():" in result


def test_remove_main_block_syntax_error() -> None:
    invalid_code = "def foo(:"
    with pytest.raises(LanguageParsingError):
        remove_main_block(invalid_code)
    with pytest.raises(LanguageParsingError):
        remove_main_block(invalid_code)
