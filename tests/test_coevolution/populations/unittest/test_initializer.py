import textwrap

from infrastructure.languages.python.adapter import PythonLanguage


def test_unittest_initializer_extraction_single_block():
    """Verify that multiple tests ARE extracted from a single code block if valid."""
    lang = PythonLanguage()
    parser = lang.parser

    code = textwrap.dedent("""
        def test_one():
            assert 1 == 1
            
        def test_two():
            assert 2 == 2
            
        def not_a_test():
            pass
    """)

    # 1. Test split_tests
    tests = parser.split_tests(code)
    assert len(tests) == 2
    assert "def test_one():" in tests[0]
    assert "def test_two():" in tests[1]


def test_unittest_initializer_extraction_with_syntax_error():
    """Verify that a syntax error in a block causes the block to be rejected by the parser."""
    lang = PythonLanguage()
    parser = lang.parser

    # Block with 3 tests, but the 2nd one has a syntax error
    code = textwrap.dedent("""
        def test_one():
            assert 1 == 1
            
        def test_two(
            assert 2 == 2
            
        def test_three():
            assert 3 == 3
    """)

    # extract_code_blocks will call is_syntax_valid
    assert not parser.is_syntax_valid(code)

    # In initializer.py, this block would be rejected before split_tests is called
    blocks = parser.extract_code_blocks(f"```python\n{code}\n```")
    assert len(blocks) == 0
