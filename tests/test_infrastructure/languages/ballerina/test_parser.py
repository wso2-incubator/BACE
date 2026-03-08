import pytest

from infrastructure.languages.ballerina import BallerinaLanguage


@pytest.fixture
def adapter() -> BallerinaLanguage:
    """Create BallerinaLanguage instance for testing."""
    return BallerinaLanguage()


class TestIsSyntaxValid:
    """Test is_syntax_valid method."""

    def test_valid_simple_function(self, adapter: BallerinaLanguage) -> None:
        code = """function add(int a, int b) returns int {
    return a + b;
}"""
        # Syntax validation is disabled, always returns True
        assert adapter.is_syntax_valid(code)

    def test_invalid_empty_string(self, adapter: BallerinaLanguage) -> None:
        # basic_syntax_check requires a 'function' keyword
        assert not adapter.is_syntax_valid("")
        assert not adapter.is_syntax_valid("   ")

    def test_invalid_unbalanced_braces(self, adapter: BallerinaLanguage) -> None:
        code = "function test() { return;"
        # Basic check should catch unbalanced braces
        result = adapter.is_syntax_valid(code)
        assert not result

    def test_invalid_unbalanced_parens(self, adapter: BallerinaLanguage) -> None:
        code = "function test( { return; }"
        result = adapter.is_syntax_valid(code)
        assert not result

    def test_basic_syntax_check_requires_function(
        self, adapter: BallerinaLanguage
    ) -> None:
        code = "int x = 5;"
        result = adapter.is_syntax_valid(code)
        assert not result


class TestExtractTestNames:
    """Test extract_test_names method."""

    def test_extracts_single_test_name(self, adapter: BallerinaLanguage) -> None:
        test_code = """@test:Config
function testAdd() {
    test:assertEquals(add(1, 2), 3);
}"""
        names = adapter.extract_test_names(test_code)
        assert names == ["testAdd"]

    def test_extracts_multiple_test_names(self, adapter: BallerinaLanguage) -> None:
        test_code = """@test:Config
function testAdd() {
    test:assertEquals(add(1, 2), 3);
}

@test:Config
function testSubtract() {
    test:assertEquals(subtract(5, 3), 2);
}"""
        names = adapter.extract_test_names(test_code)
        assert names == ["testAdd", "testSubtract"]

    def test_returns_empty_list_when_no_tests(self, adapter: BallerinaLanguage) -> None:
        code = "function regular() { return 5; }"
        names = adapter.extract_test_names(code)
        assert names == []


class TestSplitTests:
    """Test split_tests method."""

    def test_splits_single_test(self, adapter: BallerinaLanguage) -> None:
        test_code = """@test:Config
function testOne() {
    test:assertEquals(1, 1);
}"""
        tests = adapter.split_tests(test_code)
        assert len(tests) == 1
        assert "@test:Config" in tests[0]
        assert "testOne" in tests[0]

    def test_splits_multiple_tests(self, adapter: BallerinaLanguage) -> None:
        test_code = """@test:Config
function testOne() {
    test:assertEquals(1, 1);
}

@test:Config
function testTwo() {
    test:assertEquals(2, 2);
}"""
        tests = adapter.split_tests(test_code)
        assert len(tests) == 2
        assert "testOne" in tests[0]
        assert "testTwo" in tests[1]

    def test_returns_empty_list_when_no_tests(self, adapter: BallerinaLanguage) -> None:
        code = "function regular() { return 5; }"
        tests = adapter.split_tests(code)
        assert tests == []

    def test_handles_nested_braces(self, adapter: BallerinaLanguage) -> None:
        test_code = """@test:Config
function testComplex() {
    if (true) {
        test:assertEquals(1, 1);
    }
}"""
        tests = adapter.split_tests(test_code)
        assert len(tests) == 1
        assert "if (true)" in tests[0]


class TestNormalizeCode:
    """Test normalize_code method."""

    def test_removes_single_line_comments(self, adapter: BallerinaLanguage) -> None:
        code = """function test() {
    // This is a comment
    return 1;
}"""
        normalized = adapter.normalize_code(code)
        assert "// This is a comment" not in normalized
        assert "return 1;" in normalized

    def test_removes_multiline_comments(self, adapter: BallerinaLanguage) -> None:
        code = """function test() {
    /* This is a
       multiline comment */
    return 1;
}"""
        normalized = adapter.normalize_code(code)
        assert "/* This is a" not in normalized
        assert "multiline comment */" not in normalized
        assert "return 1;" in normalized

    def test_normalizes_whitespace(self, adapter: BallerinaLanguage) -> None:
        code = """function   test()   {
    return    1;
        }"""
        normalized = adapter.normalize_code(code)
        lines = normalized.split("\n")
        # Each line should be stripped
        assert all(line == line.strip() for line in lines)


class TestContainsStarterCode:
    """Test contains_starter_code method."""

    def test_exact_match_after_normalization(self, adapter: BallerinaLanguage) -> None:
        starter = "function add(int a, int b) returns int"
        code = """function add(int a, int b) returns int {
    return a + b;
}"""
        assert adapter.contains_starter_code(code, starter)

    def test_matches_function_signature(self, adapter: BallerinaLanguage) -> None:
        starter = "function multiply(int x, int y) returns int"
        code = """// Different formatting
function multiply(int x, int y) returns int {
    return x * y;
}"""
        assert adapter.contains_starter_code(code, starter)

    def test_returns_false_when_function_not_present(
        self, adapter: BallerinaLanguage
    ) -> None:
        starter = "function divide(int a, int b) returns int"
        code = "function add(int a, int b) returns int { return a + b; }"
        assert not adapter.contains_starter_code(code, starter)


class TestGetStructuralMetadata:
    """Test get_structural_metadata method."""

    def test_extracts_function_metadata(self, adapter: BallerinaLanguage) -> None:
        code = """public function add(int a, int b) returns int {
    return a + b;
}

function subtract(int a, int b) returns int {
    return a - b;
}"""
        metadata = adapter.get_structural_metadata(code)
        assert len(metadata["functions"]) == 2
        assert metadata["functions"][0]["name"] == "add"
        assert metadata["functions"][0]["visibility"] == "public"
        assert metadata["functions"][0]["returns"] == "int"
        assert metadata["functions"][1]["name"] == "subtract"
        assert metadata["functions"][1]["visibility"] == ""

    def test_detects_main_function(self, adapter: BallerinaLanguage) -> None:
        code = """public function main() {
    io:println("Hello");
}"""
        metadata = adapter.get_structural_metadata(code)
        assert metadata["has_main"] is True

    def test_extracts_imports(self, adapter: BallerinaLanguage) -> None:
        code = """import ballerina/io;
import ballerina/test;

function test() {}"""
        metadata = adapter.get_structural_metadata(code)
        assert "ballerina/io" in metadata["imports"]
        assert "ballerina/test" in metadata["imports"]

    def test_returns_empty_metadata_for_empty_code(
        self, adapter: BallerinaLanguage
    ) -> None:
        metadata = adapter.get_structural_metadata("")
        assert metadata["functions"] == []
        assert metadata["imports"] == []
        assert metadata["has_main"] is False


class TestParseTestInputs:
    """Test parse_test_inputs method."""

    def test_parses_structured_format(self, adapter: BallerinaLanguage) -> None:
        outputs = """input: add(1, 2)
output: 3
input: add(5, 7)
output: 12"""
        test_cases = adapter.parse_test_inputs(outputs)
        assert len(test_cases) == 2
        assert test_cases[0]["input"] == "add(1, 2)"
        assert test_cases[0]["output"] == "3"
        assert test_cases[1]["input"] == "add(5, 7)"
        assert test_cases[1]["output"] == "12"

    def test_handles_missing_output(self, adapter: BallerinaLanguage) -> None:
        outputs = """input: test(1)
input: test(2)
output: 4"""
        test_cases = adapter.parse_test_inputs(outputs)
        assert len(test_cases) == 2
        assert test_cases[0]["output"] == ""
        assert test_cases[1]["output"] == "4"

    def test_tries_python_literal_eval_fallback(
        self, adapter: BallerinaLanguage
    ) -> None:
        outputs = '[{"input": "test(1)", "output": "1"}]'
        test_cases = adapter.parse_test_inputs(outputs)
        assert len(test_cases) == 1
        assert test_cases[0]["input"] == "test(1)"

    def test_returns_empty_list_on_invalid_format(
        self, adapter: BallerinaLanguage
    ) -> None:
        outputs = "This is not a valid format"
        test_cases = adapter.parse_test_inputs(outputs)
        assert test_cases == []
