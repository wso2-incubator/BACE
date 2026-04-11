import pytest

from coevolution.core.interfaces.language import LanguageTransformationError
from infrastructure.languages.ballerina import BallerinaLanguage


@pytest.fixture
def adapter() -> BallerinaLanguage:
    """Create BallerinaLanguage instance for testing."""
    return BallerinaLanguage()


class TestComposeTestScript:
    """Test compose_test_script method."""

    def test_adds_test_import_when_missing(self, adapter: BallerinaLanguage) -> None:
        code = "function add(int a, int b) returns int { return a + b; }"
        test = """@test:Config
function testAdd() {
    test:assertEquals(add(1, 2), 3);
}"""
        script = adapter.composer.compose_test_script(code, test)
        assert "import ballerina/test;" in script
        assert "function add" in script
        assert "testAdd" in script

    def test_does_not_duplicate_test_import(self, adapter: BallerinaLanguage) -> None:
        code = "function add(int a, int b) returns int { return a + b; }"
        test = """import ballerina/test;

@test:Config
function testAdd() {
    test:assertEquals(add(1, 2), 3);
}"""
        script = adapter.composer.compose_test_script(code, test)
        # Should only have one test import
        assert script.count("import ballerina/test") == 1

    def test_removes_test_import_from_code(self, adapter: BallerinaLanguage) -> None:
        code = """import ballerina/test;

function add(int a, int b) returns int { return a + b; }"""
        test = """@test:Config
function testAdd() {
    test:assertEquals(add(1, 2), 3);
}"""
        script = adapter.composer.compose_test_script(code, test)
        # Test import should appear only once, at the top
        lines = script.split("\n")
        import_count = sum(1 for line in lines if "import ballerina/test" in line)
        assert import_count == 1
        assert lines[0] == "import ballerina/test;"


class TestComposeEvaluationScript:
    """Test compose_evaluation_script method."""

    def test_creates_executable_script(self, adapter: BallerinaLanguage) -> None:
        code = "function add(int a, int b) returns int { return a + b; }"
        # New format is JSON with 'inputdata' key or just a JSON dict
        import json
        input_data = json.dumps({"a": 5, "b": 3})
        script = adapter.composer.compose_evaluation_script(code, input_data)
        assert "import ballerina/io;" in script
        assert "public function main()" in script
        assert "var result = add(" in script
        assert "5" in script
        assert "3" in script

    def test_raises_error_on_invalid_input_format(
        self, adapter: BallerinaLanguage
    ) -> None:
        code = "function add(int a, int b) returns int { return a + b; }"
        input_data = "not a function call"
        with pytest.raises(LanguageTransformationError, match="Failed to compose evaluation script"):
            adapter.composer.compose_evaluation_script(code, input_data)

    def test_handles_complex_arguments(self, adapter: BallerinaLanguage) -> None:
        code = "function concat(string a, string b) returns string { return a + b; }"
        import json
        input_data = json.dumps({"a": "hello", "b": " world"})
        script = adapter.composer.compose_evaluation_script(code, input_data)
        assert 'concat(' in script
        assert '"hello"' in script
        assert '" world"' in script


class TestGenerateTestCase:
    """Test generate_test_case method."""

    def test_generates_basic_test_case(self, adapter: BallerinaLanguage) -> None:
        starter = "function add(int a, int b) returns int {"
        input_str = "add(1, 2)"
        output_str = "3"
        test = adapter.composer.generate_test_case(input_str, output_str, starter, 1)
        assert "@test:Config" in test
        assert "testAdd1" in test
        assert "var result = add(1, 2);" in test
        assert "test:assertEquals(result, 3" in test

    def test_raises_error_when_no_function_in_starter(
        self, adapter: BallerinaLanguage
    ) -> None:
        starter = "int x = 5;"
        with pytest.raises(
            LanguageTransformationError, match="Failed to generate test case"
        ):
            adapter.composer.generate_test_case("test()", "5", starter, 1)

    def test_handles_different_test_numbers(self, adapter: BallerinaLanguage) -> None:
        starter = "function multiply(int a, int b) returns int {"
        test1 = adapter.composer.generate_test_case("multiply(2, 3)", "6", starter, 1)
        test2 = adapter.composer.generate_test_case("multiply(4, 5)", "20", starter, 2)
        assert "testMultiply1" in test1
        assert "testMultiply2" in test2


class TestGenerateTestCaseWithRealDataset:
    """Test generate_test_case with real data from HumanEval Ballerina dataset."""

    def test_generates_public_test_cases_from_humaneval(
        self, adapter: BallerinaLanguage
    ) -> None:
        """Test generating test cases from HumanEval public tests."""
        # Data from humaneval-ballerina.py
        starter = "function hasCloseElements(float[] numbers, float threshold) returns boolean {"

        # Public test case 1
        test1 = adapter.composer.generate_test_case("[1.0, 2.0, 3.0], 0.5", "false", starter, 1)
        assert "@test:Config" in test1
        assert "testHasCloseElements1" in test1
        assert "hasCloseElements([1.0, 2.0, 3.0], 0.5)" in test1
        assert "test:assertEquals(result, false" in test1

        # Public test case 2
        test2 = adapter.composer.generate_test_case(
            "[1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3", "true", starter, 2
        )
        assert "@test:Config" in test2
        assert "testHasCloseElements2" in test2
        assert "hasCloseElements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)" in test2
        assert "test:assertEquals(result, true" in test2

    def test_generates_private_test_cases_from_humaneval(
        self, adapter: BallerinaLanguage
    ) -> None:
        """Test generating test cases from HumanEval private tests."""
        starter = "function hasCloseElements(float[] numbers, float threshold) returns boolean {"

        # Private test case 1
        test3 = adapter.composer.generate_test_case(
            "[1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3", "true", starter, 3
        )
        assert "@test:Config" in test3
        assert "testHasCloseElements3" in test3
        assert "hasCloseElements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3)" in test3
        assert "test:assertEquals(result, true" in test3

        # Private test case 2
        test4 = adapter.composer.generate_test_case(
            "[1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05", "false", starter, 4
        )
        assert "@test:Config" in test4
        assert "testHasCloseElements4" in test4
        assert "hasCloseElements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05)" in test4
        assert "test:assertEquals(result, false" in test4

    def test_generates_all_private_tests_from_humaneval(
        self, adapter: BallerinaLanguage
    ) -> None:
        """Test generating all 7 private test cases from HumanEval dataset."""
        starter = "function hasCloseElements(float[] numbers, float threshold) returns boolean {"

        # All private test cases from humaneval-ballerina.py
        private_tests = [
            ("[1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3", "true"),
            ("[1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05", "false"),
            ("[1.0, 2.0, 5.9, 4.0, 5.0], 0.95", "true"),
            ("[1.0, 2.0, 5.9, 4.0, 5.0], 0.8", "false"),
            ("[1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1", "true"),
            ("[1.1, 2.2, 3.1, 4.1, 5.1], 1.0", "true"),
            ("[1.1, 2.2, 3.1, 4.1, 5.1], 0.5", "false"),
        ]

        generated_tests = []
        for i, (input_str, output_str) in enumerate(private_tests, start=1):
            test = adapter.composer.generate_test_case(input_str, output_str, starter, i)
            generated_tests.append(test)

            # Verify each test has correct structure
            assert "@test:Config" in test
            assert f"testHasCloseElements{i}" in test
            assert f"hasCloseElements({input_str})" in test
            assert f"test:assertEquals(result, {output_str}" in test

        # Verify we generated all 7 tests
        assert len(generated_tests) == 7

        # Verify tests are unique
        assert len(set(generated_tests)) == 7

    def test_generated_tests_are_valid_ballerina_syntax(
        self, adapter: BallerinaLanguage
    ) -> None:
        """Verify generated test code has valid Ballerina structure."""
        starter = "function hasCloseElements(float[] numbers, float threshold) returns boolean {"

        test = adapter.composer.generate_test_case("[1.0, 2.0, 3.0], 0.5", "false", starter, 1)

        # Check basic Ballerina test syntax
        assert test.count("{") == test.count("}")
        assert test.count("(") == test.count(")")
        assert "@test:Config" in test
        assert "function test" in test
        assert "var result =" in test
        assert "test:assertEquals" in test

        # Should be parseable by extract_test_names
        test_names = adapter.parser.extract_test_names(test)
        assert len(test_names) == 1
        assert "testHasCloseElements1" in test_names


class TestRemoveMainBlock:
    """Test remove_main_block method."""

    def test_removes_main_function(self, adapter: BallerinaLanguage) -> None:
        code = """function add(int a, int b) returns int {
    return a + b;
}

public function main() {
    io:println(add(1, 2));
}"""
        cleaned = adapter.parser.remove_main_block(code)
        assert "public function main()" not in cleaned
        assert "function add" in cleaned

    def test_preserves_code_without_main(self, adapter: BallerinaLanguage) -> None:
        code = "function add(int a, int b) returns int { return a + b; }"
        cleaned = adapter.parser.remove_main_block(code)
        assert "function add" in cleaned

    def test_cleans_up_extra_blank_lines(self, adapter: BallerinaLanguage) -> None:
        code = """function add() { return 1; }



public function main() { }"""
        cleaned = adapter.parser.remove_main_block(code)
        # Should not have 3+ consecutive newlines
        assert "\n\n\n" not in cleaned
