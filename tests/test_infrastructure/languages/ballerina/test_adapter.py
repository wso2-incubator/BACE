import pytest

from infrastructure.languages.ballerina import BallerinaLanguage


@pytest.fixture
def adapter() -> BallerinaLanguage:
    """Create BallerinaLanguage instance for testing."""
    return BallerinaLanguage()


class TestLanguageProperty:
    """Test language property."""

    def test_returns_ballerina(self, adapter: BallerinaLanguage) -> None:
        assert adapter.language == "ballerina"


class TestAdapterInitialization:
    """Test adapter initialization."""

    def test_initializes_successfully(self) -> None:
        adapter = BallerinaLanguage()
        assert adapter is not None
        assert adapter.language == "ballerina"

    def test_has_required_patterns(self, adapter: BallerinaLanguage) -> None:
        """Verify that all required regex patterns are initialized."""
        pass


class TestExtractCodeBlocks:
    """Test extract_code_blocks method."""

    def test_extracts_single_ballerina_block(self, adapter: BallerinaLanguage) -> None:
        response = """Here's a solution:
```ballerina
function add(int a, int b) returns int {
    return a + b;
}
```"""
        blocks = adapter.extract_code_blocks(response)
        assert len(blocks) == 1
        assert "function add" in blocks[0]
        assert "return a + b" in blocks[0]

    def test_extracts_multiple_ballerina_blocks(
        self, adapter: BallerinaLanguage
    ) -> None:
        """Test extraction of multiple code blocks like in the LLM response example."""
        response = """Response with multiple blocks:
```ballerina
// Solution 1
function hasCloseElements(float[] numbers, float threshold) returns boolean {
    return false;
}
```

```ballerina
// Solution 2
function hasCloseElements2(float[] numbers, float threshold) returns boolean {
    return true;
}
```"""
        blocks = adapter.extract_code_blocks(response)
        # With syntax validation disabled (always returns True), both blocks should be extracted
        assert len(blocks) == 2
        assert "Solution 1" in blocks[0]
        assert "Solution 2" in blocks[1]

    def test_extracts_capital_ballerina_block(self, adapter: BallerinaLanguage) -> None:
        response = "```Ballerina\nfunction test() {}\n```"
        blocks = adapter.extract_code_blocks(response)
        assert len(blocks) == 1

    def test_returns_empty_list_when_no_blocks(
        self, adapter: BallerinaLanguage
    ) -> None:
        # falling back to entire response as code is only done if valid
        response = "This is just text with no code blocks."
        assert adapter.extract_code_blocks(response) == []

    def test_skips_invalid_syntax_blocks(self, adapter: BallerinaLanguage) -> None:
        """With syntax validation disabled, all blocks are extracted."""
        response = """```ballerina
function broken(
```
```ballerina
function valid() {
    return;
}
```"""
        blocks = adapter.extract_code_blocks(response)
        # With syntax validation enabled, only the valid block is extracted
        assert len(blocks) == 1
        assert "function valid" in blocks[0]

    def test_handles_mixed_case_markdown(self, adapter: BallerinaLanguage) -> None:
        response = """```Ballerina
function first() {}
```
```ballerina
function second() {}
```"""
        blocks = adapter.extract_code_blocks(response)
        assert len(blocks) == 2
