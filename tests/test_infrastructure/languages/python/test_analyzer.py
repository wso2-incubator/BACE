"""Tests for the PythonTestAnalyzer."""

from infrastructure.languages.python.analyzer import PythonTestAnalyzer

class TestPythonTestAnalyzer:
    """Test cases for PythonTestAnalyzer logic."""

    def test_ansi_sanitization(self) -> None:
        """Test that ANSI codes are stripped from details."""
        analyzer = PythonTestAnalyzer()
        text_with_ansi = "\x1b[31mRed Text\x1b[0m and \x1b[1mBold\x1b[0m"
        sanitized = analyzer._sanitize_details(text_with_ansi)
        assert sanitized == "Red Text and Bold"

    def test_temp_path_sanitization(self) -> None:
        """Test that temporary file paths are replaced with test_script.py."""
        analyzer = PythonTestAnalyzer()
        text_with_paths = "Error in /tmp/tmpxyz123.py:42 and C:\\Temp\\tmpabc456.py:10"
        sanitized = analyzer._sanitize_details(text_with_paths)
        assert sanitized is not None
        assert "test_script.py:42" in sanitized
        assert "test_script.py:10" in sanitized
        assert "tmpxyz123" not in sanitized
