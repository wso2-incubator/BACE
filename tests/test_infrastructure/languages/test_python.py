import pytest

from infrastructure.languages.python import PythonLanguage


class TestDocstringExtraction:
    @pytest.fixture
    def adapter(self) -> PythonLanguage:
        return PythonLanguage()

    def test_priority_1_class_docstring(self, adapter: PythonLanguage) -> None:
        """If class has a docstring, ignore everything else."""
        code = """
class Solution:
    \"\"\"I am the class docstring\"\"\"
    
    # I am a method comment
    def solve(self):
        \"\"\"I am the method docstring\"\"\"
        pass
"""
        assert adapter.get_docstring(code) == "I am the class docstring"

    def test_priority_2_method_docstring(self, adapter: PythonLanguage) -> None:
        """If class has NO docstring, grab the method's docstring."""
        code = """
class Solution:
    def solve(self):
        \"\"\"I am the method docstring\"\"\"
        pass
"""
        assert adapter.get_docstring(code) == "I am the method docstring"

    def test_priority_3_method_comments(self, adapter: PythonLanguage) -> None:
        """If class/method have no docstrings, grab method comments."""
        code = """
class Solution:
    # I am the method comment
    def solve(self):
        pass
"""
        assert adapter.get_docstring(code) == "I am the method comment"

    def test_priority_4_class_comments(self, adapter: PythonLanguage) -> None:
        """If nothing else exists, grab comments above the class."""
        code = """
# I am the class comment
class Solution:
    def solve(self):
        pass
"""
        assert adapter.get_docstring(code) == "I am the class comment"

    def test_mixed_comments_precedence(self, adapter: PythonLanguage) -> None:
        """Ensure method comments beat class comments if class docstring is missing."""
        code = """
# Class comment (should be ignored)
class Solution:
    # Method comment (should be picked)
    def solve(self):
        pass
"""
        assert adapter.get_docstring(code) == "Method comment (should be picked)"

    def test_leetcode_boilerplate(self, adapter: PythonLanguage) -> None:
        """Real-world LeetCode scenario."""
        code = """
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        \"\"\"
        Given an array of integers nums and an integer target, return indices.
        \"\"\"
        return []
"""
        expected = (
            "Given an array of integers nums and an integer target, return indices."
        )
        assert adapter.get_docstring(code).strip() == expected
