"""Tests for code_preprocessing.composition module."""

import pytest

from common.code_preprocessing.composition import (
    compose_lcb_test_script,
    rebuild_unittest_with_methods,
)
from common.code_preprocessing.exceptions import (
    CodeParsingError,
    CodeTransformationError,
)


class TestComposeLcbTestScript:
    """Test compose_lcb_test_script function."""

    def test_combines_solution_class_and_tests(self) -> None:
        prog = """
class Solution:
    def solve(self):
        return 42
"""
        test = """
import unittest

class TestSolution(unittest.TestCase):
    def test_solve(self):
        self.assertEqual(Solution().solve(), 42)
"""
        script = compose_lcb_test_script(prog, test)
        assert "class Solution" in script
        assert "class TestSolution" in script

    def test_wraps_functions_into_solution_class(self) -> None:
        prog = "def solve():\n    return 42"
        test = "import unittest\nclass TestSolution(unittest.TestCase): pass"
        script = compose_lcb_test_script(prog, test)
        assert "class Solution" in script
        # Function should be converted to method
        assert "def solve(self)" in script

    def test_removes_solution_imports_from_test(self) -> None:
        prog = "class Solution:\n    pass"
        test = """
from solution import Solution
import unittest

class TestSolution(unittest.TestCase):
    pass
"""
        script = compose_lcb_test_script(prog, test)
        assert "from solution import Solution" not in script

    def test_adds_unittest_import_if_missing(self) -> None:
        prog = "class Solution:\n    pass"
        test = "class TestSolution:\n    pass"
        script = compose_lcb_test_script(prog, test)
        assert "import unittest" in script

    def test_raises_error_when_no_solution_found(self) -> None:
        prog = "x = 1"  # No Solution class or functions
        test = "import unittest\nclass TestSolution(unittest.TestCase): pass"
        with pytest.raises(CodeTransformationError, match="No Solution class"):
            compose_lcb_test_script(prog, test)

    def test_preserves_helper_classes(self) -> None:
        prog = """
class Helper:
    pass

class Solution:
    pass
"""
        test = "import unittest\nclass TestSolution(unittest.TestCase): pass"
        script = compose_lcb_test_script(prog, test)
        assert "class Helper" in script
        assert "class Solution" in script


class TestRebuildUnittestWithMethods:
    """Test rebuild_unittest_with_methods function."""

    def test_replaces_test_methods(self) -> None:
        original = """
import unittest

class TestFoo(unittest.TestCase):
    def test_old(self):
        pass
"""
        new_methods = ["def test_new(self):\n    self.assertTrue(True)"]
        result = rebuild_unittest_with_methods(original, new_methods)
        assert "test_new" in result
        assert "test_old" not in result

    def test_preserves_imports(self) -> None:
        original = """
import unittest
import os

class TestFoo(unittest.TestCase):
    def test_old(self):
        pass
"""
        new_methods = ["def test_new(self): pass"]
        result = rebuild_unittest_with_methods(original, new_methods)
        assert "import unittest" in result
        assert "import os" in result

    def test_preserves_setup_and_teardown(self) -> None:
        original = """
class TestFoo(unittest.TestCase):
    def setUp(self):
        self.x = 1
    
    def tearDown(self):
        self.x = None
    
    def test_old(self):
        pass
"""
        new_methods = ["def test_new(self): pass"]
        result = rebuild_unittest_with_methods(original, new_methods)
        assert "def setUp" in result
        assert "def tearDown" in result

    def test_handles_multiple_new_methods(self) -> None:
        original = "class TestFoo(unittest.TestCase):\n    def test_old(self): pass"
        new_methods = [
            "def test_one(self): pass",
            "def test_two(self): pass",
            "def test_three(self): pass",
        ]
        result = rebuild_unittest_with_methods(original, new_methods)
        assert "test_one" in result
        assert "test_two" in result
        assert "test_three" in result

    def test_raises_error_when_no_class(self) -> None:
        code = "def foo(): pass"
        with pytest.raises(CodeParsingError, match="No class definition found"):
            rebuild_unittest_with_methods(code, [])

    def test_skips_invalid_method_code(self) -> None:
        original = "class TestFoo(unittest.TestCase):\n    def test_old(self): pass"
        new_methods = [
            "def test_valid(self): pass",
            "invalid syntax here",
            "def test_also_valid(self): pass",
        ]
        result = rebuild_unittest_with_methods(original, new_methods)
        # Should include valid methods, skip invalid
        assert "test_valid" in result
        assert "test_also_valid" in result
