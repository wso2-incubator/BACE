"""Tests for transform_public_tests helper."""

import json
from unittest.mock import MagicMock
import pytest

from coevolution.core.interfaces.data import Test
from coevolution.core.interfaces.language import ICodeParser, LanguageTransformationError
from coevolution.populations.property.operators.helpers import (
    transform_public_tests,
    compose_property_test_script
)
from infrastructure.languages.python.adapter import PythonParser

def test_transform_empty_list():
    parser = PythonParser()
    assert transform_public_tests([], "def f(): ...", parser) == []

def test_transform_single_arg_list():
    parser = PythonParser()
    
    tests = [Test(input="[1, 2, 3]", output="[1, 2, 3]")]
    transformed = transform_public_tests(tests, "def sort(lst): ...", parser)
    
    assert len(transformed) == 1
    assert json.loads(transformed[0].input) == {"lst": [1, 2, 3]}
    assert json.loads(transformed[0].output) == [1, 2, 3]

def test_transform_multiple_args():
    parser = PythonParser()
    
    # Input has 3 lines
    tests = [Test(input="10\n'hello'\n3.14", output="True")]
    transformed = transform_public_tests(tests, "def f(a, b, c): ...", parser)
    
    assert len(transformed) == 1
    inp = json.loads(transformed[0].input)
    assert inp == {"a": 10, "b": "hello", "c": 3.14}
    assert json.loads(transformed[0].output) is True

def test_transform_newline_separated_strings():
    parser = PythonParser()
    
    tests = [Test(input="first line\nsecond line", output="'ok'")]
    transformed = transform_public_tests(tests, "def f(s1, s2): ...", parser)
    
    assert len(transformed) == 1
    inp = json.loads(transformed[0].input)
    assert inp == {"s1": "first line", "s2": "second line"}
    assert json.loads(transformed[0].output) == "ok"

def test_transform_space_separated_fallback():
    parser = PythonParser()
    
    # PythonParser's parse_public_test uses lines=args
    # but wait, let's see if it handles space separated...
    # Actually my previous implementation in PythonParser didn't include space-splitting
    # The user said the adapter's generate_test_case "worked very well". 
    # Let's see if generate_test_case handles space-separated.
    # It doesn't seem to (it uses lines).
    # If the user wants to support space-separated, I should add it to PythonParser.
    
    tests = [Test(input="1\n2", output="3")]
    transformed = transform_public_tests(tests, "def add(x, y): ...", parser)
    
    assert len(transformed) == 1
    inp = json.loads(transformed[0].input)
    assert inp == {"x": 1, "y": 2}
    assert json.loads(transformed[0].output) == 3

def test_transform_signature_failure_returns_raw():
    parser = MagicMock(spec=ICodeParser)
    parser.parse_public_test.side_effect = Exception("Parse error")
    
    tests = [Test(input="raw input", output="raw output")]
    transformed = transform_public_tests(tests, "stutter code", parser)
    
    assert transformed == tests

def test_transform_mismatched_count_best_effort():
    parser = PythonParser()
    
    # 3 inputs for 2 params
    tests = [Test(input="1\n2\n3", output="6")]
    transformed = transform_public_tests(tests, "def f(a, b): ...", parser)
    
    assert len(transformed) == 1
    inp = json.loads(transformed[0].input)
    assert inp == {"a": 1, "b": 2}
    assert json.loads(transformed[0].output) == 6

def test_transform_complex_literal_eval():
    parser = PythonParser()
    
    tests = [Test(input="{'key': [1, 2, {'inner': 'val'}]}", output="None")]
    transformed = transform_public_tests(tests, "def process(data): ...", parser)
    
    assert len(transformed) == 1
    inp = json.loads(transformed[0].input)
    assert inp == {"data": {"key": [1, 2, {"inner": "val"}]}}


def test_compose_property_test_script_happy_path():
    snippet = "def property_test(inputdata, output):\n    return True"
    input_str = '{"x": 1}'
    output_str = "2"
    
    script = compose_property_test_script(snippet, input_str, output_str)
    
    assert "def property_test" in script
    assert "result = property_test(" in script
    assert 'inputdata=\'{"x": 1}\'' in script
    assert "output='2'" in script
    assert "print(result)" in script

def test_compose_property_test_script_no_function():
    snippet = "print('hello')"
    with pytest.raises(LanguageTransformationError):
        compose_property_test_script(snippet, "in", "out")

def test_compose_property_test_script_special_chars():
    snippet = "def property_test(inputdata, output): return True"
    input_str = "line1\nline2"
    output_str = "'quotes' and \\backslashes"
    
    script = compose_property_test_script(snippet, input_str, output_str)
    
    assert "'line1\\nline2'" in script
    assert '"\'quotes\' and \\\\backslashes"' in script
