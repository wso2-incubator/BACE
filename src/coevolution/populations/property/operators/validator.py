"""Shared validation helper for property test operators."""

from __future__ import annotations
import json
from coevolution.core.interfaces import Test
from coevolution.core.interfaces.sandbox import ISandbox
from infrastructure.languages import PythonLanguage

from .helpers import compose_property_test_script


def validate_property_test(
    snippet: str,
    public_test_cases: list[Test],
    sandbox: ISandbox,
) -> bool:
    """Return True iff the property test snippet passes ALL public test cases.

    A snippet is rejected on the first failure so that validation short-circuits
    as quickly as possible.  Uses a fresh temporary directory per property
    test to avoid any state leakage between runs.

    Args:
        snippet:            Python source of one ``def property_<name>(...)`` function.
        public_test_cases:  Test cases from the Problem object.
        sandbox:            An ISandbox instance connected to a Python execution environment.

    Returns:
        ``True`` if the snippet returns ``True`` for every test case,
        ``False`` otherwise (includes execution errors and malformed snippets).
    """
    if not public_test_cases:
        # Nothing to validate against — accept the snippet.
        return True

    python = PythonLanguage()

    for test in public_test_cases:
        try:
            # test.input and test.output are JSON strings in the transformed public tests
            input_dict = json.loads(test.input)
            output_val = json.loads(test.output)
            script = compose_property_test_script(snippet, input_dict, output_val)
        except Exception:
            # compose_property_test_script raises LanguageTransformationError
            # if no property_<name> function is found.
            return False

        try:
            exec_result = sandbox.execute_code(script, python.runtime)
        except Exception:
            return False

        if exec_result.error:
            return False

        stdout = exec_result.output.strip()
        last_line = stdout.splitlines()[-1] if stdout else ""
        if last_line != "True":
            return False

    return True


__all__ = ["validate_property_test"]
