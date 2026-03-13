"""Helper functions for property test operators."""

from __future__ import annotations

import json
import re
from loguru import logger

from coevolution.core.interfaces.data import Test
from coevolution.core.interfaces.language import ICodeParser, LanguageTransformationError


def transform_public_tests(
    tests: list[Test], starter_code: str, parser: ICodeParser
) -> list[Test]:
    """Transform raw public test inputs into the JSON protocol expected by property tests.

    Delegates language-specific parsing to the ICodeParser.
    """
    if not tests:
        return []

    transformed = []
    for t in tests:
        try:
            # Delegate parsing to the language-specific parser
            # Each parser knows how to turn its own raw public test strings into a dict of args
            input_dict, output_val = parser.parse_public_test(
                t.input, t.output, starter_code
            )

            # Create JSON-serialized Test object for the property test protocol
            transformed.append(
                Test(input=json.dumps(input_dict), output=json.dumps(output_val))
            )
        except Exception as e:
            logger.debug(f"Failed to transform public test: {e}")
            # Fallback: if transformation fails, keep the original test
            # (Evaluation might fail later, but we shouldn't crash initialization)
            transformed.append(t)

    return transformed


def compose_property_test_script(snippet: str, inputdata: str, output: str) -> str:
    """Combine a property test snippet with one (inputdata, output) pair.

    Produces a self-contained Python script whose stdout is either ``True`` or
    ``False``.

    Args:
        snippet:   Full property test function definition, e.g.::

                       def property_sorted_output(inputdata, output):
                           return output == sorted(output)

        inputdata: Raw input string (verbatim from IOPair.inputdata or Test.input).
        output:    Raw output string (verbatim from IOPair.output or Test.output).

    Returns:
        A runnable Python script string.

    Raises:
        LanguageTransformationError: if no ``def property_<name>(`` function is
            found in *snippet*.
    """
    match = re.search(r"def (property_\w+)\s*\(", snippet)
    if not match:
        raise LanguageTransformationError(
            f"No property_<name> function found in snippet: {snippet[:80]!r}"
        )
    fn_name = match.group(1)
    safe_input = repr(inputdata)
    safe_output = repr(output)
    return (
        f"{snippet}\n\n"
        f"result = {fn_name}(\n"
        f"    inputdata={safe_input},\n"
        f"    output={safe_output},\n"
        f")\n"
        f"print(result)\n"
    )
