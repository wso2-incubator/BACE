"""Property test population — deterministic script composer.

compose_property_test_script is the analogue of compose_evaluation_script on the
language adapters, but is always Python because property tests are always Python.
"""

from __future__ import annotations

import json
import re

from coevolution.core.interfaces.language import LanguageTransformationError


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
    safe_input = json.dumps(inputdata)
    safe_output = json.dumps(output)
    return (
        f"{snippet}\n\n"
        f"result = {fn_name}(\n"
        f"    inputdata={safe_input},\n"
        f"    output={safe_output},\n"
        f")\n"
        f"print(result)\n"
    )


__all__ = ["compose_property_test_script"]
