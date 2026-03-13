"""Tests for PropertyTestInitializer.

Strategy
--------
- ``create_sandbox`` is patched at module level to return an ``InProcessSandbox``
  that runs code via exec() — no filesystem or subprocess required.
- ``_call_gen_inputs`` / ``_call_gen_property`` are mocked at the instance level
  where needed, bypassing tenacity retry timing for failure scenarios.
- All happy-path tests use a real LLM mock (``generate`` returns valid content)
  so the full code-block-extraction  and validation paths are exercised.
"""

from __future__ import annotations

import io
import textwrap
from contextlib import redirect_stdout
from typing import Any
from unittest.mock import MagicMock, patch

from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import OPERATION_INITIAL, PopulationConfig
from coevolution.core.interfaces.data import BasicExecutionResult, Problem, Test
from coevolution.populations.property.operators.initializer import (
    PropertyTestInitializer,
)
from coevolution.populations.property.types import IOPairCache
from infrastructure.languages import PythonLanguage
from infrastructure.sandbox import SandboxConfig

# ── In-process sandbox (no subprocess / tempfile) ─────────────────────────────


class InProcessSandbox:
    def execute_code(self, code: str, runtime: Any) -> BasicExecutionResult:
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(code, {})  # noqa: S102
        except Exception as exc:
            return BasicExecutionResult(
                success=False,
                output=buf.getvalue(),
                error=str(exc),
                execution_time=0.0,
                timeout=False,
                return_code=1,
            )
        return BasicExecutionResult(
            success=True,
            output=buf.getvalue(),
            error="",
            execution_time=0.0,
            timeout=False,
            return_code=0,
        )

    def execute_command(self, *a: Any, **kw: Any) -> Any:
        raise NotImplementedError

    def execute_test_script(self, *a: Any, **kw: Any) -> Any:
        raise NotImplementedError


# ── Fixtures / helpers ─────────────────────────────────────────────────────────

SORT_PROBLEM = Problem(
    question_title="sort",
    question_content="Sort a list of integers in ascending order.",
    question_id="test/sort",
    starter_code="def sort(lst): ...",
    public_test_cases=[
        Test(input="[3, 1, 2]", output="[1, 2, 3]"),
        Test(input="[5, 5, 1]", output="[1, 5, 5]"),
    ],
    private_test_cases=[],
)

SANDBOX_CONFIG = SandboxConfig(language="python", timeout=10)
POP_CONFIG = PopulationConfig(
    initial_prior=0.6,
    initial_population_size=5,
    max_population_size=10,
)

# A minimal valid generator script (LLM call 1 response)
GEN_INPUTS_RESPONSE = textwrap.dedent("""\
    ```python
    def generate_test_inputs(num_inputs: int) -> list[dict]:
        return [{"lst": [i, i+1]} for i in range(num_inputs)]
    ```
""")

# Two candidate property snippets (LLM call 2 response)
PROPERTY_SNIPPET_PASS = textwrap.dedent("""\
    def property_same_length(inputdata, output):
        import json
        inp = json.loads(inputdata)
        out = json.loads(output)
        return len(inp["lst"]) == len(out)
""")

PROPERTY_SNIPPET_FAIL = textwrap.dedent("""\
    def property_always_false(inputdata, output):
        return False
""")

# Stage 1 response: descriptions
PROPERTY_DESCRIPTIONS_RESPONSE = textwrap.dedent("""\
    <property_description>The length of the returned list is identical to the length of the input list.</property_description>
    <property_description>The elements in the returned list are sorted in ascending order.</property_description>
""")
 
# Stage 2 response: snippets
PROPERTY_SNIPPET_PASS_RESPONSE = f"```python\n{PROPERTY_SNIPPET_PASS}```"
PROPERTY_SNIPPET_FAIL_RESPONSE = f"```python\n{PROPERTY_SNIPPET_FAIL}```"


def make_initializer(
    llm_generate_side_effect: Any = None,
    cache: IOPairCache | None = None,
) -> tuple[PropertyTestInitializer, IOPairCache, MagicMock]:
    """Build an initializer with a mock LLM and an InProcessSandbox."""
    mock_llm = MagicMock()
    if llm_generate_side_effect is not None:
        mock_llm.generate.side_effect = llm_generate_side_effect
    cache = cache or IOPairCache()
    python = PythonLanguage()
    initializer = PropertyTestInitializer(
        llm=mock_llm,
        parser=python.parser,
        language_name="python",
        pop_config=POP_CONFIG,
        sandbox_config=SANDBOX_CONFIG,
        io_pair_cache=cache,
    )
    return initializer, cache, mock_llm


# patch target — create_sandbox is imported into the initializer module
_PATCH = "coevolution.populations.property.operators.initializer.create_sandbox"


# ── Happy-path tests ───────────────────────────────────────────────────────────


class TestHappyPath:
    def test_returns_one_individual_from_valid_snippet(self) -> None:
        init, _, mock_llm = make_initializer()
        mock_llm.generate.side_effect = [
            GEN_INPUTS_RESPONSE,
            "<property_description>matches length</property_description>",
            PROPERTY_SNIPPET_PASS_RESPONSE,
        ]

        with patch(_PATCH, return_value=InProcessSandbox()):
            individuals = init.initialize(SORT_PROBLEM)

        assert len(individuals) == 1
        assert isinstance(individuals[0], TestIndividual)

    def test_generator_script_cached_on_success(self) -> None:
        init, cache, mock_llm = make_initializer()
        mock_llm.generate.side_effect = [
            GEN_INPUTS_RESPONSE,
            "<property_description>matches length</property_description>",
            PROPERTY_SNIPPET_PASS_RESPONSE,
        ]
 
        with patch(_PATCH, return_value=InProcessSandbox()):
            init.initialize(SORT_PROBLEM)

        script = cache.get_generator_script()
        assert script is not None
        assert "generate_test_inputs" in script

    def test_individual_has_initial_prior(self) -> None:
        init, _, mock_llm = make_initializer()
        mock_llm.generate.side_effect = [
            GEN_INPUTS_RESPONSE,
            "<property_description>matches length</property_description>",
            PROPERTY_SNIPPET_PASS_RESPONSE,
        ]

        with patch(_PATCH, return_value=InProcessSandbox()):
            individuals = init.initialize(SORT_PROBLEM)

        assert individuals[0].probability == POP_CONFIG.initial_prior

    def test_individual_creation_op_is_initial(self) -> None:
        init, _, mock_llm = make_initializer()
        mock_llm.generate.side_effect = [
            GEN_INPUTS_RESPONSE,
            "<property_description>matches length</property_description>",
            PROPERTY_SNIPPET_PASS_RESPONSE,
        ]

        with patch(_PATCH, return_value=InProcessSandbox()):
            individuals = init.initialize(SORT_PROBLEM)

        assert individuals[0].creation_op == OPERATION_INITIAL

    def test_individual_generation_born_is_zero(self) -> None:
        init, _, mock_llm = make_initializer()
        mock_llm.generate.side_effect = [
            GEN_INPUTS_RESPONSE,
            "<property_description>matches length</property_description>",
            PROPERTY_SNIPPET_PASS_RESPONSE,
        ]

        with patch(_PATCH, return_value=InProcessSandbox()):
            individuals = init.initialize(SORT_PROBLEM)

        assert individuals[0].generation_born == 0

    def test_individual_snippet_content(self) -> None:
        init, _, mock_llm = make_initializer()
        mock_llm.generate.side_effect = [
            GEN_INPUTS_RESPONSE,
            "<property_description>matches length</property_description>",
            PROPERTY_SNIPPET_PASS_RESPONSE,
        ]

        with patch(_PATCH, return_value=InProcessSandbox()):
            individuals = init.initialize(SORT_PROBLEM)

        assert "def property_same_length" in individuals[0].snippet

    def test_only_passing_snippets_kept_from_mixed_response(self) -> None:
        """Two snippets in response; one always returns False → only 1 individual."""
        init, _, mock_llm = make_initializer()
        mock_llm.generate.side_effect = [
            GEN_INPUTS_RESPONSE,
            PROPERTY_DESCRIPTIONS_RESPONSE,
            PROPERTY_SNIPPET_PASS_RESPONSE,
            PROPERTY_SNIPPET_FAIL_RESPONSE,
        ]
 
        with patch(_PATCH, return_value=InProcessSandbox()):
            individuals = init.initialize(SORT_PROBLEM)

        assert len(individuals) == 1
        assert "property_same_length" in individuals[0].snippet

    def test_empty_public_tests_accepts_all_syntactically_valid(self) -> None:
        """With no public test cases, any syntactically valid snippet is accepted."""
        problem = Problem(
            question_title="x",
            question_content="x",
            question_id="x",
            starter_code="",
            public_test_cases=[],
            private_test_cases=[],
        )
        init, _, mock_llm = make_initializer()
        mock_llm.generate.side_effect = [
            GEN_INPUTS_RESPONSE,
            "<property_description>always false</property_description>",
            PROPERTY_SNIPPET_FAIL_RESPONSE,
        ]
 
        with patch(_PATCH, return_value=InProcessSandbox()):
            individuals = init.initialize(problem)

        assert len(individuals) == 1


# ── gen_inputs failure scenarios ───────────────────────────────────────────────


class TestGenInputsFailures:
    def test_gen_inputs_llm_failure_still_returns_property_individuals(self) -> None:
        """gen_inputs crashing should not prevent property test generation."""
        init, _, mock_llm = make_initializer()
        mock_llm.generate.side_effect = [
            "<property_description>matches length</property_description>",
            PROPERTY_SNIPPET_PASS_RESPONSE,
        ]
 
        with (
            patch.object(init, "_call_gen_inputs", side_effect=Exception("LLM down")),
            patch(_PATCH, return_value=InProcessSandbox()),
        ):
            individuals = init.initialize(SORT_PROBLEM)

        assert len(individuals) == 1

    def test_gen_inputs_llm_failure_nothing_cached(self) -> None:
        init, cache, mock_llm = make_initializer()
        mock_llm.generate.side_effect = [
            "<property_description>matches length</property_description>",
            PROPERTY_SNIPPET_PASS_RESPONSE,
        ]
 
        with (
            patch.object(init, "_call_gen_inputs", side_effect=Exception("LLM down")),
            patch(_PATCH, return_value=InProcessSandbox()),
        ):
            init.initialize(SORT_PROBLEM)

        assert cache.get_generator_script() is None

    def test_gen_inputs_no_code_block_nothing_cached(self) -> None:
        """LLM returns a response with no code block → script discarded."""
        init, cache, mock_llm = make_initializer()
        mock_llm.generate.side_effect = [
            "<property_description>matches length</property_description>",
            PROPERTY_SNIPPET_PASS_RESPONSE,
        ]
 
        with (
            patch.object(
                init,
                "_call_gen_inputs",
                side_effect=Exception(
                    "gen_inputs: no code block found in LLM response."
                ),
            ),
            patch(_PATCH, return_value=InProcessSandbox()),
        ):
            init.initialize(SORT_PROBLEM)

        assert cache.get_generator_script() is None

    def test_gen_inputs_invalid_python_not_cached(self) -> None:
        """gen_inputs returns syntactically invalid Python → script discarded."""
        init, cache, mock_llm = make_initializer()
        mock_llm.generate.side_effect = [
            "<property_description>matches length</property_description>",
            PROPERTY_SNIPPET_PASS_RESPONSE,
        ]
 
        with (
            patch.object(init, "_call_gen_inputs", return_value="def broken(:"),
            patch(_PATCH, return_value=InProcessSandbox()),
        ):
            init.initialize(SORT_PROBLEM)

        assert cache.get_generator_script() is None


# ── gen_property failure scenarios ────────────────────────────────────────────


class TestGenPropertyFailures:
    def test_gen_property_llm_failure_returns_empty(self) -> None:
        init, _, _ = make_initializer()

        with (
            patch.object(
                init,
                "_call_gen_inputs",
                return_value="def generate_inputs(n): return []",
            ),
            patch.object(init, "_call_gen_property", side_effect=Exception("LLM down")),
            patch(_PATCH, return_value=InProcessSandbox()),
        ):
            individuals = init.initialize(SORT_PROBLEM)

        assert individuals == []

    def test_snippet_without_property_prefix_rejected(self) -> None:
        """A snippet whose function name doesn't start with property_ is rejected."""
        bad_snippet = "def check_output(inputdata, output):\n    return True\n"
        init, _, mock_llm = make_initializer()
        mock_llm.generate.side_effect = [
            GEN_INPUTS_RESPONSE,
            "<property_description>check output</property_description>",
            f"```python\n{bad_snippet}```",
        ]
 
        with patch(_PATCH, return_value=InProcessSandbox()):
            individuals = init.initialize(SORT_PROBLEM)

        assert individuals == []

    def test_crashing_snippet_rejected(self) -> None:
        crashing = (
            "def property_crashes(inputdata, output):\n    raise ValueError('boom')\n"
        )
        init, _, mock_llm = make_initializer()
        mock_llm.generate.side_effect = [
            GEN_INPUTS_RESPONSE,
            "<property_description>crashes</property_description>",
            f"```python\n{crashing}```",
        ]
 
        with patch(_PATCH, return_value=InProcessSandbox()):
            individuals = init.initialize(SORT_PROBLEM)

        assert individuals == []

    def test_syntactically_invalid_snippet_excluded(self) -> None:
        """Blocks that are syntactically invalid are dropped before validation."""
        # PythonParser.extract_code_blocks already filters invalid syntax,
        # so the initializer should get zero candidates and return [].
        init, _, mock_llm = make_initializer()
        mock_llm.generate.side_effect = [
            GEN_INPUTS_RESPONSE,
            "<property_description>broken</property_description>",
            "```python\ndef property_broken(:\n```",
        ]
 
        with patch(_PATCH, return_value=InProcessSandbox()):
            individuals = init.initialize(SORT_PROBLEM)

        assert individuals == []

    def test_gen_property_returns_empty_list_yields_no_individuals(self) -> None:
        init, _, _ = make_initializer()

        with (
            patch.object(
                init,
                "_call_gen_inputs",
                return_value="def generate_inputs(n): return []",
            ),
            patch.object(init, "_call_gen_property", return_value=[]),
            patch(_PATCH, return_value=InProcessSandbox()),
        ):
            individuals = init.initialize(SORT_PROBLEM)

        assert individuals == []
