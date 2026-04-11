"""Property test population — shared data types.

IOPair: a single (input_arg, output) pair captured by running a code individual
        against one input in the target language.
IOPairCache: thread-safe store shared between PropertyTestInitializer and
             PropertyTestEvaluator (and future operators).
"""

from __future__ import annotations

import threading
from typing import Optional, TypedDict


class IOPair(TypedDict):
    """A single (input_arg, output) pair produced by executing a code individual."""

    input_arg: str  # raw input string in the property-eval wire format, e.g. json.dumps({"input_arg": <dict>})
    output: str  # raw stdout captured from running the code in the target language


class IOPairCache:
    """Thread-safe store with two responsibilities:

    1. **Generator script** — written once by the initializer; read by the evaluator
       on the first call to execute_tests to produce test inputs.
    2. **Per-code IOPairs** — populated by the evaluator on first encounter of each
       code individual; reused in all subsequent epochs.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generator_script: Optional[str] = None
        self._generated_inputs: list[str] = []
        self._pairs: dict[str, list[IOPair]] = {}

    # ── Generator script ────────────────────────────────────────────────────────

    def store_generator_script(self, script: str) -> None:
        with self._lock:
            self._generator_script = script

    def get_generator_script(self) -> Optional[str]:
        return self._generator_script

    # ── Generated inputs ─────────────────────────────────────────────────────────

    def store_generated_inputs(self, inputs: list[str]) -> None:
        with self._lock:
            self._generated_inputs = list(inputs)

    def get_generated_inputs(self) -> list[str]:
        return list(self._generated_inputs)

    # ── Per-code-individual (input, output) pairs ────────────────────────────────

    def store(self, code_id: str, pairs: list[IOPair]) -> None:
        with self._lock:
            self._pairs[code_id] = list(pairs)

    def get(self, code_id: str) -> list[IOPair]:
        """Return cached pairs for code_id, or [] if not yet populated."""
        return list(self._pairs.get(code_id, []))

    def has(self, code_id: str) -> bool:
        return code_id in self._pairs


__all__ = ["IOPair", "IOPairCache"]
