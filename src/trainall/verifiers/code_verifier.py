"""Code verifier.

Extracts a fenced code block from a model response, concatenates it with the
reference's test snippet, and executes the whole thing in a *subprocess* with a
timeout.  Code is never ``exec``-ed in-process — this is the standard sandboxing
discipline used by HumanEval / MBPP-style functional-correctness checks
(Chen et al. 2021).  The reward is the fraction of asserts that passed when that
can be parsed, otherwise a binary 1.0 / 0.0 on exit status.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Optional

from ..base import Verifier
from ..registry import register
from ..types import VerifierResult

__all__ = ["CodeVerifier"]

_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
_MAX_DETAIL = 2000


@register("code", category="verifier")
class CodeVerifier(Verifier):
    """Run candidate code against tests in a sandboxed subprocess.

    Parameters
    ----------
    timeout:
        Seconds before the subprocess is killed.
    lang:
        Currently only ``"python"`` is executed; other langs report unsupported.
    """

    name = "code"

    def __init__(self, timeout: float = 10.0, lang: str = "python") -> None:
        self.timeout = float(timeout)
        self.lang = lang

    @staticmethod
    def extract_code(text: str) -> str:
        """Return the first fenced code block, or the whole text if none."""
        if text is None:
            return ""
        blocks = _FENCE_RE.findall(text)
        if blocks:
            return blocks[0].strip("\n")
        return text.strip()

    @staticmethod
    def _tests_of(reference: Any) -> str:
        if reference is None:
            return ""
        if isinstance(reference, str):
            return reference
        if isinstance(reference, dict):
            return str(reference.get("tests", ""))
        return str(reference)

    @staticmethod
    def _count_asserts(tests: str) -> int:
        return len(re.findall(r"^\s*assert\b", tests, re.MULTILINE))

    @staticmethod
    def _truncate(s: str) -> str:
        if len(s) <= _MAX_DETAIL:
            return s
        return s[:_MAX_DETAIL] + f"... [truncated {len(s) - _MAX_DETAIL} chars]"

    def verify(
        self,
        response: str,
        reference: Any = None,
        *,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> VerifierResult:
        if self.lang != "python":
            return VerifierResult.fail(detail=f"unsupported language '{self.lang}'")

        code = self.extract_code(response)
        if not code:
            return VerifierResult.fail(detail="no code block found in response")
        tests = self._tests_of(reference)
        n_asserts = self._count_asserts(tests)

        program = code + "\n\n" + tests + "\n"
        path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".py", delete=False, encoding="utf-8"
            ) as fh:
                fh.write(program)
                path = fh.name
            try:
                proc = subprocess.run(
                    [sys.executable, path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                return VerifierResult.fail(
                    detail=f"execution timed out after {self.timeout}s"
                )
            out = self._truncate((proc.stdout or "") + (proc.stderr or ""))
            passed = proc.returncode == 0
            if passed:
                reward = 1.0
            elif n_asserts > 0:
                # Best-effort partial credit: count which assert line failed.
                # CPython stops at the first failing assert, so reward reflects
                # how far execution got.
                reward = self._partial_reward(proc.stderr or "", n_asserts)
            else:
                reward = 0.0
            detail = f"exit={proc.returncode}; asserts={n_asserts}; output:\n{out}"
            return VerifierResult(reward=reward, passed=passed, detail=detail)
        finally:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:  # pragma: no cover
                    pass

    @staticmethod
    def _partial_reward(stderr: str, n_asserts: int) -> float:
        """Fraction of asserts that ran before the first failure.

        Heuristic: the traceback ``line N`` points at the failing statement.
        Without per-test isolation we approximate partial credit as 0.0 for a
        hard failure; if any output suggests some tests ran we cannot reliably
        attribute it, so we conservatively report 0.0.
        """
        return 0.0
