"""Math answer verifier.

Extracts a final answer from a free-form model response and compares it to a
reference by, in order of preference: exact string equality, numeric equality
within a relative tolerance, and (when ``sympy`` is installed) symbolic
equality.  This mirrors the answer-checking used in RLVR math pipelines
(e.g. the MATH / GSM8K verifiers behind DeepSeek-R1, Guo et al. 2025).
"""
from __future__ import annotations

import math
import re
from typing import Any, Optional, Tuple

from .._optional import has, require
from ..base import Verifier
from ..registry import register
from ..types import VerifierResult

__all__ = ["MathVerifier"]

# ``\boxed{...}`` with brace balancing handled separately; "answer:" patterns.
_ANSWER_PATTERNS = [
    re.compile(r"answer\s*(?:is)?\s*[:=]\s*(.+)", re.IGNORECASE),
    re.compile(r"final\s+answer\s*[:=]?\s*(.+)", re.IGNORECASE),
    re.compile(r"####\s*(.+)"),  # GSM8K-style.
]
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


@register("math", category="verifier")
class MathVerifier(Verifier):
    """Verify a math answer against a ground-truth reference.

    Parameters
    ----------
    rtol:
        Relative tolerance used for numeric comparison.
    use_sympy:
        If True (default) attempt symbolic equality when ``sympy`` is present.
    """

    name = "math"

    def __init__(self, rtol: float = 1e-6, use_sympy: bool = True) -> None:
        self.rtol = float(rtol)
        self.use_sympy = bool(use_sympy)

    # -- extraction ------------------------------------------------------- #
    @staticmethod
    def _extract_boxed(text: str) -> Optional[str]:
        """Return the contents of the last ``\\boxed{...}`` with brace balance."""
        idx = text.rfind(r"\boxed")
        if idx < 0:
            return None
        i = text.find("{", idx)
        if i < 0:
            return None
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1 : j]
        return None

    @classmethod
    def extract_answer(cls, text: str) -> Tuple[Optional[str], str]:
        """Return ``(answer, how)`` describing the parse strategy used."""
        if text is None:
            return None, "empty response"
        boxed = cls._extract_boxed(text)
        if boxed is not None:
            return boxed, "boxed{...}"
        for pat in _ANSWER_PATTERNS:
            m = pat.search(text)
            if m:
                # Take the first line of the captured group.
                cand = m.group(1).strip().splitlines()[0].strip()
                if cand:
                    return cand, f"pattern '{pat.pattern[:20]}'"
        nums = _NUMBER_RE.findall(text)
        if nums:
            return nums[-1], "last number"
        return None, "no answer found"

    @staticmethod
    def _normalize(s: Any) -> str:
        s = str(s).strip()
        # Strip latex math delimiters and common decorations.
        s = s.replace("$", "").replace("\\!", "").replace("\\,", "")
        s = s.replace("\\left", "").replace("\\right", "")
        s = s.replace("\\%", "").replace("%", "")
        s = s.replace(",", "").replace(" ", "")
        s = s.strip("{}")
        if s.endswith("."):
            s = s[:-1]
        return s

    @staticmethod
    def _to_float(s: str) -> Optional[float]:
        try:
            return float(s)
        except (ValueError, TypeError):
            # Allow simple fractions like "3/4".
            if "/" in s:
                try:
                    num, den = s.split("/", 1)
                    return float(num) / float(den)
                except (ValueError, ZeroDivisionError):
                    return None
            return None

    def _numeric_equal(self, a: float, b: float) -> bool:
        return math.isclose(a, b, rel_tol=self.rtol, abs_tol=1e-9)

    def _symbolic_equal(self, a: str, b: str) -> Optional[bool]:
        if not (self.use_sympy and has("sympy")):
            return None
        try:
            sympy = require("sympy", feature="symbolic math equality")
            from sympy.parsing.sympy_parser import (  # type: ignore
                parse_expr,
                standard_transformations,
                implicit_multiplication_application,
            )

            tr = standard_transformations + (implicit_multiplication_application,)
            ea = parse_expr(a, transformations=tr)
            eb = parse_expr(b, transformations=tr)
            return bool(sympy.simplify(ea - eb) == 0)
        except Exception:  # pragma: no cover - sympy parse failures are benign
            return None

    # -- verify ----------------------------------------------------------- #
    def verify(
        self,
        response: str,
        reference: Any = None,
        *,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> VerifierResult:
        ans, how = self.extract_answer(response)
        if ans is None:
            return VerifierResult.fail(detail=f"parse failed: {how}")
        if reference is None:
            return VerifierResult(
                reward=0.0,
                passed=False,
                detail=f"extracted '{ans}' via {how}; no reference to compare",
                meta={"answer": ans},
            )

        na, nb = self._normalize(ans), self._normalize(reference)
        detail = f"parsed '{ans}' via {how}; normalized '{na}' vs '{nb}'"

        # 1) exact string match.
        if na == nb:
            return VerifierResult.ok(detail=detail + " -> exact string match")

        # 2) numeric tolerance.
        fa, fb = self._to_float(na), self._to_float(nb)
        if fa is not None and fb is not None and self._numeric_equal(fa, fb):
            return VerifierResult.ok(detail=detail + f" -> numeric match (rtol={self.rtol})")

        # 3) symbolic equality.
        sym = self._symbolic_equal(na, nb)
        if sym:
            return VerifierResult.ok(detail=detail + " -> symbolic match (sympy)")

        suffix = " -> mismatch"
        if sym is None and self.use_sympy and not has("sympy"):
            suffix += " (sympy unavailable)"
        return VerifierResult(
            reward=0.0, passed=False, detail=detail + suffix, meta={"answer": ans}
        )
