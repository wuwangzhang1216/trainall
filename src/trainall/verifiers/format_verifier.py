"""Structural / pattern verifiers.

* :class:`FormatVerifier` checks that a response *looks right* structurally:
  required keys present, required tags balanced, and a set of named regexes all
  matching.  This is the kind of cheap format reward used to teach models to
  emit ``<think>...</think><answer>...</answer>`` scaffolds (DeepSeek-R1,
  Guo et al. 2025).
* :class:`RegexVerifier` is the minimal single-pattern check.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..base import Verifier
from ..registry import register
from ..types import VerifierResult

__all__ = ["FormatVerifier", "RegexVerifier"]


@register("format", category="verifier")
class FormatVerifier(Verifier):
    """Check presence of required keys, tags and pattern matches.

    Parameters
    ----------
    required_keys:
        Substrings/section labels that must appear verbatim in the response.
    regexes:
        A pattern (or mapping ``name -> pattern``) that must each ``search``.
    must_have_tags:
        Tag names (e.g. ``"think"``) requiring both ``<tag>`` and ``</tag>``.
    flags:
        ``re`` flags applied to ``regexes``.
    """

    name = "format"

    def __init__(
        self,
        required_keys: Optional[Sequence[str]] = None,
        regexes: Any = None,
        must_have_tags: Optional[Sequence[str]] = None,
        flags: int = 0,
    ) -> None:
        self.required_keys = list(required_keys or [])
        self.must_have_tags = list(must_have_tags or [])
        self.flags = flags
        self._regexes: Dict[str, "re.Pattern[str]"] = {}
        if regexes is None:
            pass
        elif isinstance(regexes, Mapping):
            self._regexes = {k: re.compile(v, flags) for k, v in regexes.items()}
        elif isinstance(regexes, (list, tuple)):
            self._regexes = {f"re[{i}]": re.compile(p, flags) for i, p in enumerate(regexes)}
        else:  # single pattern string/compiled
            self._regexes = {"re": re.compile(regexes, flags)}

    def verify(
        self,
        response: str,
        reference: Any = None,
        *,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> VerifierResult:
        text = response or ""
        checks: List[bool] = []
        failures: List[str] = []

        for key in self.required_keys:
            ok = key in text
            checks.append(ok)
            if not ok:
                failures.append(f"missing key '{key}'")

        for tag in self.must_have_tags:
            ok = f"<{tag}>" in text and f"</{tag}>" in text
            checks.append(ok)
            if not ok:
                failures.append(f"missing/unbalanced <{tag}>")

        for nm, pat in self._regexes.items():
            ok = pat.search(text) is not None
            checks.append(ok)
            if not ok:
                failures.append(f"regex {nm} did not match")

        if not checks:
            return VerifierResult(
                reward=1.0,
                passed=True,
                detail="no constraints configured; vacuously passes",
            )

        n_pass = sum(checks)
        reward = n_pass / len(checks)
        passed = n_pass == len(checks)
        detail = f"{n_pass}/{len(checks)} structural checks passed"
        if failures:
            detail += "; failures: " + "; ".join(failures)
        return VerifierResult(reward=reward, passed=passed, detail=detail)


@register("regex", category="verifier")
class RegexVerifier(Verifier):
    """Binary pass/fail on a single regular-expression search.

    Parameters
    ----------
    pattern:
        Regex to ``search`` for in the response.  May be overridden per-call by
        passing a string ``reference``.
    flags:
        ``re`` flags.
    """

    name = "regex"

    def __init__(self, pattern: str = r".+", flags: int = 0) -> None:
        self.pattern_str = pattern
        self.flags = flags
        self._pattern = re.compile(pattern, flags)

    def verify(
        self,
        response: str,
        reference: Any = None,
        *,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> VerifierResult:
        pat = self._pattern
        if isinstance(reference, str):
            pat = re.compile(reference, self.flags)
        m = pat.search(response or "")
        if m:
            return VerifierResult.ok(detail=f"matched /{pat.pattern}/ at {m.start()}")
        return VerifierResult.fail(detail=f"no match for /{pat.pattern}/")
