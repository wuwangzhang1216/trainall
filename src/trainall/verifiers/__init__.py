"""Deterministic verifiers — the engine of RLVR.

Each turns ``(response, reference)`` into a :class:`trainall.types.VerifierResult`
with a reward in ``[0, 1]`` and a boolean ``passed``.  All are pure-python
(``sympy`` / ``jsonschema`` are optional niceties), so this package imports
with no ML stack.

* :class:`MathVerifier`     — numeric / symbolic answer equality
* :class:`CodeVerifier`     — run candidate against unit tests in a subprocess
* :class:`SQLVerifier`      — execute against SQLite, compare result sets
* :class:`JSONVerifier`     — validity + optional JSON-Schema conformance
* :class:`FormatVerifier` / :class:`RegexVerifier` — structural / pattern checks
* :class:`CitationVerifier` — quoted spans must exist in the provided sources
* :class:`CompositeVerifier`— weighted combination of the above
"""
from __future__ import annotations

from .math_verifier import MathVerifier
from .code_verifier import CodeVerifier
from .sql_verifier import SQLVerifier
from .json_verifier import JSONVerifier
from .format_verifier import FormatVerifier, RegexVerifier
from .citation_verifier import CitationVerifier
from .composite import CompositeVerifier

__all__ = [
    "MathVerifier",
    "CodeVerifier",
    "SQLVerifier",
    "JSONVerifier",
    "FormatVerifier",
    "RegexVerifier",
    "CitationVerifier",
    "CompositeVerifier",
]
