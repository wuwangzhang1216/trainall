"""JSON verifier.

Parses a model response as JSON (extracting the first balanced object/array
block when the response is wrapped in prose or code fences).  A merely valid
document earns a baseline reward; when ``reference`` is a JSON Schema and the
``jsonschema`` package is installed, full schema validation lifts the reward to
1.0 on conformance.  Useful for tool-calling / structured-output RLVR.
"""
from __future__ import annotations

import json
from typing import Any, Optional, Tuple

from .._optional import has, require
from ..base import Verifier
from ..registry import register
from ..types import VerifierResult

__all__ = ["JSONVerifier"]

_VALID_BASELINE = 0.5


@register("json", category="verifier")
class JSONVerifier(Verifier):
    """Validate that a response is JSON and (optionally) schema-conformant.

    Parameters
    ----------
    baseline:
        Reward for syntactically valid JSON when no schema validation applies.
    """

    name = "json"

    def __init__(self, baseline: float = _VALID_BASELINE) -> None:
        self.baseline = float(baseline)

    @staticmethod
    def _extract_block(text: str) -> Optional[str]:
        """Return the first balanced ``{...}`` or ``[...]`` substring."""
        if not text:
            return None
        starts = [(text.find("{"), "{", "}"), (text.find("["), "[", "]")]
        starts = [s for s in starts if s[0] >= 0]
        if not starts:
            return None
        start, open_ch, close_ch = min(starts, key=lambda s: s[0])
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    @classmethod
    def parse(cls, text: str) -> Tuple[Optional[Any], str]:
        """Return ``(obj, how)`` from a direct parse or extracted block."""
        if text is None:
            return None, "empty response"
        try:
            return json.loads(text), "direct parse"
        except (json.JSONDecodeError, TypeError):
            pass
        block = cls._extract_block(text)
        if block is None:
            return None, "no JSON block found"
        try:
            return json.loads(block), "extracted block"
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc}"

    def verify(
        self,
        response: str,
        reference: Any = None,
        *,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> VerifierResult:
        obj, how = self.parse(response)
        if obj is None:
            return VerifierResult.fail(detail=f"not valid JSON ({how})")

        # No schema: validity is the whole signal.
        if reference is None:
            return VerifierResult(
                reward=self.baseline,
                passed=True,
                detail=f"valid JSON via {how}; no schema supplied",
                meta={"parsed": obj},
            )

        if not has("jsonschema"):
            return VerifierResult(
                reward=self.baseline,
                passed=True,
                detail=(
                    f"valid JSON via {how}; schema given but 'jsonschema' "
                    "unavailable (install trainall[verify]) -> baseline reward"
                ),
                meta={"parsed": obj},
            )

        jsonschema = require("jsonschema", feature="JSON schema validation")
        try:
            jsonschema.validate(instance=obj, schema=reference)
        except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
            return VerifierResult(
                reward=self.baseline,
                passed=False,
                detail=f"valid JSON via {how} but schema violation: {exc.message}",
                meta={"parsed": obj},
            )
        except jsonschema.SchemaError as exc:  # type: ignore[attr-defined]
            return VerifierResult.fail(detail=f"invalid schema supplied: {exc.message}")
        return VerifierResult.ok(
            detail=f"valid JSON via {how} and conforms to schema",
        )
