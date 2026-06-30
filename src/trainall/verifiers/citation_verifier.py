"""Citation grounding verifier.

Guards against fabricated citations: every quoted span and every ``[n]`` /
``[id]`` reference in the response must actually appear in / point at a real
source.  The reward is the fraction of citations that are grounded — a direct
attribution-faithfulness signal (cf. retrieval-augmented faithfulness metrics,
e.g. Gao et al. 2023, "Enabling Large Language Models to Generate Text with
Citations").
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..base import Verifier
from ..registry import register
from ..types import VerifierResult

__all__ = ["CitationVerifier"]

# Quoted spans: straight or curly double quotes.
_QUOTE_RE = re.compile(r'"([^"]{3,})"|“([^”]{3,})”')
# Bracketed citations: [3], [src1], [a, b] -> split inner on commas.
_BRACKET_RE = re.compile(r"\[([^\]\n]{1,40})\]")


@register("citation", category="verifier")
class CitationVerifier(Verifier):
    """Verify quoted spans and bracketed references against provided sources.

    Parameters
    ----------
    min_quote_len:
        Minimum length for a quoted span to be treated as a citation.
    require_any:
        If True and the response contains no citations at all, the result fails
        (a response that cites nothing is ungrounded).  Default False -> such a
        response passes vacuously with reward 1.0.
    """

    name = "citation"

    def __init__(self, min_quote_len: int = 3, require_any: bool = False) -> None:
        self.min_quote_len = int(min_quote_len)
        self.require_any = bool(require_any)

    @staticmethod
    def _sources(reference: Any) -> Dict[str, str]:
        """Normalise ``reference`` to ``{id: text}``.

        Lists become 1-indexed string ids matching ``[n]`` conventions.
        """
        if reference is None:
            return {}
        if isinstance(reference, dict):
            return {str(k): str(v) for k, v in reference.items()}
        if isinstance(reference, (list, tuple)):
            return {str(i + 1): str(v) for i, v in enumerate(reference)}
        return {"1": str(reference)}

    def verify(
        self,
        response: str,
        reference: Any = None,
        *,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> VerifierResult:
        text = response or ""
        sources = self._sources(reference)
        source_blob = "\n".join(sources.values()).lower()

        grounded = 0
        total = 0
        notes: List[str] = []

        # 1) Quoted spans must appear verbatim (case-insensitive) in a source.
        for m in _QUOTE_RE.finditer(text):
            span = (m.group(1) or m.group(2) or "").strip()
            if len(span) < self.min_quote_len:
                continue
            total += 1
            if span.lower() in source_blob:
                grounded += 1
            else:
                notes.append(f"fabricated quote: {span[:40]!r}")

        # 2) Bracketed ids must refer to a real source id.
        for m in _BRACKET_RE.finditer(text):
            for raw in m.group(1).split(","):
                cid = raw.strip()
                if not cid:
                    continue
                total += 1
                if cid in sources:
                    grounded += 1
                else:
                    notes.append(f"dangling citation [{cid}]")

        if total == 0:
            if self.require_any:
                return VerifierResult.fail(detail="no citations found in response")
            return VerifierResult(
                reward=1.0,
                passed=True,
                detail="no citations to verify (passes vacuously)",
            )

        reward = grounded / total
        passed = grounded == total
        detail = f"{grounded}/{total} citations grounded"
        if notes:
            detail += "; " + "; ".join(notes)
        return VerifierResult(reward=reward, passed=passed, detail=detail)
