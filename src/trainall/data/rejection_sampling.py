"""Rejection sampling / best-of-N distillation data path.

Given a fixed set of prompts (with optional references), sample ``n`` candidate
responses per prompt, score them with a verifier, and keep the verifier-passing
ones as SFT :class:`Sample` objects.  This is the rejection-sampling fine-tuning
(RFT) / distillation recipe: turn an expensive sampler + a cheap verifier into
clean supervised data, learning only from successes.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

from ..base import Verifier
from ..types import Sample, VerifierResult

__all__ = ["RejectionSampler"]


def _coerce_prompt(item: Any) -> Tuple[str, Optional[Any], dict]:
    if isinstance(item, Sample):
        return item.prompt or item.text or "", item.reference, dict(item.meta)
    if isinstance(item, dict):
        return str(item.get("prompt", "")), item.get("reference"), dict(item.get("meta", {}))
    if isinstance(item, (tuple, list)):
        ref = item[1] if len(item) > 1 else None
        return str(item[0]), ref, {}
    return str(item), None, {}


def _run_verifier(verifier: Any, response: str, reference: Any, prompt: str) -> VerifierResult:
    res = (
        verifier.verify(response, reference, prompt=prompt)
        if isinstance(verifier, Verifier)
        else verifier(response, reference)
    )
    if isinstance(res, VerifierResult):
        return res
    if isinstance(res, bool):
        return VerifierResult.ok() if res else VerifierResult.fail()
    val = float(res)
    return VerifierResult(reward=val, passed=val >= 0.5)


class RejectionSampler:
    """Best-of-N over a solver, filtered by a verifier, into SFT samples.

    Parameters
    ----------
    solver:
        Callable ``prompt -> response`` (called repeatedly) or
        ``prompt -> [responses]``.
    verifier:
        A :class:`Verifier` or plain callable scoring each candidate.
    n:
        Number of candidates to draw per prompt.
    keep:
        ``"best"`` keeps the single highest-reward passing candidate;
        ``"all"`` keeps every passing candidate;
        ``"first"`` keeps the first passing candidate.
    """

    def __init__(
        self,
        solver: Callable[[str], Any],
        verifier: Any,
        n: int = 8,
        keep: str = "best",
    ) -> None:
        self.solver = solver
        self.verifier = verifier
        self.n = max(1, int(n))
        if keep not in {"best", "all", "first"}:
            raise ValueError("keep must be 'best', 'all' or 'first'")
        self.keep = keep

    def _candidates(self, prompt: str) -> List[str]:
        out = self.solver(prompt)
        cands: List[str]
        if isinstance(out, str):
            cands = [out]
        elif isinstance(out, (list, tuple)):
            cands = [str(c) for c in out]
        else:
            cands = [str(out)]
        while len(cands) < self.n:
            extra = self.solver(prompt)
            if isinstance(extra, (list, tuple)):
                cands.extend(str(c) for c in extra)
            else:
                cands.append(str(extra))
        return cands[: self.n]

    def sample_one(self, prompt: str, reference: Any = None, meta: Optional[dict] = None) -> List[Sample]:
        """Run best-of-N for a single prompt; return kept SFT samples."""
        meta = dict(meta or {})
        cands = self._candidates(prompt)
        scored: List[Tuple[float, str]] = []
        for cand in cands:
            res = _run_verifier(self.verifier, cand, reference, prompt)
            if res.passed:
                scored.append((res.reward, cand))

        if not scored:
            return []

        if self.keep == "first":
            picks = [scored[0]]
        elif self.keep == "best":
            picks = [max(scored, key=lambda t: t[0])]
        else:  # all
            picks = scored

        pass_rate = len(scored) / max(1, len(cands))
        return [
            Sample(
                prompt=prompt,
                response=resp,
                reference=reference,
                meta={**meta, "reward": reward, "pass_rate": pass_rate, "rejection_sampled": True},
            )
            for reward, resp in picks
        ]

    def run(self, prompts: Sequence[Any]) -> List[Sample]:
        """Apply best-of-N over an iterable of prompts / samples / dicts."""
        kept: List[Sample] = []
        for item in prompts:
            prompt, reference, meta = _coerce_prompt(item)
            kept.extend(self.sample_one(prompt, reference, meta))
        return kept

    # Alias mirroring the engine API.
    def generate(self, prompts: Sequence[Any]) -> List[Sample]:
        return self.run(prompts)
