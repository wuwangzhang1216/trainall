"""The synthetic-data flywheel: propose -> solve -> verify -> keep.

A :class:`SyntheticDataEngine` manufactures supervised data without human
labels by chaining three callables (so it is fully testable without a model):

* **proposer** ``() -> task`` invents a problem (a prompt, optionally a
  reference answer).
* **solver** ``(prompt) -> [responses]`` produces ``k`` candidate solutions.
* **verifier** ``(response, reference, prompt=...) -> VerifierResult`` filters
  the candidates, keeping only those that pass.

This is the self-improvement / rejection-distillation loop popularised by
STaR (Zelikman et al. 2022) and RFT: a model bootstraps its own training set
from verifiable signal.  Passing traces become :class:`Sample` objects tagged
with a difficulty derived from the per-task pass-rate.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from ..base import Verifier
from ..types import Sample, VerifierResult

__all__ = ["SyntheticDataEngine"]


def _normalise_task(task: Any) -> Tuple[str, Optional[Any], dict]:
    """Coerce a proposer output into ``(prompt, reference, meta)``."""
    if isinstance(task, Sample):
        return task.prompt or task.text or "", task.reference, dict(task.meta)
    if isinstance(task, dict):
        return (
            str(task.get("prompt", task.get("task", ""))),
            task.get("reference"),
            dict(task.get("meta", {})),
        )
    if isinstance(task, (tuple, list)):
        prompt = task[0]
        reference = task[1] if len(task) > 1 else None
        return str(prompt), reference, {}
    return str(task), None, {}


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


def _difficulty(pass_rate: float) -> str:
    """Tag difficulty from the fraction of candidates that passed."""
    if pass_rate <= 0.0:
        return "unsolved"
    if pass_rate < 0.34:
        return "hard"
    if pass_rate < 0.84:
        return "medium"
    return "easy"


class SyntheticDataEngine:
    """Generate verified SFT samples by running the propose/solve/verify loop.

    Parameters
    ----------
    proposer:
        Callable returning a task (prompt, ``(prompt, reference)``, dict, or
        :class:`Sample`).
    solver:
        Callable ``prompt -> response`` or ``prompt -> [responses]``; called to
        obtain up to ``k`` candidates per task.
    verifier:
        A :class:`Verifier` or plain callable filtering candidates.
    k:
        Number of candidate solutions to sample per task.
    dedup:
        Drop duplicate ``(prompt, response)`` pairs across the whole run.
    keep_per_task:
        ``"first"`` keeps the first passing candidate; ``"all"`` keeps every
        passing candidate.
    """

    def __init__(
        self,
        proposer: Callable[[], Any],
        solver: Callable[[str], Any],
        verifier: Any,
        k: int = 4,
        dedup: bool = True,
        keep_per_task: str = "first",
    ) -> None:
        self.proposer = proposer
        self.solver = solver
        self.verifier = verifier
        self.k = max(1, int(k))
        self.dedup = dedup
        if keep_per_task not in {"first", "all"}:
            raise ValueError("keep_per_task must be 'first' or 'all'")
        self.keep_per_task = keep_per_task

    def _candidates(self, prompt: str) -> List[str]:
        out = self.solver(prompt)
        if isinstance(out, str):
            cands = [out]
        elif isinstance(out, (list, tuple)):
            cands = [str(c) for c in out]
        else:
            cands = [str(out)]
        # Top up to k by re-sampling when the solver returns fewer.
        while len(cands) < self.k:
            extra = self.solver(prompt)
            if isinstance(extra, (list, tuple)):
                cands.extend(str(c) for c in extra)
            else:
                cands.append(str(extra))
        return cands[: self.k]

    def generate(self, n: int) -> List[Sample]:
        """Run the flywheel for ``n`` proposed tasks; return kept samples."""
        kept: List[Sample] = []
        seen: set[Tuple[str, str]] = set()
        for _ in range(int(n)):
            prompt, reference, meta = _normalise_task(self.proposer())
            cands = self._candidates(prompt)
            passing: List[str] = []
            n_pass = 0
            for cand in cands:
                res = _run_verifier(self.verifier, cand, reference, prompt)
                if res.passed:
                    n_pass += 1
                    passing.append(cand)
            pass_rate = n_pass / max(1, len(cands))
            diff = _difficulty(pass_rate)
            chosen = passing if self.keep_per_task == "all" else passing[:1]
            for resp in chosen:
                if self.dedup:
                    key = (prompt, resp)
                    if key in seen:
                        continue
                    seen.add(key)
                kept.append(
                    Sample(
                        prompt=prompt,
                        response=resp,
                        reference=reference,
                        meta={**meta, "difficulty": diff, "pass_rate": pass_rate, "synthetic": True},
                    )
                )
        return kept
