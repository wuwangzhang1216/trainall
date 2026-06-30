"""Self-play data generation with an adaptive difficulty curriculum.

The self-play loop closes the generative flywheel into a *curriculum*: a
:class:`TaskProposer` invents tasks at a target difficulty, a solver attempts
them, a verifier judges, retained traces become training data, and a
:class:`Curriculum` watches the pass-rate to decide whether to raise or lower
difficulty for the next round.

The curriculum embodies three ideas from the open-ended-learning / self-play
literature (e.g. AlphaZero-style curricula, ADA / PAIRED):

* **Zone of proximal development** — push difficulty up only when the solver is
  comfortably succeeding, down when it is failing, keeping tasks learnable.
* **Diversity** — discourage the proposer from collapsing onto a few prompts.
* **Anti-collapse** — guard against the distribution narrowing over rounds.

Everything is plain-python and callable-friendly so it is testable without a
model.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

from ..base import Verifier
from ..types import Sample, VerifierResult

__all__ = ["TaskProposer", "Curriculum", "SelfPlayLoop"]


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


def _normalise_task(task: Any) -> Tuple[str, Optional[Any], dict]:
    if isinstance(task, Sample):
        return task.prompt or task.text or "", task.reference, dict(task.meta)
    if isinstance(task, dict):
        return (
            str(task.get("prompt", task.get("task", ""))),
            task.get("reference"),
            dict(task.get("meta", {})),
        )
    if isinstance(task, (tuple, list)):
        ref = task[1] if len(task) > 1 else None
        return str(task[0]), ref, {}
    return str(task), None, {}


class TaskProposer:
    """Proposes tasks at a requested difficulty level.

    Subclass and override :meth:`propose`, or pass a plain callable to the
    constructor.  A callable may accept zero args or a single ``difficulty``
    arg; both are supported.
    """

    def __init__(self, fn: Optional[Callable[..., Any]] = None) -> None:
        self._fn = fn

    def propose(self, difficulty: float) -> Any:
        """Return a task for the given ``difficulty`` in ``[0, 1]``."""
        if self._fn is None:
            raise NotImplementedError("override propose() or pass a callable")
        try:
            return self._fn(difficulty)
        except TypeError:
            return self._fn()

    def __call__(self, difficulty: float = 0.5) -> Any:
        return self.propose(difficulty)


class Curriculum:
    """Adapts difficulty from observed pass-rate; tracks diversity & collapse.

    Parameters
    ----------
    difficulty:
        Starting difficulty in ``[0, 1]``.
    step:
        Amount to move difficulty per round.
    target_low, target_high:
        Desired pass-rate band.  Above ``target_high`` -> harder; below
        ``target_low`` -> easier.
    min_diversity:
        Minimum fraction of *unique* prompts in a round before a
        ``collapse`` warning is recorded.
    """

    def __init__(
        self,
        difficulty: float = 0.2,
        step: float = 0.1,
        target_low: float = 0.4,
        target_high: float = 0.8,
        min_diversity: float = 0.5,
    ) -> None:
        self.difficulty = float(difficulty)
        self.step = float(step)
        self.target_low = float(target_low)
        self.target_high = float(target_high)
        self.min_diversity = float(min_diversity)
        self.history: List[dict] = []

    @staticmethod
    def _diversity(prompts: Sequence[str]) -> float:
        if not prompts:
            return 1.0
        return len(set(prompts)) / len(prompts)

    def update(self, pass_rate: float, prompts: Sequence[str]) -> float:
        """Record a round's stats and return the next difficulty.

        Raises difficulty when the solver is succeeding too easily, lowers it
        when struggling, and flags low prompt diversity (distribution collapse).
        """
        diversity = self._diversity(prompts)
        collapsed = diversity < self.min_diversity

        if pass_rate > self.target_high:
            self.difficulty = min(1.0, self.difficulty + self.step)
            decision = "harder"
        elif pass_rate < self.target_low:
            self.difficulty = max(0.0, self.difficulty - self.step)
            decision = "easier"
        else:
            decision = "hold"

        self.history.append(
            {
                "pass_rate": pass_rate,
                "diversity": diversity,
                "collapsed": collapsed,
                "decision": decision,
                "difficulty": self.difficulty,
            }
        )
        return self.difficulty


class SelfPlayLoop:
    """Run ``propose -> solve -> verify -> retain`` over several rounds.

    Each round the proposer is queried ``tasks_per_round`` times at the current
    curriculum difficulty, the solver produces ``k`` candidates per task, the
    verifier judges, and passing ``(prompt, response)`` traces are retained as
    :class:`Sample` objects.  The curriculum then adjusts difficulty.
    """

    def __init__(
        self,
        proposer: Any,
        solver: Callable[[str], Any],
        verifier: Any,
        curriculum: Optional[Curriculum] = None,
        rounds: int = 3,
        tasks_per_round: int = 8,
        k: int = 4,
        dedup: bool = True,
    ) -> None:
        self.proposer = proposer if isinstance(proposer, TaskProposer) else TaskProposer(proposer)
        self.solver = solver
        self.verifier = verifier
        self.curriculum = curriculum or Curriculum()
        self.rounds = max(1, int(rounds))
        self.tasks_per_round = max(1, int(tasks_per_round))
        self.k = max(1, int(k))
        self.dedup = dedup

    def _candidates(self, prompt: str) -> List[str]:
        out = self.solver(prompt)
        if isinstance(out, str):
            cands = [out]
        elif isinstance(out, (list, tuple)):
            cands = [str(c) for c in out]
        else:
            cands = [str(out)]
        while len(cands) < self.k:
            extra = self.solver(prompt)
            if isinstance(extra, (list, tuple)):
                cands.extend(str(c) for c in extra)
            else:
                cands.append(str(extra))
        return cands[: self.k]

    def run(self) -> List[Sample]:
        """Execute all rounds; return all retained samples."""
        kept: List[Sample] = []
        seen: set[Tuple[str, str]] = set()
        for rnd in range(self.rounds):
            difficulty = self.curriculum.difficulty
            round_prompts: List[str] = []
            n_pass_total = 0
            n_cand_total = 0
            for _ in range(self.tasks_per_round):
                prompt, reference, meta = _normalise_task(self.proposer.propose(difficulty))
                round_prompts.append(prompt)
                cands = self._candidates(prompt)
                n_cand_total += len(cands)
                for cand in cands:
                    res = _run_verifier(self.verifier, cand, reference, prompt)
                    if res.passed:
                        n_pass_total += 1
                        if self.dedup:
                            key = (prompt, cand)
                            if key in seen:
                                continue
                            seen.add(key)
                        kept.append(
                            Sample(
                                prompt=prompt,
                                response=cand,
                                reference=reference,
                                meta={
                                    **meta,
                                    "round": rnd,
                                    "difficulty": difficulty,
                                    "reward": res.reward,
                                    "self_play": True,
                                },
                            )
                        )
            pass_rate = n_pass_total / max(1, n_cand_total)
            self.curriculum.update(pass_rate, round_prompts)
        return kept

    def generate(self) -> List[Sample]:
        return self.run()
