"""05 — Distillation + the self-play / rejection-sampling data flywheel.

Two complementary pieces of the frontier stack:

1. **Data flywheel** (pure-python, no model): grow a dataset of *verified* traces
   with ``RejectionSampler`` (best-of-N kept by a verifier), ``SyntheticDataEngine``
   (proposer -> solver -> verifier) and ``SelfPlayLoop`` (curriculum that raises
   difficulty as the solver's pass-rate climbs).  All accept plain callables.

2. **Knowledge distillation**: train a tiny student with the ``distill`` objective
   (KD; Hinton et al. 2015) against a frozen teacher's logits, supplied via a
   custom ``collate`` that writes ``batch.extra['teacher_logits']``.

Everything runs on CPU in seconds.  ``--real`` is a no-op pointer to swapping in
real model-backed solvers/teachers.
"""
from __future__ import annotations

import copy
import random
import sys

import torch

import trainall
from trainall.data import (
    Curriculum,
    InMemorySource,
    RejectionSampler,
    SelfPlayLoop,
    SyntheticDataEngine,
    TaskProposer,
)
from trainall.training import Trainer, TrainerConfig
from trainall.types import Batch

from _toy import sft_record, tiny_model


# --------------------------------------------------------------------------- #
# 1. Data flywheel — verifier-gated synthetic data, no model required.
# --------------------------------------------------------------------------- #
def data_flywheel() -> None:
    verifier = trainall.build("math", category="verifier")
    rng = random.Random(0)

    def propose():
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        return {"prompt": f"{a}+{b}=", "reference": str(a + b)}

    def solver(prompt: str) -> str:
        # Noisy solver: right ~60% of the time so rejection sampling has work.
        a, b = (int(x) for x in prompt.rstrip("=").split("+"))
        val = a + b if rng.random() < 0.6 else a + b + rng.randint(1, 3)
        return f"\\boxed{{{val}}}"

    # Rejection sampling: best-of-N completions kept if the verifier passes.
    rs = RejectionSampler(solver=solver, verifier=verifier, n=8, keep="best")
    kept = rs.generate([{"prompt": "3+4=", "reference": "7"}, {"prompt": "5+6=", "reference": "11"}])
    print(f"[05] rejection sampling kept {len(kept)} verified traces")

    # Synthetic data engine: proposer -> solver(k) -> verifier -> keep passing.
    engine = SyntheticDataEngine(proposer=propose, solver=solver, verifier=verifier, k=4)
    synth = engine.generate(6)
    print(f"[05] synthetic engine produced {len(synth)} verified samples")

    # Self-play with a curriculum that adapts difficulty to the pass-rate.
    loop = SelfPlayLoop(
        proposer=TaskProposer(lambda: propose()),
        solver=solver,
        verifier=verifier,
        curriculum=Curriculum(difficulty=0.2),
        rounds=2,
        tasks_per_round=4,
    )
    sp = loop.generate()
    print(f"[05] self-play loop generated {len(sp)} samples over its rounds")


# --------------------------------------------------------------------------- #
# 2. Knowledge distillation — student learns the teacher's soft logits.
# --------------------------------------------------------------------------- #
def make_distill_collate(teacher):
    def collate(items):
        ids = [it["input_ids"] for it in items]
        m = max(len(x) for x in ids)
        input_ids = torch.tensor([x + [0] * (m - len(x)) for x in ids], dtype=torch.long)
        labels = torch.tensor(
            [it["labels"] + [-100] * (m - len(it["labels"])) for it in items], dtype=torch.long
        )
        attn = torch.tensor([[1] * len(x) + [0] * (m - len(x)) for x in ids], dtype=torch.long)
        with torch.no_grad():
            teacher_logits = teacher(input_ids=input_ids, attention_mask=attn).logits
        batch = Batch.of(input_ids=input_ids, attention_mask=attn, labels=labels)
        batch.extra["teacher_logits"] = teacher_logits
        return batch

    return collate


def distillation() -> None:
    trainall.seed_everything(0)
    data = InMemorySource(
        [sft_record(p, r) for p, r in [("hi", " there"), ("2+2=", " 4"), ("sky is", " blue")] * 3]
    )
    teacher = tiny_model(n_layers=2).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = tiny_model(n_layers=1)

    objective = trainall.build("distill", category="objective", temperature=2.0, alpha=0.5)
    trainer = Trainer(
        model=student,
        objective=objective,
        data=data,
        collate=make_distill_collate(teacher),
        config=TrainerConfig(lr=5e-3, batch_size=3, max_steps=6, device="cpu", log_every=2),
    )
    trainer.train()
    print(f"[05] distillation done: {trainer.global_step} KD steps (student <- teacher)")


def run_tiny() -> None:
    data_flywheel()
    distillation()


def run_real() -> None:  # pragma: no cover - opt-in
    print("[05] --real: swap callable solver/teacher for model-backed ones; the same")
    print("     RejectionSampler / SyntheticDataEngine / distill objective apply.")


if __name__ == "__main__":
    run_real() if "--real" in sys.argv else run_tiny()
