"""03 — RL with Verifiable Rewards via GRPO.

The RLVR loop: for each prompt, sample a *group* of candidate answers, score
each one with a deterministic ``Verifier`` (here the ``math`` verifier), turn the
rewards into group-normalised advantages (GRPO; Shao et al. 2024), collate into a
policy-gradient ``Batch`` and take a GRPO step.

This wires together four public pieces: ``rl.Rollout`` + ``rl.RolloutConfig``,
a ``verifier`` Reward, ``rl.compute_group_advantages`` and the ``grpo`` objective
run through the ``Trainer``.  A plain ``str -> str`` callable stands in for the
policy so the whole thing runs on CPU in seconds.

``--real`` shows where a torch policy + tokenizer would plug in.
"""
from __future__ import annotations

import random
import sys

import torch

import trainall
from trainall.rl import Rollout, RolloutConfig, compute_group_advantages
from trainall.training import Trainer, TrainerConfig
from trainall.types import Batch

from _toy import encode, tiny_model

PROMPTS = ["2+2", "10-3", "5*2"]
ANSWERS = {"2+2": "4", "10-3": "7", "5*2": "10"}


def toy_policy(prompt: str) -> str:
    """A noisy policy: usually right, sometimes wrong (so groups have variance)."""
    correct = ANSWERS.get(prompt.strip(), "0")
    if random.random() < 0.5:
        return f"The answer is \\boxed{{{correct}}}"
    return f"The answer is \\boxed{{{random.randint(0, 20)}}}"


def grpo_collate(trajs):
    """Collate scored+advantaged trajectories into a GRPO policy-gradient Batch."""
    rows, resp_masks, advs = [], [], []
    for t in trajs:
        p_ids, r_ids = encode(t.prompt), encode(t.response)
        rows.append(p_ids + r_ids)
        resp_masks.append([0] * len(p_ids) + [1] * len(r_ids))
        advs.append(float(t.advantage if t.advantage is not None else 0.0))
    m = max(len(r) for r in rows)
    pad = lambda seqs, f: torch.tensor([s + [f] * (m - len(s)) for s in seqs], dtype=torch.long)
    return Batch.of(
        input_ids=pad(rows, 0),
        attention_mask=pad([[1] * len(r) for r in rows], 0),
        response_mask=pad(resp_masks, 0),
        advantages=torch.tensor(advs, dtype=torch.float32),
    )


def run_tiny() -> None:
    trainall.seed_everything(0)
    random.seed(0)

    rollout = Rollout(policy=toy_policy, config=RolloutConfig(group_size=6))
    verifier = trainall.build("math", category="verifier")
    reward = trainall.build("verifier", category="reward", verifier=verifier)

    # One RLVR iteration: sample groups, score, normalise advantages.
    trajs = rollout.group_sample(PROMPTS)
    scores = reward.score([_with_ref(t) for t in trajs])
    for t, s in zip(trajs, scores):
        t.reward = s
    compute_group_advantages(trajs)
    print(f"[03_rlvr_grpo] mean reward = {sum(scores) / len(scores):.3f}")

    model = tiny_model()
    objective = trainall.build("grpo", category="objective", clip_range=0.2)
    trainer = Trainer(
        model=model,
        objective=objective,
        data=[grpo_collate(trajs)],  # one pre-collated GRPO batch
        config=TrainerConfig(lr=1e-2, max_steps=4, device="cpu", log_every=1),
    )
    trainer.train()
    print(f"[03_rlvr_grpo] done: {trainer.global_step} GRPO steps")


def _with_ref(traj):
    """Attach the ground-truth answer where the verifier reward looks for it."""
    traj.meta["reference"] = ANSWERS.get(traj.prompt.strip(), "0")
    return traj


def run_real() -> None:  # pragma: no cover - opt-in
    print("[03_rlvr_grpo] --real: pass Rollout(policy=hf_model, tokenizer=tok) to sample")
    print("              real token ids + logprobs, then reuse the same GRPO collate.")


if __name__ == "__main__":
    run_real() if "--real" in sys.argv else run_tiny()
