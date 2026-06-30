"""04 — Agentic RL (multi-step tool use, scored by outcome).

Drives a policy through a multi-step ``Environment`` with a real tool: the
``ExpressionEnv`` asks the agent to reach a target number using the
``CalculatorTool``, then submit ``answer: <value>``.  ``AgenticRunner`` records
the episode and flattens it into GRPO ``Trajectory`` objects, which we score with
group advantages and feed to a GRPO ``Trainer`` step — the same RLVR machinery as
example 03, but over multi-turn tool-use episodes.

The policy here is a scripted ``observation -> action`` callable (no model
needed), so it runs on CPU in seconds.  ``--real`` shows where a model-backed
policy plugs in.
"""
from __future__ import annotations

import re
import sys

import torch

import trainall
from trainall.rl import AgenticRunner, compute_group_advantages
from trainall.training import Trainer, TrainerConfig
from trainall.types import Batch, Sample

from _toy import encode, tiny_model

TASKS = [
    Sample(prompt="Compute 6 * 7.", reference=42.0),
    Sample(prompt="Compute 12 + 5.", reference=17.0),
    Sample(prompt="Compute 100 - 1.", reference=99.0),
]


def scripted_policy(observation):
    """Two-turn agent for ``ExpressionEnv``.

    The task observation contains ``"Compute <expr>."`` — on seeing it, call the
    calculator tool.  The tool's reply is a bare number, which we then submit as
    the final answer.
    """
    text = str(observation)
    m = re.search(r"Compute ([\d\s\+\-\*/\.\(\)]+)\.", text)
    if m:
        return f"calculator: {m.group(1).strip()}"
    # Otherwise the observation is the calculator's numeric result -> submit it.
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return f"answer: {nums[-1]}" if nums else "answer: 0"


def grpo_collate(trajs):
    rows, resp_masks, advs = [], [], []
    for t in trajs:
        p_ids, r_ids = encode(t.prompt), encode(t.response or " ")
        rows.append(p_ids + r_ids)
        resp_masks.append([0] * len(p_ids) + [1] * len(r_ids))
        advs.append(float(t.advantage or 0.0))
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

    env = trainall.build("expression_env", category="environment")
    runner = AgenticRunner(env=env, policy=scripted_policy, max_steps=4)

    # Each task repeated to form GRPO groups; one episode per copy.
    samples = [s for s in TASKS for _ in range(4)]
    trajs = runner.collect(samples)
    # Re-group by the underlying task so advantages normalise within a task.
    for gid in range(len(samples)):
        trajs[gid].group_id = gid % len(TASKS)

    compute_group_advantages(trajs)
    succ = sum(t.meta.get("success", False) for t in trajs) / len(trajs)
    print(f"[04_agentic_rl] episode success rate = {succ:.2f}")

    model = tiny_model()
    objective = trainall.build("grpo", category="objective")
    trainer = Trainer(
        model=model,
        objective=objective,
        data=[grpo_collate(trajs)],
        config=TrainerConfig(lr=1e-2, max_steps=3, device="cpu", log_every=1),
    )
    trainer.train()
    print(f"[04_agentic_rl] done: {trainer.global_step} GRPO steps over agentic episodes")


def run_real() -> None:  # pragma: no cover - opt-in
    print("[04_agentic_rl] --real: replace scripted_policy with a model wrapper that")
    print("                emits tool calls ('calculator: ...') and answers ('answer: ...').")


if __name__ == "__main__":
    run_real() if "--real" in sys.argv else run_tiny()
