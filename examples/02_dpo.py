"""02 — Direct Preference Optimisation (DPO).

Aligns a tiny ``DecoderLM`` on ``chosen`` vs ``rejected`` pairs with the ``dpo``
objective (Rafailov et al. 2023).  DPO needs a *preference* batch
(``chosen_*`` / ``rejected_*``) and a frozen *reference* model, so this example
shows how to supply a custom ``collate`` that builds that ``Batch`` and stashes
the reference under ``batch.extra["ref_model"]``.

Tiny mode (default) runs on CPU in seconds.  ``--real`` swaps in a larger HF
model + JSONL preference data (opt-in).
"""
from __future__ import annotations

import copy
import sys

import torch

import trainall
from trainall.data import InMemorySource
from trainall.training import Trainer, TrainerConfig
from trainall.types import Batch, PreferenceSample

from _toy import encode, tiny_model


def make_preference_collate(ref_model):
    """Return a collate that turns ``PreferenceSample``s into a DPO ``Batch``."""

    def _pad(seqs, fill):
        m = max(len(s) for s in seqs)
        return torch.tensor([s + [fill] * (m - len(s)) for s in seqs], dtype=torch.long)

    def collate(items):
        c_ids, c_lab, r_ids, r_lab = [], [], [], []
        for s in items:
            p = encode(s.prompt)
            cho, rej = encode(s.chosen), encode(s.rejected)
            c_ids.append(p + cho)
            c_lab.append([-100] * len(p) + cho)
            r_ids.append(p + rej)
            r_lab.append([-100] * len(p) + rej)
        batch = Batch.of(
            chosen_input_ids=_pad(c_ids, 0),
            chosen_attention_mask=_pad([[1] * len(x) for x in c_ids], 0),
            chosen_labels=_pad(c_lab, -100),
            rejected_input_ids=_pad(r_ids, 0),
            rejected_attention_mask=_pad([[1] * len(x) for x in r_ids], 0),
            rejected_labels=_pad(r_lab, -100),
        )
        batch.extra["ref_model"] = ref_model
        return batch

    return collate


def run_tiny() -> None:
    trainall.seed_everything(0)

    prefs = [
        PreferenceSample(prompt="Be helpful: ", chosen="here is a clear answer", rejected="no."),
        PreferenceSample(prompt="Greet: ", chosen="hello, how can I help?", rejected="go away"),
        PreferenceSample(prompt="Explain: ", chosen="step by step reasoning", rejected="idk"),
    ] * 3
    data = InMemorySource(prefs)

    model = tiny_model()
    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    objective = trainall.build("dpo", category="objective", beta=0.1, loss_type="sigmoid")

    trainer = Trainer(
        model=model,
        objective=objective,
        data=data,
        collate=make_preference_collate(ref_model),
        config=TrainerConfig(lr=5e-3, batch_size=3, max_steps=8, device="cpu", log_every=2),
    )
    trainer.train()
    print(f"[02_dpo] done: {trainer.global_step} DPO steps")


def run_real() -> None:  # pragma: no cover - opt-in
    print("[02_dpo] --real: point a JSONL of {prompt,chosen,rejected} pairs at a HF model")
    print("         and reuse make_preference_collate with a frozen copy as the reference.")


if __name__ == "__main__":
    run_real() if "--real" in sys.argv else run_tiny()
