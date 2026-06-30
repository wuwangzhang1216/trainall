"""07 — Parameter-efficient fine-tuning with LoRA / QLoRA.

Same SFT loop as example 01, but the ``Algorithm`` axis swaps full fine-tuning
for **LoRA** (Hu et al. 2021): the base ``DecoderLM`` weights are frozen and only
small low-rank adapters on the attention/MLP projections are trained.  We print
the trainable-parameter count to show the saving, then demonstrate
``QLoRA`` (Dettmers et al. 2023), which quantises the frozen base before adding
adapters (gracefully falling back to fp when ``bitsandbytes`` is absent on CPU).

Runs on CPU in seconds.  ``--real`` is a pointer to applying the same algorithm
to a real HF checkpoint.
"""
from __future__ import annotations

import sys

import trainall
from trainall.algorithms import LoRAConfig
from trainall.data import InMemorySource
from trainall.training import Trainer, TrainerConfig

from _toy import sft_record, tiny_model


def _trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _toy_data() -> InMemorySource:
    pairs = [("Q: capital of Japan?\nA:", " Tokyo"), ("Q: 3*3?\nA:", " 9")] * 6
    return InMemorySource([sft_record(p, r) for p, r in pairs])


def run_lora() -> None:
    trainall.seed_everything(0)
    model = tiny_model()
    total = sum(p.numel() for p in model.parameters())

    algorithm = trainall.build(
        "lora",
        category="algorithm",
        config=LoRAConfig(r=4, alpha=8, target_modules=["q_proj", "v_proj"]),
    )
    objective = trainall.build("sft", category="objective")
    trainer = Trainer(
        model=model,
        objective=objective,
        algorithm=algorithm,
        data=_toy_data(),
        config=TrainerConfig(lr=1e-2, batch_size=4, max_steps=8, device="cpu", log_every=4),
    )
    trained = trainer.train()
    print(f"[07] LoRA: trainable {_trainable(trained):,} / {total:,} base params")


def run_qlora() -> None:
    trainall.seed_everything(0)
    model = tiny_model()
    algorithm = trainall.build("qlora", category="algorithm")
    objective = trainall.build("sft", category="objective")
    trainer = Trainer(
        model=model,
        objective=objective,
        algorithm=algorithm,
        data=_toy_data(),
        config=TrainerConfig(lr=1e-2, batch_size=4, max_steps=6, device="cpu", log_every=3),
    )
    trained = trainer.train()
    print(f"[07] QLoRA: trainable {_trainable(trained):,} params (fp fallback on CPU is expected)")


def run_tiny() -> None:
    run_lora()
    run_qlora()


def run_real() -> None:  # pragma: no cover - opt-in
    print("[07] --real: build('lora'/'qlora') prepare_model() works the same on a HF model;")
    print("     pass use_peft=True to delegate to the peft library when installed.")


if __name__ == "__main__":
    run_real() if "--real" in sys.argv else run_tiny()
