"""01 — Supervised fine-tuning (SFT).

Trains a tiny ``DecoderLM`` to follow instructions with the ``sft`` objective:
cross-entropy over response tokens only (the prompt is masked to ``-100``).
Demonstrates the core ``Trainer(model, objective, data, config)`` loop fed by an
``InMemorySource`` of pre-tokenised records.

Tiny mode (default) runs end-to-end on CPU in a couple of seconds.
Pass ``--real`` to fine-tune a real HF checkpoint instead (requires network +
``transformers``; off by default).
"""
from __future__ import annotations

import sys

import trainall
from trainall.data import InMemorySource
from trainall.training import Trainer, TrainerConfig

from _toy import sft_record, tiny_model


def run_tiny() -> None:
    trainall.seed_everything(0)

    # Toy instruction-following data, tokenised to (input_ids, labels).
    pairs = [
        ("Q: capital of France?\nA:", " Paris"),
        ("Q: 2+2?\nA:", " 4"),
        ("Q: color of the sky?\nA:", " blue"),
        ("Q: opposite of hot?\nA:", " cold"),
    ] * 4
    data = InMemorySource([sft_record(p, r) for p, r in pairs])

    model = tiny_model()
    objective = trainall.build("sft", category="objective", label_smoothing=0.0)

    trainer = Trainer(
        model=model,
        objective=objective,
        data=data,
        config=TrainerConfig(lr=1e-2, batch_size=4, max_steps=10, device="cpu", log_every=2),
    )
    trained = trainer.train()
    print(f"[01_sft] done: {trainer.global_step} steps on {type(trained).__name__}")


def run_real() -> None:  # pragma: no cover - opt-in, needs network + transformers
    from trainall.config import (
        DataConfig,
        ModelConfig,
        ObjectiveConfig,
        OptimConfig,
        RunConfig,
        TrainConfig,
    )

    cfg = RunConfig(
        model=ModelConfig(pretrained="sshleifer/tiny-gpt2"),
        data=DataConfig(source="jsonl", path="data/sft.jsonl"),
        objective=ObjectiveConfig(name="sft"),
        optim=OptimConfig(lr=2e-5),
        train=TrainConfig(batch_size=4, max_steps=50, bf16=False),
    )
    Trainer.from_config(cfg).train()
    print("[01_sft] real run complete")


if __name__ == "__main__":
    run_real() if "--real" in sys.argv else run_tiny()
