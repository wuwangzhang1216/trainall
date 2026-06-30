"""06 — Pre-training a DecoderLM from scratch.

Builds a fresh tiny ``DecoderLM`` (random init) and trains the next-token
language-modelling objective (``pretrain`` / causal-LM) over raw text packed into
fixed-length sequences with ``pack_sequences`` — the standard pre-training data
path.  Also shows ``cpt`` (continued pre-training) is the same objective family.

Tiny mode (default) trains a few thousand-parameter model on CPU in seconds.
``--real`` sketches scaling the ``ArchConfig`` up (still local, no download).
"""
from __future__ import annotations

import sys

import trainall
from trainall.data import InMemorySource, pack_sequences
from trainall.training import Trainer, TrainerConfig

from _toy import MAX_SEQ_LEN, encode, tiny_model

CORPUS = [
    "the quick brown fox jumps over the lazy dog. ",
    "language models predict the next token given the past. ",
    "attention is all you need for sequence modelling. ",
    "rotary embeddings encode relative position in the keys and queries. ",
]


def run_tiny() -> None:
    trainall.seed_everything(0)

    # Tokenise the corpus and greedily pack into fixed-length training windows.
    token_lists = [encode(doc * 4, max_len=10_000) for doc in CORPUS]
    packed = pack_sequences(token_lists, max_len=MAX_SEQ_LEN, pad_id=0)
    data = InMemorySource([{"input_ids": ids, "labels": list(ids)} for ids in packed])
    print(f"[06] packed {len(token_lists)} docs into {len(packed)} sequences of len<= {MAX_SEQ_LEN}")

    model = tiny_model(n_layers=2)  # fresh random init
    objective = trainall.build("pretrain", category="objective")
    trainer = Trainer(
        model=model,
        objective=objective,
        data=data,
        config=TrainerConfig(lr=1e-2, batch_size=2, max_steps=12, device="cpu", log_every=3),
    )
    trainer.train()
    print(f"[06] pretrain done: {trainer.global_step} steps from scratch")


def run_real() -> None:  # pragma: no cover - opt-in
    from trainall.models import ArchConfig, DecoderLM

    big = ArchConfig(vocab_size=32000, dim=512, n_layers=8, n_heads=8, n_kv_heads=2, max_seq_len=2048)
    model = DecoderLM.from_config(big)
    print(f"[06] --real: built a larger DecoderLM with {sum(p.numel() for p in model.parameters()):,} params")
    print("     Point a JsonlSource/HFDatasetSource of text at the same pretrain objective.")


if __name__ == "__main__":
    run_real() if "--real" in sys.argv else run_tiny()
