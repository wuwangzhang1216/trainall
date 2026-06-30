"""Training: a tiny end-to-end SFT loop on CPU, loss finite, model returned."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import trainall  # noqa: E402
from trainall.data import InMemorySource  # noqa: E402
from trainall.training import Trainer, TrainerConfig  # noqa: E402


def _tokenised_items(n=6, seqlen=8, vocab=37):
    """Pre-tokenised records the default collate understands.

    Each dict carries ``input_ids`` and ``labels`` (prompt masked) as python
    lists, matching the ``default_collate`` contract.
    """
    items = []
    for i in range(n):
        ids = torch.randint(0, vocab, (seqlen,)).tolist()
        labels = list(ids)
        labels[:2] = [-100, -100]  # mask a 2-token prompt
        items.append({"input_ids": ids, "labels": labels})
    return items


def test_trainer_sft_end_to_end(tiny_model):
    data = InMemorySource(_tokenised_items())
    obj = trainall.build("sft", category="objective")
    cfg = TrainerConfig(lr=1e-3, batch_size=2, max_steps=3, device="cpu", bf16=False, log_every=1)

    captured = {}

    from trainall.training import Callback

    class _Spy(Callback):
        def on_step_end(self, step, metrics, trainer=None):
            captured["last"] = metrics

    trainer = Trainer(tiny_model, obj, data=data, config=cfg, callbacks=[_Spy()])
    trained = trainer.train()

    assert trained is not None
    assert trainer.global_step == 3
    assert "loss" in captured["last"]
    assert torch.isfinite(torch.tensor(captured["last"]["loss"]))


def test_trainer_default_algorithm_is_full(tiny_model):
    obj = trainall.build("sft", category="objective")
    trainer = Trainer(tiny_model, obj, data=InMemorySource(_tokenised_items(2)))
    assert trainer.algorithm.__class__.__name__ == "FullFinetune"


def test_trainer_grad_accum(tiny_model):
    data = InMemorySource(_tokenised_items(8))
    obj = trainall.build("sft", category="objective")
    cfg = TrainerConfig(lr=1e-3, batch_size=2, grad_accum=2, max_steps=2, device="cpu", bf16=False)
    trainer = Trainer(tiny_model, obj, data=data, config=cfg)
    trainer.train()
    assert trainer.global_step == 2


def test_trainer_from_config_builds_decoder_lm():
    cfg = trainall.RunConfig()
    cfg = cfg.merge(
        model=trainall.ModelConfig(options={"vocab_size": 37, "dim": 16, "n_layers": 2,
                                            "n_heads": 4, "n_kv_heads": 2, "ffn_dim": 32,
                                            "max_seq_len": 16}),
        data=trainall.DataConfig(source="memory", options={"items": []}),
    )
    trainer = Trainer.from_config(cfg)
    assert trainer.model.__class__.__name__ == "DecoderLM"
    assert trainer.objective.__class__.__name__ == "SFTObjective"
    assert trainer.algorithm.__class__.__name__ == "FullFinetune"
