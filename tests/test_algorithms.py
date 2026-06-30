"""Algorithms: LoRA adapter-only trainable + merge equivalence, FullFinetune."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from trainall.algorithms import FullFinetune, LoRA, LoRAConfig, LoRALinear, merge_lora  # noqa: E402


class _Tiny(nn.Module):
    """A small module with a ``q_proj`` linear LoRA should target."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.other = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.other(self.q_proj(x))


def test_lora_only_adapter_trainable():
    model = _Tiny()
    algo = LoRA(LoRAConfig(r=4, alpha=8, target_modules=["q_proj"]))
    model = algo.prepare_model(model)

    # q_proj is now a LoRALinear; only its A/B are trainable.
    assert isinstance(model.q_proj, LoRALinear)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all("lora_" in n for n in trainable)
    assert any("lora_A" in n for n in trainable)
    # Base + untargeted layers are frozen.
    assert not model.q_proj.base.weight.requires_grad
    assert not model.other.weight.requires_grad


def test_lora_starts_as_noop():
    """B is zero-initialised so the adapted layer == base at init."""
    model = _Tiny()
    x = torch.randn(2, 8)
    before = model(x)
    algo = LoRA(LoRAConfig(r=4, target_modules=["q_proj"]))
    model = algo.prepare_model(model)
    after = model(x)
    assert torch.allclose(before, after, atol=1e-6)


def test_lora_merge_equivalence():
    model = _Tiny()
    algo = LoRA(LoRAConfig(r=4, alpha=8, target_modules=["q_proj"]))
    model = algo.prepare_model(model)

    # Perturb the adapter so it is no longer a no-op.
    with torch.no_grad():
        model.q_proj.lora_B.add_(torch.randn_like(model.q_proj.lora_B))

    x = torch.randn(3, 8)
    adapted = model(x)
    merge_lora(model)
    assert isinstance(model.q_proj, nn.Linear) and not isinstance(model.q_proj, LoRALinear)
    merged = model(x)
    assert torch.allclose(adapted, merged, atol=1e-5)


def test_full_finetune_passthrough():
    model = _Tiny()
    algo = FullFinetune()
    out = algo.prepare_model(model)
    assert out is model
    assert all(p.requires_grad for p in model.parameters())
    assert len(list(algo.trainable_parameters(model))) == len(list(model.parameters()))


def test_lora_on_decoderlm(tiny_model):
    """LoRA targets the attention projections inside a real DecoderLM."""
    algo = LoRA(LoRAConfig(r=2, target_modules=["q_proj", "v_proj"]))
    model = algo.prepare_model(tiny_model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable
    assert all(
        ("lora_" in n)
        for n, p in model.named_parameters()
        if p.requires_grad
    )
    ids = torch.randint(0, model.config.vocab_size, (2, 5))
    out = model(input_ids=ids)
    out.logits.float().mean().backward()
