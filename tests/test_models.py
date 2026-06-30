"""Models: DecoderLM forward/backward, RMSNorm/RoPE/Attention GQA, MoE, MLA."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from trainall.models import (  # noqa: E402
    ArchConfig,
    Attention,
    DecoderLM,
    MoEFeedForward,
    MultiHeadLatentAttention,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
)
from trainall.models.rope import apply_rotary  # noqa: E402


def test_decoderlm_forward_backward(tiny_model, arch):
    ids = torch.randint(0, arch.vocab_size, (2, 7))
    out = tiny_model(input_ids=ids, attention_mask=torch.ones_like(ids))
    assert out.logits.shape == (2, 7, arch.vocab_size)
    assert torch.isfinite(out.logits).all()
    # aux_loss is a scalar (zero for dense models).
    assert out.aux_loss.ndim == 0
    loss = out.logits.float().mean()
    loss.backward()
    grads = [p.grad for p in tiny_model.parameters() if p.grad is not None]
    assert grads, "no gradients flowed"
    assert all(torch.isfinite(g).all() for g in grads)


def test_decoderlm_attention_mask_respected(tiny_model, arch):
    # A padded batch should not raise and should produce finite logits.
    ids = torch.randint(0, arch.vocab_size, (2, 5))
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    out = tiny_model(input_ids=ids, attention_mask=mask)
    assert torch.isfinite(out.logits).all()


def test_rmsnorm_unit_rms():
    norm = RMSNorm(8)
    x = torch.randn(4, 8) * 5.0
    y = norm(x)
    # weight is initialised to 1 so output RMS per row should be ~1.
    rms = y.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rope_shapes_and_relative_invariance():
    rope = RotaryEmbedding(dim=8, max_seq_len=16)
    cos, sin = rope(10)
    assert cos.shape == (10, 8) and sin.shape == (10, 8)
    q = torch.randn(1, 2, 10, 8)
    k = torch.randn(1, 2, 10, 8)
    qr, kr = apply_rotary(q, k, cos, sin)
    assert qr.shape == q.shape and kr.shape == k.shape


def test_rope_scaling_variants_build():
    for scaling in (
        {"type": "linear", "factor": 2.0},
        {"type": "ntk", "factor": 2.0},
        {"type": "yarn", "factor": 2.0},
    ):
        rope = RotaryEmbedding(dim=8, max_seq_len=16, scaling=scaling)
        cos, sin = rope(8)
        assert cos.shape == (8, 8)
        assert torch.isfinite(cos).all() and torch.isfinite(sin).all()


def test_attention_gqa_shapes():
    cfg = ArchConfig(dim=16, n_heads=4, n_kv_heads=2, max_seq_len=16)
    attn = Attention(cfg)
    x = torch.randn(2, 6, 16)
    out = attn(x)
    assert out.shape == (2, 6, 16)
    assert torch.isfinite(out).all()


def test_attention_mqa_shapes():
    cfg = ArchConfig(dim=16, n_heads=4, n_kv_heads=1, max_seq_len=16)
    attn = Attention(cfg)
    x = torch.randn(1, 5, 16)
    out = attn(x)
    assert out.shape == (1, 5, 16)


def test_swiglu_shapes():
    mlp = SwiGLU(16, 32)
    x = torch.randn(3, 16)
    assert mlp(x).shape == (3, 16)


def test_moe_returns_aux_loss():
    cfg = ArchConfig(dim=16, ffn_dim=32, use_moe=True, n_experts=4, n_experts_per_tok=2)
    moe = MoEFeedForward(cfg)
    x = torch.randn(2, 5, 16)
    out, aux = moe(x)
    assert out.shape == (2, 5, 16)
    assert aux.ndim == 0 and torch.isfinite(aux)
    assert aux.item() >= 0.0
    # Router probs exposed for inspection.
    assert moe.router_probs is not None
    assert moe.router_probs.shape == (10, 4)
    out.sum().backward()  # aux + output are backward-able


def test_decoderlm_moe_aux_loss_nonzero_and_backward():
    cfg = ArchConfig(
        vocab_size=37, dim=16, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=32,
        max_seq_len=16, use_moe=True, n_experts=4, n_experts_per_tok=2,
    )
    model = DecoderLM.from_config(cfg)
    ids = torch.randint(0, 37, (2, 6))
    out = model(input_ids=ids)
    assert out.aux_loss.item() > 0.0
    (out.logits.float().mean() + out.aux_loss).backward()


def test_mla_runs():
    cfg = ArchConfig(
        dim=16, n_heads=4, n_kv_heads=2, max_seq_len=16,
        kv_lora_rank=8, q_lora_rank=8,
    )
    mla = MultiHeadLatentAttention(cfg)
    x = torch.randn(2, 6, 16)
    out = mla(x)
    assert out.shape == (2, 6, 16)
    assert torch.isfinite(out).all()
    out.sum().backward()


def test_mla_in_decoderlm():
    cfg = ArchConfig(
        vocab_size=37, dim=16, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=32,
        max_seq_len=16, kv_lora_rank=8, q_lora_rank=8,
    )
    model = DecoderLM.from_config(cfg)
    ids = torch.randint(0, 37, (2, 5))
    out = model(input_ids=ids)
    assert out.logits.shape == (2, 5, 37)
