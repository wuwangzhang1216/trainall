<!-- nav -->
<p align="center">
  <a href="10-lora-qlora.md">← LoRA/QLoRA</a> ·
  <a href="README.md">Index</a> ·
  <a href="../../GLOSSARY.md">Glossary</a> ·
  <a href="../11-architectures.md">中文</a>
</p>
<!-- /nav -->

# Model Architectures

> **The architecture is the "chassis" that every training method runs on top of: each objective (SFT, DPO, GRPO, …) is just a different gradient signal poured into the same decoder-only transformer. Understand the building blocks RMSNorm / RoPE / GQA / MLA / SwiGLU / MoE and you can see why modern LLMs are fast, long-context, and cheap.**

![Overall structure of a decoder-only transformer](../../assets/transformer_arch.png)

## Intuition

The earlier docs are all about "which loss do we train the model with." This one is about "what the model itself looks like" — i.e. what parts make up the object that gets called over and over as `model(input_ids=...)` and fed gradients by `loss.backward()`.

trainall's `DecoderLM` is a standard **decoder-only transformer** (the GPT / Llama / DeepSeek lineage). What it does can be condensed into one sentence: **embed a token sequence into vectors, repeatedly mix information across positions with "self-attention + feed-forward network," and finally project each position's vector back to the vocabulary to predict the distribution over the next token.**

But behind the word "standard" lie years of accumulated engineering refinements that make the model faster, the context longer, and inference cheaper. This doc dissects, one by one, the parts trainall implements, each with a "why you need it" plus the corresponding `ArchConfig` knob. Turn these knobs and, from the same codebase, you can sculpt a dense Llama, a GQA Mistral, or a MoE DeepSeek-style model.

## How it works (deep dive)

### Overall data flow

The path of one forward pass (`src/trainall/models/transformer.py`) is:

```
input_ids (B,T)
  └─ embed_tokens          # (B,T) -> (B,T,dim)
  └─ for layer in layers:  # n_layers DecoderBlocks
        x = x + Attn(RMSNorm(x))      # self-attention sub-layer + residual
        x = x + FFN(RMSNorm(x))       # feed-forward sub-layer + residual (dense or MoE)
  └─ norm                  # final RMSNorm
  └─ lm_head               # (B,T,dim) -> (B,T,vocab)  returns logits
```

Each layer is **pre-norm** (normalize first, then enter the sub-layer), with a residual wrapped around the outside of the sub-layer. This is the formulation used by GPT-NeoX / Llama; it trains deep networks more easily than the original transformer's post-norm — there is no normalization on the residual pathway, so gradients can pass straight through dozens of layers without decaying. The two core lines of `DecoderBlock` (`block.py`) are exactly the two `x = x + ...` lines above.

Below, each part is explained thoroughly on its own.

### RMSNorm — cheaper normalization

**Why you need it.** A transformer needs to keep the scale of activations stable before each sub-layer, otherwise the numerics of a deep network blow up or vanish. Classic LayerNorm does two things: subtract the mean (re-centering) and divide by the standard deviation (re-scaling), then add a learnable scale and bias. RMSNorm (Zhang & Sennrich 2019) found that **the re-centering step is basically useless** — drop it and keep only "divide by the root mean square (RMS)":

It computes just one RMS, skips one mean subtraction, and drops a set of bias parameters, so it is faster and saves more memory, while performing on par with or even better than LayerNorm. It is the default pre-norm of modern decoders (since Llama). trainall's implementation (`norm.py`) computes the RMS in fp32 and then casts back to the original dtype, keeping the numerics stable during bf16 training.

**Knob:** `norm_eps` (the ε in the denominator, prevents division by zero, default `1e-5`).

### RoPE — "rotating" position information into Q/K

**Why you need it.** Attention itself knows nothing about position — it only looks at inner products between tokens, and the result is unchanged if you shuffle the order. Position information must be injected explicitly. Early on, learnable absolute position embeddings were used, but they extrapolate poorly and cannot exceed the training length. RoPE (Su et al. 2021, RoFormer) takes an elegant approach: **rotate the query/key vectors by an angle that depends on position.** The vector at position $m$ is rotated by $m\theta$, the one at position $n$ by $n\theta$, so that their inner product depends only on the **relative offset** $m-n$. This naturally encodes "relative position" into the attention scores, and extrapolation is smoother too.

trainall's `RotaryEmbedding` (`rope.py`) precomputes `cos`/`sin` tables, and `apply_rotary` applies them to Q and K inside each layer's attention (note: **only Q/K are rotated, not V**).

**Long-context scaling (NTK / YaRN).** RoPE is trained within `max_seq_len` by default. To make the model work on longer sequences without retraining, scale the inverse frequency:

- `linear` (Position Interpolation, Chen 2023): divide the position coordinate by `factor`, which compresses all frequencies — simple, but high-frequency information gets blurred.
- `ntk` (bloc97 2023): enlarge the base `theta`, so high frequencies are "stretched less" and low frequencies "stretched more," preserving detail better than linear. In code this is `theta * factor**(dim/(dim-2))`.
- `yarn` (Peng et al. 2023): a per-frequency interpolation ramp that transitions smoothly between high and low frequencies, with an `mscale` temperature coefficient (`attn_factor`) to compensate the logit scale. The current SOTA long-context scheme.

**Knobs:** `rope_theta` (the base, default `1e4`), `rope_scaling` (`{"type": "linear"|"ntk"|"yarn", "factor": 2.0}` or `None`).

### GQA / MQA vs MHA — slimming down the KV cache

![Attention variants: how MHA / GQA / MQA / MLA share KV](../../assets/attention_variants.png)

**Why you need it.** At inference time, to avoid recomputation, each token's key/value is cached (the KV cache). The size of this cache = `n_layers × sequence_length × n_KV_heads × head_dim`, and under long context it eats up all the memory and becomes the decoding bottleneck.

- **MHA (Multi-Head Attention)**: each query head gets its own independent KV head. Most expressive, but the largest KV cache.
- **MQA (Multi-Query Attention, Shazeer 2019)**: all query heads share **one** KV head. The KV cache shrinks to `1/n_heads`, decoding is blazing fast, but quality drops slightly.
- **GQA (Grouped-Query Attention, Ainslie et al. 2023)**: a compromise — split the query heads into a few groups, each group sharing one KV head. This is what Llama-2/3 and Mistral use, striking the best balance between quality and memory.

trainall expresses all three with a single knob: `n_kv_heads`. `n_kv_heads == n_heads` is MHA, `n_kv_heads == 1` is MQA, and anything in between is GQA. In the attention code (`attention.py`), `repeat_kv` replicates the KV heads by `n_rep = n_heads // n_kv_heads` to align them with the number of query heads. Note: what shrinks is only the KV's **parameters and cache**; the number of query heads is unchanged, so the loss of expressiveness is limited.

**Knobs:** `n_heads`, `n_kv_heads` (requires `n_heads % n_kv_heads == 0`).

### MLA — DeepSeek's low-rank KV compression

**Why you need it.** GQA saves KV cache by "sharing heads," but sharing itself loses expressiveness. MLA (Multi-head Latent Attention, DeepSeek-V2 2024) takes a different angle: **don't share heads — instead, compress KV into a low-rank latent vector and up-project it later.** When caching, only that small latent (of dimension `kv_lora_rank`) is stored; it is decompressed into the full multi-head KV only when used. This makes the KV cache even smaller than GQA, yet preserves the "every head is independent" expressiveness.

One subtle engineering point: position information (RoPE) cannot be compressed and then decompressed, or the rotation would be broken. MLA's solution is **decoupled RoPE** — split each head into two segments: one "content / nope" segment (no position, goes through low-rank compression) and one small "rope" sub-dimension (`qk_rope_head_dim`, which carries RoPE separately, and this key segment is shared across all heads). trainall's `MultiHeadLatentAttention` (`attention.py`) implements this fully: `kv_a_proj` compresses, `kv_b_proj` up-projects, and the decoupled rope key is rotated separately and then `expand`ed to all heads. The query can optionally also go through `q_lora_rank` low-rank compression.

**Knobs:** `kv_lora_rank` (KV latent rank, setting it enables MLA), `q_lora_rank` (query low-rank, optional), `qk_rope_head_dim` (decoupled rope sub-dimension, default around `head_dim//2`). In `DecoderBlock`, as long as `kv_lora_rank` or `q_lora_rank` is non-empty it automatically switches to MLA, otherwise it uses GQA `Attention`.

### SwiGLU / GeGLU — gated feed-forward networks

**Why you need it.** The original transformer's feed-forward network is `down(ReLU(up(x)))`. The GLU variants (Shazeer 2020, "GLU Variants Improve Transformer") found that adding a **gate** significantly improves quality: use two parallel up-projections, one passed through an activation function to act as the "gate" and the other as the "value," multiply them element-wise, then down-project:

$$\text{FFN}(x) = \text{down}\big(\,\sigma(\text{gate}(x)) \odot \text{up}(x)\,\big)$$

`SwiGLU` uses SiLU/swish as the gate activation (Llama/PaLM); `GeGLU` uses GELU. The gate lets the network "selectively" let information through, boosting expressiveness at near-zero cost, and has become standard equipment in modern decoders. trainall's implementation (`mlp.py`) uses three bias-free `Linear`s: `gate_proj`, `up_proj`, `down_proj`.

**Knob:** `ffn_dim` (the intermediate hidden width; the gated version typically scales by `2/3` to keep the parameter count comparable).

### MoE — trading sparse activation for greater capacity

**Why you need it.** To make a model "know more," the most direct route is to widen the FFN — but then every token has to compute all the parameters, making both training and inference more expensive. MoE (Mixture-of-Experts, Shazeer 2017) takes the idea: place many FF "experts," but for each token **activate only the top-k of them.** So the total parameters (capacity) is large, yet the per-token compute (FLOPs) depends only on k. DeepSeek and Mixtral both rely on this to achieve "hundreds of B parameters, only tens of B activated."

trainall's `MoEFeedForward` (`moe.py`) flow: a `gate` linear layer scores each token → softmax → take the top-k experts and renormalize their gating weights → each selected expert (a SwiGLU) computes its own output, and these are summed weighted by the weights.

**The load-balancing aux loss is key.** Without a constraint, the router gets lazy — always sending tokens to the same few experts, while the rest starve and capacity is wasted. So an **auxiliary loss** is added (the importance × load form from Switch/Mixtral) to force the router to spread tokens evenly: it equals "the fraction of tokens each expert actually receives" times "the average probability mass the router assigns to each expert," times `n_experts` and a coefficient. This aux loss backpropagates together with the main LM loss. trainall accumulates it across all layers and attaches it to `out.aux_loss` (a 0 scalar for dense models). This is also why, when training MoE, your total loss is `lm_loss + aux_loss`.

**Knobs:** `use_moe` (the switch), `n_experts` (total number of experts), `n_experts_per_tok` (the top-k activated per token), `moe_aux_loss_coef` (load-balancing loss weight, default `0.01`).

### The data → objective → algorithm view

Putting this doc back into the whole framework: the architecture is the **object the objective acts upon.** The objectives SFT, DPO, GRPO always call the same `model(input_ids=...)` interface, take `out.logits`, and compute their own loss; the algorithm (full / LoRA / QLoRA, see [LoRA/QLoRA](10-lora-qlora.md)) decides **which parameters** receive gradients. In other words: the architecture defines "what the model is," the objective defines "in which direction to learn," and the algorithm defines "which weights to move." MoE's `aux_loss` is one of the few signals that "leak" from the architecture layer into the objective — it is a regularizer built into the model structure, and it must be added into the total loss to be optimized together.

## Objective (the math)

The architecture itself does not define the training loss (that's the objective's job), but there are two pieces of **architecture-intrinsic math** worth writing down clearly.

**RMSNorm.** For an input $x$ of dimension $d$:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot g$$

where $g \in \mathbb{R}^d$ is the learnable per-channel gain (`self.weight`) and $\epsilon$ is `norm_eps`. Compared with LayerNorm, it omits the mean-subtraction term $-\,\mathbb{E}[x]$.

**RoPE's relative-position property.** Take the query at position $m$ and the key at position $n$; the rotation matrix $R_\Theta^m$ rotates each channel pair $(2i, 2i{+}1)$ by angle $m\theta_i$, where $\theta_i = \text{theta}^{-2i/d}$. Then the rotated inner product satisfies

$$\langle R_\Theta^m\, q,\; R_\Theta^n\, k \rangle = \langle q,\; R_\Theta^{\,n-m}\, k \rangle$$

i.e. the attention score **depends only on the relative offset $n-m$** — this is precisely the mathematical root of how RoPE gives the model a translation-invariant sense of position.

**MoE load-balancing aux loss.** Let there be $N$ tokens and $E$ experts, with router probability $p_{j,e}$ (token $j$ to expert $e$). Define the fraction of tokens each expert is actually dispatched, $f_e$ (load), and the average router mass $P_e$ (importance):

$$f_e = \frac{1}{N}\sum_{j=1}^{N} \mathbb{1}[\,e \in \text{top-}k(j)\,], \qquad P_e = \frac{1}{N}\sum_{j=1}^{N} p_{j,e}$$

$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{e=1}^{E} f_e\, P_e$$

$\alpha$ is `moe_aux_loss_coef`. When an expert is both frequently dispatched (large $f_e$) and highly preferred by the router (large $P_e$), this term grows, and optimizing it pushes the load back toward a uniform distribution. The total training objective is $\mathcal{L} = \mathcal{L}_{\text{LM}} + \mathcal{L}_{\text{aux}}$.

## Data format

The architecture layer does not consume a `Batch`, but the most basic tensors. The signature of `DecoderLM.forward` has only two arguments:

```python
import torch
from trainall.models import ArchConfig, DecoderLM

model = DecoderLM.from_config(
    ArchConfig(vocab_size=64, dim=32, n_layers=2, n_heads=4, n_kv_heads=2,
               ffn_dim=64, max_seq_len=64))
input_ids = torch.randint(0, 64, (2, 8))       # LongTensor (B, T): token id
attention_mask = torch.ones_like(input_ids)    # optional (B, T): 1=real token, 0=padding

out = model(
    input_ids,        # LongTensor (B, T): token id
    attention_mask,   # optional (B, T): 1=real token, 0=padding
)
# returns LMOutput:
#   out.logits    FloatTensor (B, T, vocab_size)   unnormalized distribution over the next token
#   out.aux_loss  scalar Tensor                     MoE load-balancing loss (0 for dense)
```

- `attention_mask` is a **padding mask** (which positions are real tokens). Internally the model `AND`s it with the **causal triangular matrix** (each position can only see itself and to its left) before feeding it to `scaled_dot_product_attention`. So you don't need to build a causal mask yourself — whether or not you pass `attention_mask`, it is causal.
- **No `labels`**: the architecture is only responsible for computing logits; the loss is computed by the objective layer (e.g. `SFTObjective`) from logits and labels. This is exactly the embodiment of "decoupling architecture from objective."

## Using it in trainall

The snippet below runs in a few milliseconds on CPU: build a mini `DecoderLM` with GQA (`n_kv_heads=2 < n_heads=4`) + MoE (top-2 of 4 experts), run one forward + backward, and print the shape of `out.logits`, the `out.aux_loss`, and the number of parameters that received gradients.

```python
import torch
from trainall.models import ArchConfig, DecoderLM

# A tiny DeepSeek/Llama-style decoder: GQA (n_kv_heads < n_heads) + MoE FFN.
cfg = ArchConfig(
    vocab_size=64, dim=32, n_layers=2,
    n_heads=4, n_kv_heads=2,            # GQA: 4 query heads share 2 KV heads
    ffn_dim=64, max_seq_len=64,
    use_moe=True, n_experts=4, n_experts_per_tok=2,  # top-2 of 4 experts
)
model = DecoderLM.from_config(cfg)

ids = torch.randint(0, cfg.vocab_size, (2, 8))         # (B=2, T=8)
out = model(input_ids=ids, attention_mask=torch.ones_like(ids))

print("logits:", tuple(out.logits.shape))               # (2, 8, 64)
print("aux_loss:", out.aux_loss.item())                  # MoE load-balance term > 0

# Joint backward over the LM signal + the MoE auxiliary loss.
loss = out.logits.float().mean() + out.aux_loss
loss.backward()
n_grad = sum(p.grad is not None for p in model.parameters())
print("loss:", loss.item(), "| params with grad:", n_grad)
```

Actual run output (your numbers will differ slightly with the random seed, but the shapes and "aux_loss > 0" always hold):

```
logits: (2, 8, 64)
aux_loss: 0.04033871740102768
loss: 0.05070743337273598 | params with grad: 40
```

Set `use_moe=False` to go back to dense SwiGLU (`aux_loss` becomes a 0 scalar); set `n_kv_heads=4` for MHA, `n_kv_heads=1` for MQA; set `kv_lora_rank=8, q_lora_rank=8` to switch to MLA attention. The same `ArchConfig` covers all variants.

## When to use / when not

This doc is not a decision about "when to train," but a cheat sheet for **architecture selection**:

- **RMSNorm + RoPE + SwiGLU**: the no-brainer default for modern decoders, no reason not to use them.
- **GQA (`n_kv_heads` < `n_heads`)**: should almost always be on. The benefit is greatest for long context and when you need high-throughput inference; only consider keeping MHA for small models, when chasing ultimate quality and not caring about inference memory.
- **MQA (`n_kv_heads=1`)**: when extremely chasing decoding speed/memory (edge devices, ultra-long context) and able to accept a bit of quality loss.
- **MLA (`kv_lora_rank`)**: when you want a smaller KV cache than GQA without sacrificing expressiveness — but it is complex to implement and has little ecosystem support, so unless you are replicating DeepSeek-class models, GQA is usually enough.
- **MoE (`use_moe=True`)**: when you want to substantially boost capacity (knowledge) at a fixed inference FLOPs budget. The cost is exploding memory (all experts must reside), unstable training (needs aux-loss tuning, load balancing), and complex distributed implementation. For small-scale experiments or when memory is tight, dense is more carefree.
- **RoPE scaling (NTK/YaRN)**: only needed when you want to use an already-trained model on contexts beyond `max_seq_len`; if training from scratch, just set `max_seq_len` large.

## Pitfalls & practical notes

- **MoE's `aux_loss` must be added into the total loss.** Backpropagating only `lm_loss` will make the router collapse onto a few experts. trainall attaches it to `out.aux_loss`, and the training loop should be written as `loss = lm_loss + out.aux_loss`. trainall's built-in objectives already handle this for you; just don't forget it when writing your own loss.
- **`dim` must be divisible by `n_heads`** (unless you give `head_dim` explicitly), and **`n_heads` must be an integer multiple of `n_kv_heads`**, otherwise `ArchConfig.__post_init__` raises an error directly.
- **RoPE applies only to Q/K, not to V.** This is RoPE's definition; modifying V would break the relative-position property.
- **Don't manually stack a causal mask.** `forward` already merges the padding mask with the causal triangle internally; applying it again would scramble the mask.
- **GQA saves on KV, not query.** `repeat_kv` replicates the KV heads to align with the query heads, so the attention compute is basically unchanged — what's saved is KV cache memory and K/V projection parameters, not the attention FLOPs.
- **`tie_embeddings=True` (default) makes `lm_head` and `embed_tokens` share weights**, saving parameters and often being more stable; but if you want to train input/output embeddings separately, set it to `False`.
- **MoE is far harder to converge than dense.** Top-k routing is non-differentiable (approximated with straight-through), and early training easily leads to imbalanced experts. Get dense working first, then move to MoE.
- **NTK/YaRN is an inference-time extrapolation trick, not a free lunch.** After scaling, short-context performance may degrade slightly; the ideal approach is to do a small amount of long-context fine-tuning after scaling.

## Related

- [Pretraining](01-pretraining.md) — how this architecture first learned to be a language model from random initialization.
- [LoRA / QLoRA](10-lora-qlora.md) — how the algorithm layer decides "which weights to move," complementing this doc's "what the model is."
- [SFT](03-sft.md) — how the objective layer takes `out.logits` to compute the supervised loss.
- [Preference Optimization](04-preference-optimization.md) and [RLVR / GRPO](06-rlvr-grpo.md) — advanced objectives run on the same architecture.
- Glossary: [RoPE](../../GLOSSARY.md#rope), [GQA](../../GLOSSARY.md#gqa), [MoE](../../GLOSSARY.md#moe), [RMSNorm](../../GLOSSARY.md#rmsnorm), [SwiGLU](../../GLOSSARY.md#swiglu).
- Back to the [methods index](README.md).
