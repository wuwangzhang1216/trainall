<!-- nav -->
<table width="100%"><tr><td align="left" width="30%"><a href="10-lora-qlora.md">← LoRA / QLoRA</a></td><td align="center" width="40%"><a href="README.md">📑 索引</a> · <a href="../GLOSSARY.md">📖 术语词典</a> · <a href="en/11-architectures.md">🌐 English</a></td><td align="right" width="30%"><sub>&nbsp;</sub></td></tr></table>
<!-- /nav -->

# 模型架构 (Model Architectures)

> **架构是所有训练方法运行其上的"底座"：每一个 objective（SFT、DPO、GRPO……）都只是往同一个 decoder-only transformer 里灌不同的梯度信号。理解 RMSNorm / RoPE / GQA / MLA / SwiGLU / MoE 这几块积木，才能看懂为什么现代 LLM 又快又长又便宜。**

![Decoder-only transformer 的整体结构](../assets/transformer_arch.png)

## 直觉：它到底在做什么

前面的文档讲的都是"用什么 loss 训练模型"。这一篇讲的是"模型本身长什么样"——也就是那个被反复 `model(input_ids=...)` 调用、被 `loss.backward()` 灌梯度的对象到底由哪些零件组成。

trainall 里的 `DecoderLM` 是一个标准的 **decoder-only transformer**（GPT / Llama / DeepSeek 一脉）。它的工作可以浓缩成一句话：**把 token 序列嵌入成向量，反复用"自注意力 + 前馈网络"对每个位置做信息混合，最后把每个位置的向量投影回词表，预测下一个 token 的分布。**

但"标准"这两个字背后，过去几年积累了一整套让模型更快、上下文更长、推理更省钱的工程改良。这篇文档逐一拆解 trainall 实现的那些零件，每个都给出"为什么需要它"加上"对应的 `ArchConfig` 旋钮"。把这些旋钮调一调，你就能在同一份代码里捏出 dense Llama、GQA Mistral、或者 MoE 的 DeepSeek 风格模型。

## 原理与架构（深度讲解）

### 整体数据流

一次前向（`src/trainall/models/transformer.py`）的路径是：

```
input_ids (B,T)
  └─ embed_tokens          # (B,T) -> (B,T,dim)
  └─ for layer in layers:  # n_layers 个 DecoderBlock
        x = x + Attn(RMSNorm(x))      # 自注意力子层 + 残差
        x = x + FFN(RMSNorm(x))       # 前馈子层 + 残差（dense 或 MoE）
  └─ norm                  # 最后一层 RMSNorm
  └─ lm_head               # (B,T,dim) -> (B,T,vocab)  返回 logits
```

每一层都是 **pre-norm**（先归一化再进子层），子层外面包一层残差。这是 GPT-NeoX / Llama 用的写法，比原始 transformer 的 post-norm 更容易训练深层网络——残差通路上没有归一化，梯度可以原样穿过几十层而不衰减。`DecoderBlock`（`block.py`）的两行核心就是上面那两行 `x = x + ...`。

下面把每个零件单独讲透。

### RMSNorm —— 更便宜的归一化

**为什么需要它。** Transformer 需要在每个子层前把激活值的尺度稳住，否则深层网络的数值会爆炸或消失。经典 LayerNorm 做两件事：减去均值（re-centering）、除以标准差（re-scaling），再加可学习的缩放和偏置。RMSNorm（Zhang & Sennrich 2019）发现 **re-centering 那一步基本没用**——把它砍掉，只保留"除以均方根（root mean square）"这一步：

只算一个 RMS、少一次减均值、少一组 bias 参数，于是更快、更省显存，而效果和 LayerNorm 持平甚至更好。这是现代 decoder（Llama 起）的默认 pre-norm。trainall 的实现（`norm.py`）在 fp32 里算 RMS 再 cast 回原 dtype，保证 bf16 训练时数值稳定。

**旋钮：** `norm_eps`（分母里的 ε，防止除零，默认 `1e-5`）。

### RoPE —— 把位置信息"旋转"进 Q/K

**为什么需要它。** 注意力本身对位置一无所知——它只看 token 之间的内积，打乱顺序结果不变。必须显式注入位置信息。早期用可学习的绝对位置嵌入，但它外推（extrapolation）能力差、不能超过训练长度。RoPE（Su et al. 2021, RoFormer）的做法很优雅：**把 query/key 向量按位置旋转一个角度。** 第 $m$ 个位置的向量旋转 $m\theta$，第 $n$ 个位置旋转 $n\theta$，于是它们的内积只依赖**相对偏移** $m-n$。这就把"相对位置"天然编码进了注意力分数里，外推也更平滑。

trainall 的 `RotaryEmbedding`（`rope.py`）预计算 `cos`/`sin` 表，`apply_rotary` 在每层注意力里把它作用到 Q、K 上（注意：**只旋转 Q/K，不旋转 V**）。

**长上下文缩放（NTK / YaRN）。** RoPE 默认训练在 `max_seq_len` 内。想让模型在更长序列上工作而不重训，就缩放 inverse frequency：

- `linear`（Position Interpolation, Chen 2023）：把位置坐标除以 `factor`，等于把所有频率压缩——简单但高频信息会糊。
- `ntk`（bloc97 2023）：放大底数 `theta`，让高频"少拉伸"、低频"多拉伸"，比 linear 更保细节。代码里是 `theta * factor**(dim/(dim-2))`。
- `yarn`（Peng et al. 2023）：逐频率的插值 ramp，在高低频之间平滑过渡，并用一个 `mscale` 温度系数（`attn_factor`）补偿 logits 尺度。当前 SOTA 的长上下文方案。

**旋钮：** `rope_theta`（底数，默认 `1e4`）、`rope_scaling`（`{"type": "linear"|"ntk"|"yarn", "factor": 2.0}` 或 `None`）。

### GQA / MQA vs MHA —— 让 KV cache 瘦下来

![注意力变体：MHA / GQA / MQA / MLA 的 KV 共享方式](../assets/attention_variants.png)

**为什么需要它。** 推理时为了不重复计算，会把每个 token 的 key/value 缓存起来（KV cache）。这块缓存的大小 = `层数 × 序列长度 × KV 头数 × head_dim`，在长上下文下会吃光显存、成为解码瓶颈。

- **MHA（Multi-Head Attention）**：每个 query 头配一个独立的 KV 头。表达力最强，但 KV cache 最大。
- **MQA（Multi-Query Attention, Shazeer 2019）**：所有 query 头共享**一个** KV 头。KV cache 缩到 `1/n_heads`，解码飞快，但质量略降。
- **GQA（Grouped-Query Attention, Ainslie et al. 2023）**：折中——把 query 头分成几组，每组共享一个 KV 头。Llama-2/3、Mistral 用的就是它，在质量和显存之间取得最好平衡。

trainall 用一个旋钮统一表达三者：`n_kv_heads`。`n_kv_heads == n_heads` 是 MHA，`n_kv_heads == 1` 是 MQA，介于两者之间是 GQA。注意力里（`attention.py`）的 `repeat_kv` 把 KV 头按 `n_rep = n_heads // n_kv_heads` 复制对齐到 query 头数。注意：少的只是 KV 的**参数和缓存**，query 头数不变，所以表达力损失有限。

**旋钮：** `n_heads`、`n_kv_heads`（要求 `n_heads % n_kv_heads == 0`）。

### MLA —— DeepSeek 的低秩 KV 压缩

**为什么需要它。** GQA 通过"共享头"省 KV cache，但共享本身会损失表达力。MLA（Multi-head Latent Attention, DeepSeek-V2 2024）换了个思路：**不共享头，而是把 KV 压缩进一个低秩 latent 向量再上投影。** 缓存时只存那个小 latent（`kv_lora_rank` 维），用的时候才解压成完整的多头 KV。这样 KV cache 比 GQA 还小，却保住了"每个头独立"的表达力。

一个微妙的工程点：位置信息（RoPE）不能被压进去再解压，否则旋转会被破坏。MLA 的解法是 **decoupled RoPE**——把每个头切成两段：一段"content / nope"（无位置，走低秩压缩），一段小的"rope"子维度（`qk_rope_head_dim`，单独带 RoPE，且这段 key 在所有头间共享）。trainall 的 `MultiHeadLatentAttention`（`attention.py`）完整实现了这套：`kv_a_proj` 压缩、`kv_b_proj` 上投影、decoupled rope key 单独旋转再 `expand` 到所有头。query 也可选地走 `q_lora_rank` 低秩压缩。

**旋钮：** `kv_lora_rank`（KV latent 秩，设了就启用 MLA）、`q_lora_rank`（query 低秩，可选）、`qk_rope_head_dim`（decoupled rope 子维度，默认 `head_dim//2` 附近）。在 `DecoderBlock` 里，只要 `kv_lora_rank` 或 `q_lora_rank` 非空就自动切到 MLA，否则用 GQA `Attention`。

### SwiGLU / GeGLU —— 门控前馈网络

**为什么需要它。** 原始 transformer 的前馈网络是 `down(ReLU(up(x)))`。GLU 变体（Shazeer 2020, "GLU Variants Improve Transformer"）发现加一个**门控**能显著提升质量：用两个并行的上投影，一个过激活函数当"门"，另一个当"值"，逐元素相乘后再下投影：

$$\text{FFN}(x) = \text{down}\big(\,\sigma(\text{gate}(x)) \odot \text{up}(x)\,\big)$$

`SwiGLU` 用 SiLU/swish 当门激活（Llama/PaLM），`GeGLU` 用 GELU。门控让网络能"选择性地"放行信息，几乎零成本地提升了表达力，成了现代 decoder 的标配。trainall 的实现（`mlp.py`）三个无 bias 的 `Linear`：`gate_proj`、`up_proj`、`down_proj`。

**旋钮：** `ffn_dim`（中间隐层宽度；门控版通常按 `2/3` 折算保持参数量可比）。

### MoE —— 用稀疏激活换更大容量

**为什么需要它。** 想让模型"知道得更多"，最直接是把 FFN 加宽——但那样每个 token 都要算全部参数，训练和推理都变贵。MoE（Mixture-of-Experts, Shazeer 2017）的思路是：放很多个 FF"专家"，但每个 token **只激活其中 top-k 个**。于是总参数（容量）很大，单 token 的计算量（FLOPs）却只取决于 k。DeepSeek、Mixtral 都靠它做到"几百 B 参数、只激活几十 B"。

trainall 的 `MoEFeedForward`（`moe.py`）流程：一个 `gate` 线性层给每个 token 打分 → softmax → 取 top-k 个专家、对其门控权重重归一化 → 每个被选中的专家（一个 SwiGLU）算自己的输出，按权重加权求和。

**负载均衡 aux loss 是关键。** 如果不加约束，router 会偷懒——总把 token 发给同几个专家，其余专家饿死、容量浪费。所以加一个**辅助损失**（Switch/Mixtral 的 importance × load 形式）逼 router 把 token 均匀分散：它等于"每个专家实际接到的 token 比例"乘以"router 给每个专家的平均概率质量"，再乘 `n_experts` 和系数。这个 aux loss 会和主 LM loss 一起 backward。trainall 把它从每层一路加总，最后挂在 `out.aux_loss` 上（dense 模型时是 0 标量）。这也是为什么训练 MoE 时你的总 loss 是 `lm_loss + aux_loss`。

**旋钮：** `use_moe`（开关）、`n_experts`（专家总数）、`n_experts_per_tok`（每 token 激活的 top-k）、`moe_aux_loss_coef`（负载均衡损失权重，默认 `0.01`）。

### data → objective → algorithm 的视角

把这一篇放回整个框架：架构是 **objective 作用的对象**。SFT、DPO、GRPO 这些 objective 调用的永远是同一个 `model(input_ids=...)` 接口，拿到 `out.logits` 算各自的 loss；algorithm（full / LoRA / QLoRA，见 [LoRA/QLoRA](10-lora-qlora.md)）决定**哪些参数**接收梯度。换句话说：架构定义了"模型是什么"，objective 定义了"往哪个方向学"，algorithm 定义了"动哪些权重"。MoE 的 `aux_loss` 是少数从架构层"漏"进 objective 的信号——它是模型结构自带的正则项，必须被加进总 loss 里一起优化。

## 目标函数（数学）

架构本身不定义训练 loss（那是 objective 的事），但有两个**架构内生的数学**值得写清楚。

**RMSNorm.** 对维度为 $d$ 的输入 $x$：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot g$$

其中 $g \in \mathbb{R}^d$ 是可学习的逐通道增益（`self.weight`），$\epsilon$ 是 `norm_eps`。对比 LayerNorm 少了减均值项 $-\,\mathbb{E}[x]$。

**RoPE 的相对位置性质.** 设位置 $m$ 的 query 与位置 $n$ 的 key，旋转矩阵 $R_\Theta^m$ 把每对通道 $(2i, 2i{+}1)$ 旋转角度 $m\theta_i$，其中 $\theta_i = \text{theta}^{-2i/d}$。则旋转后的内积满足

$$\langle R_\Theta^m\, q,\; R_\Theta^n\, k \rangle = \langle q,\; R_\Theta^{\,n-m}\, k \rangle$$

即注意力分数**只依赖相对偏移 $n-m$**——这正是 RoPE 让模型获得平移不变位置感的数学根源。

**MoE 负载均衡 aux loss.** 设 $N$ 个 token、$E$ 个专家，router 概率 $p_{j,e}$（token $j$ 到专家 $e$）。定义每个专家被实际派到的 token 比例 $f_e$（load）和平均 router 质量 $P_e$（importance）：

$$f_e = \frac{1}{N}\sum_{j=1}^{N} \mathbb{1}[\,e \in \text{top-}k(j)\,], \qquad P_e = \frac{1}{N}\sum_{j=1}^{N} p_{j,e}$$

$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{e=1}^{E} f_e\, P_e$$

$\alpha$ 是 `moe_aux_loss_coef`。当某个专家既被频繁派到（$f_e$ 大）又被 router 高度偏好（$P_e$ 大）时该项变大，优化它会把负载推回均匀分布。总训练目标是 $\mathcal{L} = \mathcal{L}_{\text{LM}} + \mathcal{L}_{\text{aux}}$。

## 数据长什么样

架构层吃的不是 `Batch`，而是最朴素的张量。`DecoderLM.forward` 的签名只有两个参数：

```python
import torch
from trainall.models import ArchConfig, DecoderLM

model = DecoderLM.from_config(
    ArchConfig(vocab_size=64, dim=32, n_layers=2, n_heads=4, n_kv_heads=2,
               ffn_dim=64, max_seq_len=64))
input_ids = torch.randint(0, 64, (2, 8))       # LongTensor (B, T)：token id
attention_mask = torch.ones_like(input_ids)    # 可选 (B, T)：1=真实 token，0=padding

out = model(
    input_ids,        # LongTensor (B, T)：token id
    attention_mask,   # 可选 (B, T)：1=真实 token，0=padding
)
# 返回 LMOutput：
#   out.logits    FloatTensor (B, T, vocab_size)   下一 token 的未归一化分布
#   out.aux_loss  标量 Tensor                       MoE 负载均衡损失（dense 时为 0）
```

- `attention_mask` 是 **padding mask**（哪些位置是真 token）。模型内部会把它和**因果三角矩阵**（每个位置只能看自己和左边）`AND` 起来，再喂给 `scaled_dot_product_attention`。所以你不需要自己造因果 mask——传不传 `attention_mask` 它都是因果的。
- **没有 `labels`**：架构只负责算 logits，loss 由 objective 层（如 `SFTObjective`）拿 logits 和 labels 去算。这正是"架构与 objective 解耦"的体现。

## 在 trainall 中怎么用

下面这段在 CPU 上几毫秒跑完：搭一个 GQA（`n_kv_heads=2 < n_heads=4`）+ MoE（top-2 of 4 experts）的迷你 `DecoderLM`，跑一次前向 + 反向，打印 `out.logits` 形状、`out.aux_loss`、以及拿到梯度的参数数。

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

实际运行输出（你的数值会因随机种子略有不同，但形状和"aux_loss > 0"恒成立）：

```
logits: (2, 8, 64)
aux_loss: 0.04033871740102768
loss: 0.05070743337273598 | params with grad: 40
```

把 `use_moe=False` 就回到 dense SwiGLU（`aux_loss` 变成 0 标量）；把 `n_kv_heads=4` 就是 MHA、`n_kv_heads=1` 就是 MQA；设 `kv_lora_rank=8, q_lora_rank=8` 则切到 MLA 注意力。同一份 `ArchConfig` 覆盖了全部变体。

## 何时用 / 何时不用

这一篇不是"何时训练"的决策，而是**架构选型**的速查：

- **RMSNorm + RoPE + SwiGLU**：现代 decoder 的无脑默认，没有理由不用。
- **GQA（`n_kv_heads` < `n_heads`）**：几乎总该开。长上下文、需要高吞吐推理时收益最大；只有在小模型、追求极致质量且不在乎推理显存时才考虑保留 MHA。
- **MQA（`n_kv_heads=1`）**：极端追求解码速度/显存（边缘设备、超长上下文）且能接受一点质量损失时。
- **MLA（`kv_lora_rank`）**：想要比 GQA 更小的 KV cache 又不愿牺牲表达力时——但实现复杂、生态支持少，除非你在复刻 DeepSeek 类模型，否则 GQA 通常够用。
- **MoE（`use_moe=True`）**：想在固定推理 FLOPs 下大幅提升容量（知识量）时。代价是显存暴涨（所有专家都要驻留）、训练不稳定（需要 aux loss 调参、负载均衡）、分布式实现复杂。小规模实验或显存吃紧时用 dense 更省心。
- **RoPE scaling（NTK/YaRN）**：只有当你要把已训模型用到超过 `max_seq_len` 的上下文时才需要；从头训练就直接把 `max_seq_len` 设大。

## 常见陷阱与实践要点

- **MoE 的 `aux_loss` 必须加进总 loss。** 只 backward `lm_loss` 会让 router 坍缩到少数专家。trainall 把它挂在 `out.aux_loss`，训练循环要写成 `loss = lm_loss + out.aux_loss`。trainall 内置 objective 已经替你处理；自己写 loss 时别漏。
- **`dim` 必须能被 `n_heads` 整除**（除非显式给 `head_dim`），且 **`n_heads` 必须是 `n_kv_heads` 的整数倍**，否则 `ArchConfig.__post_init__` 直接报错。
- **RoPE 只作用于 Q/K，不作用于 V。** 这是 RoPE 的定义；改 V 会破坏相对位置性质。
- **不要手动叠加因果 mask。** `forward` 内部已把 padding mask 和因果三角合并，重复施加会让 mask 错乱。
- **GQA 省的是 KV，不是 query。** `repeat_kv` 把 KV 头复制对齐 query 头，所以注意力计算量基本不变——省的是 KV cache 显存和 K/V 投影参数，不是 attention 的 FLOPs。
- **`tie_embeddings=True`（默认）让 `lm_head` 和 `embed_tokens` 共享权重**，省参数也常更稳；但若想分别训练输入/输出嵌入，设为 `False`。
- **MoE 训练比 dense 难收敛得多。** top-k 路由不可导（用 straight-through 近似），early training 容易专家不均衡。先把 dense 跑通，再上 MoE。
- **NTK/YaRN 是推理期外推技巧，不是免费午餐。** 缩放后短上下文性能可能略降；理想做法是缩放后再做少量长上下文微调。

## 相关

- [Pretraining](01-pretraining.md) —— 这套架构最初是怎么从随机初始化学成语言模型的。
- [LoRA / QLoRA](10-lora-qlora.md) —— algorithm 层如何决定"动哪些权重"，与本篇的"模型是什么"互补。
- [SFT](03-sft.md) —— objective 层如何拿 `out.logits` 算监督损失。
- [偏好优化 (Preference Optimization)](04-preference-optimization.md) 与 [RLVR / GRPO](06-rlvr-grpo.md) —— 同一架构上跑的进阶 objective。
- 术语表：[RoPE](../GLOSSARY.md#rope)、[GQA](../GLOSSARY.md#gqa)、[MoE](../GLOSSARY.md#moe)、[RMSNorm](../GLOSSARY.md#rmsnorm)、[SwiGLU](../GLOSSARY.md#swiglu)。
- 返回 [方法索引](README.md)。
