<!-- nav -->
<table width="100%"><tr><td align="left" width="30%"><a href="01-pretraining.md">← 预训练</a></td><td align="center" width="40%"><a href="README.md">📑 索引</a> · <a href="../GLOSSARY.md">📖 术语词典</a> · <a href="en/02-continued-pretraining.md">🌐 English</a></td><td align="right" width="30%"><a href="03-sft.md">SFT 指令微调 →</a></td></tr></table>
<!-- /nav -->

# 继续预训练 / 领域自适应预训练 (Continued Pre-training / Domain-Adaptive Pre-training, CPT / DAPT)

> **用和预训练一模一样的 next-token 目标，在领域语料上"接着练"——本质是数据问题，不是损失函数问题；难点在于让模型学会新分布的同时别忘了旧的（catastrophic forgetting），靠 replay 混入原始分布来缓解。**

![CPT/DAPT：同一个 next-token 目标，喂领域语料 + replay 以避免遗忘](../assets/cpt_dapt.png)

## 直觉：它到底在做什么

一个基座模型 (base model) 在通用网页、书籍、代码上预训练后，对"语言一般规律"已经很在行，但对某个**狭窄分布**——比如法律判决书、临床病历、某条芯片产线的工单、一种小众编程语言——可能见得很少。CPT 的想法朴素到几乎不像一个"方法"：**继续做预训练，只不过把语料换成这个领域的原始文本**。损失函数原封不动还是 next-token 交叉熵，优化器、数据管线都不变，只是数据分布平移了。

Gururangan 等人 (2020, *Don't Stop Pretraining*) 系统验证了这件事：在领域语料上多跑一轮预训练（他们称 **DAPT, domain-adaptive pre-training**），再做下游微调，几乎在所有领域/任务上都比直接微调基座更好。直觉是：模型先在自监督信号里"读熟"领域的词汇、术语搭配、文体节奏和事实分布，把这些知识压进权重，下游任务再用就事半功倍。

但天下没有免费的午餐。当你只喂领域文本、不停地把梯度往"法律腔"上推时，模型会逐渐**遗忘**它原本会的东西——通用问答、常识、其它语言。这就是 **灾难性遗忘 (catastrophic forgetting)**。CPT 的工程核心，与其说是"怎么学新的"，不如说是"怎么在学新的时候少忘旧的"。最常用的解药是 **replay（回放）**：在领域语料里掺一小撮原始预训练分布的样本，让梯度始终被旧分布"拉"一下，权重不至于跑得太偏。

## 原理与架构（深度讲解）

**数据 → 目标 → 算法**的三层框架下，CPT 几乎只动了第一层：

- **数据 (data)**：从"通用大杂烩"变成"领域语料 + 少量 replay 样本"。这是 CPT 唯一真正特别的地方，也是它成败的关键。
- **目标 (objective)**：完全不变，还是 [预训练](01-pretraining.md) 的 causal-LM 损失——对每个位置预测下一个 token，求负对数似然 (NLL)。在 trainall 里 `ContinuedPretrainObjective` 直接**继承** `CausalLMObjective`，默认情况下连一行计算都不改，走的是父类的 fast path。
- **算法 (algorithm)**：同样不变，常规的 AdamW + 余弦/线性退火。区别更多在**超参**：CPT 通常用比初始预训练**更小的学习率**、更短的 warmup，因为你是在一个已经收敛的点附近做小幅迁移，不是从随机初始化起步——学习率太大等于把好不容易学到的通用能力直接砸碎。

**模型到底学到了什么？** 自监督的 next-token 目标逼模型去建模 $P(\text{下一个 token} \mid \text{前文})$。当前文统计从"通用英文"变成"判决书"，模型要把概率质量重新分配到领域里高频的延续上——"被告" 后面更可能跟 "应当承担"，`SELECT` 后面更可能跟某张特定表名。这种重分配会改写注意力如何聚合上下文、FFN 如何存取事实记忆。换句话说，CPT 把**领域的分布先验**写进了权重，而这是任何提示工程 (prompt engineering) 或检索增强 (RAG) 都替代不了的——后两者只能在推理时"提示"模型，无法改变模型内部对 token 序列的概率信念。

**遗忘从机制上是怎么发生的？** 神经网络的知识是分布式编码在共享权重里的。当新数据的梯度持续把某些权重往一个方向推，原本依赖这些权重表达旧知识的"电路"就被覆盖。遗忘的严重程度大致正比于：新旧分布的差异、学习率、训练步数。**Replay** 的作用是在损失里持续保留一份"旧分布的梯度信号"——只要每个 batch 里有一定比例的原始样本，优化方向就被约束在"既降低领域 NLL、又不显著抬高通用 NLL"的折中上。经验上 replay 比例常取 1%–25% 不等：领域离基座越远、越担心遗忘，就掺得越多。

**trainall 里的 replay 加权机制。** 物理上把 replay 样本混进语料、让它们以一定频率出现，是最直接的 replay。`ContinuedPretrainObjective` 在此之上多给了一个**可选的 per-sample 加权旋钮**，让 collator 能在**同一个 batch 内**对 domain / replay 样本施加不同权重，而不必改动数据管线本身。它按优先级解析权重：

1. `batch.extra["weights"]`——显式给一个 `(B,)` 的权重向量，最灵活；
2. `batch.extra[domain_field]`（默认字段名 `"domain"`）配合 `replay_weight`——domain 样本（标记为真）权重记 `1.0`，replay 样本（标记为假）权重记 `replay_weight`；
3. 两者都没有 → 均匀权重，此时 CPT 退化成**和普通预训练逐位等价**（走父类 fast path，零额外开销）。

注意 `replay_weight=0.0`（默认）也走 fast path——它表示"不做任何 batch 内重加权"，并**不是**"把 replay 样本权重置零"。要启用加权，`replay_weight` 必须非零且 batch 里带 `domain` 标记（或直接给 `weights`）。

## 目标函数（数学）

基础就是 causal-LM 的负对数似然。对一条长度 $T$ 的序列 $x = (x_1, \dots, x_T)$：

$$
\mathcal{L}_{\text{CLM}}(x) = -\frac{1}{|\mathcal{M}|}\sum_{t \in \mathcal{M}} \log P_\theta\!\left(x_{t} \mid x_{\lt t}\right)
$$

其中 $\theta$ 是模型参数，$P_\theta(x_t \mid x_{\lt t})$ 是模型对位置 $t$ 真实 token 的预测概率，$\mathcal{M}$ 是参与计损的位置集合（标签为 `-100` 的位置被忽略，CPT 通常**所有 token 都计损**），$|\mathcal{M}|$ 是有效 token 数。这正是 [预训练](01-pretraining.md) 的损失。

当启用 batch 内加权时，trainall 先对每条样本算**逐 token 平均的 NLL**，再按样本权重做加权平均。设 batch 有 $B$ 条样本，第 $i$ 条的逐 token 平均 NLL 为 $\ell_i$、权重为 $w_i$：

$$
\ell_i = -\frac{1}{n_i}\sum_{t \in \mathcal{M}_i} \log P_\theta\!\left(x^{(i)}_{t} \mid x^{(i)}_{\lt t}\right), \qquad
\mathcal{L}_{\text{CPT}} = \frac{\sum_{i=1}^{B} w_i\, \ell_i}{\sum_{i=1}^{B} w_i}
$$

- $n_i = |\mathcal{M}_i|$：第 $i$ 条样本参与计损的 token 数（`clamp(min=1)` 防止除零）；
- $w_i$：第 $i$ 条样本的权重。domain 样本取 $1.0$，replay 样本取 `replay_weight`（记作 $\rho$）；
- 分母 $\sum_i w_i$ 做归一化，使整体损失尺度与不加权时**可比**（不会因为掺了低权重样本就整体缩小）。

直觉：把 $\rho$ 调小，等于告诉优化器"replay 样本只是用来**拴住**旧分布、提供一点正则化梯度，别让它们主导更新方向"；调大则更强调保留通用能力。若 MoE 架构返回了 `aux_loss`（负载均衡项），它会被加到 $\mathcal{L}$ 上但不计入 perplexity。

## 数据长什么样

CPT 消费的就是标准的 causal/SFT 形态 `Batch`（`trainall.types.Batch`），核心张量：

- `input_ids`：`(B, T)`，token id；
- `attention_mask`：`(B, T)`，通常全 1；
- `labels`：`(B, T)`，**CPT 一般等于 `input_ids`**（纯自监督，每个 token 都预测）。`-100` 的位置会被忽略——CPT 里很少用到，因为没有"prompt 不计损"这回事。

可选的 replay 加权信息放在 `batch.extra` 里（不是张量管线的一部分）：

- `batch.extra["domain"]`：长度 `B` 的布尔/0-1 列表，标记每条样本是 domain（真）还是 replay（假）。配合 `replay_weight` 使用；
- 或 `batch.extra["weights"]`：长度 `B` 的浮点列表，直接给每条样本的权重，优先级最高。

数据来源上，领域语料通常先用 `JsonlSource` / `HFDatasetSource` 加载原始文本再 tokenize+packing（见 `pack_sequences`），或者像下面的例子用 `InMemorySource` 喂预 tokenize 好的 `{"input_ids": [...], "labels": [...]}` 字典——这类预 tokenize 样本会直接透传给 Trainer 的默认 collate。

## 在 trainall 中怎么用

下面这段在 CPU 上即可跑通：先用 `compute_loss` 看一眼带 replay 加权的单步损失，再串一个 3 步的迷你训练循环。

```python
import torch
import trainall
from trainall.models import DecoderLM, ArchConfig
from trainall.types import Batch
from trainall.data import InMemorySource
from trainall.training import Trainer, TrainerConfig

# 1) 一个迷你 decoder-only LM（CPU 即可）
cfg = ArchConfig(vocab_size=64, dim=32, n_layers=2, n_heads=4,
                 n_kv_heads=2, ffn_dim=64, max_seq_len=64)
model = DecoderLM.from_config(cfg)

# 2) CPT 目标：replay 样本权重 0.1，domain 样本权重 1.0
obj = trainall.build("cpt", replay_weight=0.1)     # 等价 build("dapt", ...)
print("objective:", type(obj).__name__)

# 3) 直接看一次 loss：一条 domain 文档 + 一条 replay 文档混在同一 batch
ids = torch.randint(0, 64, (2, 16))
batch = Batch.of(input_ids=ids,
                 attention_mask=torch.ones_like(ids),
                 labels=ids.clone())               # 纯 next-token：预测每个下一个 token
batch.extra["domain"] = [1, 0]                     # 第0行=domain, 第1行=replay
loss, metrics = obj.compute_loss(model, batch)
print("loss:", round(float(loss.detach()), 4), "ppl:", round(metrics["ppl"], 2))

# 4) 串一个极小的训练循环（domain 语料预 tokenize 好）
toks = [torch.randint(0, 64, (16,)).tolist() for _ in range(8)]
data = InMemorySource([{"input_ids": t, "labels": t} for t in toks])
trainer = Trainer(model, obj, data=data,
                  config=TrainerConfig(device="cpu", max_steps=3, batch_size=4,
                                       lr=1e-3, bf16=False, log_every=1))
trainer.train()
print("done")
```

实际运行输出（节选）：

```
objective: ContinuedPretrainObjective
... step 1 | loss=4.1664 ppl=64.4851 ...
... step 3 | loss=4.0921 ppl=59.8627 ...
loss: 4.1398 ppl: 62.79
done
```

要点：`replay_weight=0.1` + `batch.extra["domain"]` 才会触发 batch 内加权；若两者缺一，CPT 自动退回与普通预训练逐位等价的 fast path。生产里更常见的是**物理 replay**（直接把原始分布样本按比例混进语料），加权旋钮则用于在同一 batch 内做更细的配比控制。

## 何时用 / 何时不用

**适合 CPT：**

- 目标领域与基座预训练分布**差异大**，且你**有大量无标注领域文本**（百万 token 级以上）——法律、医疗、金融、特定代码库、低资源语言。
- 你希望把**领域知识/术语/事实**注入权重，而不仅仅是教模型"回答的格式"。
- 计划之后还要做 [SFT](03-sft.md)：先 CPT 把领域底子打好，再 SFT 教交互格式，二者叠加通常优于直接 SFT。

**不适合 / 别上 CPT：**

- 你只想改**行为/风格/格式**（比如"用要点回答""遵循某模板"）——那是 [SFT](03-sft.md) 的活，CPT 教不了对齐，只教分布。
- 领域数据很少（几千条）——CPT 的收益来自规模化自监督，小数据上直接 SFT 更划算，强上 CPT 反而容易过拟合 + 遗忘。
- 你需要的是**对齐到人类偏好**——那要走 [偏好优化](04-preference-optimization.md) 或 [RLHF](05-rlhf.md)。

**为什么 CPT 通常排在 SFT 之前？** 训练管线的典型顺序是 *预训练 → CPT → SFT → 偏好优化*。CPT 是**自监督**、改的是模型对领域文本的**概率信念**（"知识层");SFT 是**有监督**、改的是**输入到输出的映射/行为**（"技能层"）。先用海量无标注领域文本把知识灌进权重，模型在做 SFT 时就能用更少的标注样本学会"怎么用"这些知识；反过来如果先 SFT 再 CPT，CPT 的自监督梯度会冲淡刚学到的指令跟随能力（又一次遗忘）。所以 CPT 在前、SFT 在后，是知识在前、技能在后的自然顺序。

## 常见陷阱与实践要点

- **遗忘是默认会发生的，不是偶发 bug。** 上线前务必在**通用 benchmark**（不只是领域指标）上回归评测；只盯领域 perplexity 下降会让你对通用能力的崩塌视而不见。
- **一定要配 replay。** 纯领域语料、零 replay，是遗忘最严重的配方。从 5%–15% 原始分布混入起步，按通用指标掉多少来调。`replay_weight` 旋钮可在 batch 内细调，但**物理混入**才是主力手段。
- **学习率要比初始预训练小**（常见低一个量级），warmup 短。你是在一个收敛点附近做迁移，不是从头训练；学习率过大 = 直接砸碎已有通用能力。
- **不要训练过头。** CPT 的边际收益随步数递减，而遗忘随步数累积。盯住"领域收益 vs 通用损失"的折中点，到点就停，别追求领域 perplexity 的最后一点下降。
- **labels 别误设成 `-100`。** CPT 是纯自监督，`labels` 应等于 `input_ids`（全员计损）。如果你从 SFT 管线复用 collator，注意别把 prompt 段错误地 mask 掉。
- **数据质量 > 数据量。** CPT 直接把语料的分布写进权重，脏数据（乱码、重复、PII、低质模板）会被忠实地学进去。领域语料的去重、清洗、去 PII 比通用预训练更要紧。
- **`replay_weight=0.0` 不等于"丢弃 replay"。** 它表示"不做 batch 内重加权"、走 fast path；真正控制 replay 占比的是你混进语料的比例。

## 相关

- [预训练 (Pre-training)](01-pretraining.md)——CPT 复用的同一个 next-token 目标，从这里开始。
- [SFT (Supervised Fine-Tuning)](03-sft.md)——CPT 之后教行为/格式的下一步。
- [偏好优化 (Preference Optimization)](04-preference-optimization.md) / [RLHF](05-rlhf.md)——再往后的对齐阶段。
- [LoRA / QLoRA](10-lora-qlora.md)——参数高效地做 CPT/SFT，省显存。
- [架构 (Architectures)](11-architectures.md)——`ArchConfig` / `DecoderLM` 的细节。
- 术语表：[CPT/DAPT](../GLOSSARY.md#cpt)、[catastrophic forgetting](../GLOSSARY.md#catastrophic-forgetting)、[replay](../GLOSSARY.md#replay)。
- 返回 [方法索引](README.md)。
