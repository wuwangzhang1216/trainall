<!-- nav -->
<table width="100%"><tr><td align="left" width="30%"><sub>&nbsp;</sub></td><td align="center" width="40%"><a href="README.md">📑 索引</a> · <a href="../GLOSSARY.md">📖 术语词典</a> · <a href="en/01-pretraining.md">🌐 English</a></td><td align="right" width="30%"><a href="02-continued-pretraining.md">持续预训练 CPT →</a></td></tr></table>
<!-- /nav -->

# 预训练 (Pre-training)

> **用海量无标注文本做"下一个 token 预测"，把整个世界的统计规律压进一组权重里——这就是大模型一切能力的地基。**

![自监督预训练：把文本切成 token，模型逐位预测下一个 token](../assets/pretraining.png)

## 直觉：它到底在做什么

想象一个填空游戏：给你一句话的开头 "巴黎是法国的"，让你猜下一个词。要猜得准，你不能只背语法，你得"知道"巴黎是首都、法国是个国家、"首都"这个概念存在。把这个游戏放大到整个互联网规模的文本、放大到每一句话的每一个位置，逼一个神经网络反复去玩——它为了把"下一个 token"猜得更准,被迫在内部建立起关于语言、事实、常识乃至浅层推理的隐式模型。

这就是预训练 (pre-training) 的全部秘密：**目标函数极其简单（预测下一个 token），但因为"预测得准"这件事本身需要理解世界，模型在追逐这个简单目标的过程中,被动地学会了一切。** 它是**自监督 (self-supervised)** 的——监督信号 (label) 不来自人工标注，而直接来自文本自身：第 $t$ 个 token 的"正确答案"就是文本里第 $t+1$ 个 token。因此可用的数据量等于人类写下的所有文字,而不受标注预算限制。这正是 scaling（规模化）得以成立的前提。

预训练产出的是一个 **base model（基座模型）**：它会"续写"而非"对话"，还不会听从指令。后续的 SFT、偏好优化、RLHF 都是在这个基座上做的精修——基座决定了能力上限,精修决定了能力如何被释放。

## 原理与架构（深度讲解）

### 数据 → 目标 → 算法 三段式

trainall 把每个训练范式拆成三件正交的事，预训练是最纯粹的一例：

- **数据 (data)**：一大堆 token id 序列。没有 prompt/response 之分，没有人类偏好，就是连续文本被分词后的 `input_ids`。在 trainall 里，纯预训练的 `labels` 直接等于 `input_ids`（每个位置都是监督目标）。
- **目标 (objective)**：`CausalLMObjective`（注册名 `pretrain`，别名 `clm`）。它只做一件事——对每个 token 位置计算"预测下一个 token"的交叉熵 (cross-entropy)，再对所有 token 取平均。
- **算法 (algorithm)**：通常是 `full`（全参数训练），因为预训练要塑造的是模型的全部知识，没有理由冻结任何参数。

这种解耦让你能把同一个目标函数套到不同算法、不同数据上而代码几乎不变。

### "自回归"与因果掩码：为什么能并行又不作弊

模型是一个 **decoder-only Transformer（仅解码器）**，自回归 (autoregressive) 地建模序列概率：

把一段文本看成 token 序列 $x = (x_1, x_2, \dots, x_T)$，整段的联合概率被链式法则分解为

$$ p_\theta(x) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{\lt t}) $$

每个条件概率 $p_\theta(x_t \mid x_{\lt t})$ 只依赖**左侧**的上下文。在实现上，这靠 **因果注意力掩码 (causal mask)** 保证：位置 $t$ 的注意力被禁止看到 $t$ 之后的任何 token。这是关键的工程巧思——掩码让模型可以**一次前向传播就并行算出所有 $T$ 个位置的预测**（训练高效），同时每个位置又"看不见未来"（不会作弊、不会信息泄漏）。这也是为什么训练时能 teacher forcing（喂入真实前缀），推理时却只能一个 token 一个 token 地生成。

关于注意力本身的实现细节（多头、GQA、RoPE、MoE 等），见架构文档 [11-architectures.md](11-architectures.md)。

### "预测下一个 token"为什么能涌现出知识与推理

很多人第一次听到"模型只是在预测下一个词"会觉得这太肤浅,不可能产生智能。但要理解它的威力，关键在于：**完美地压缩文本等价于理解文本。**

- **世界知识 (world knowledge)**：要在 "光速约为每秒 ___ 公里" 后面填对 "30 万"，模型必须把这个事实存进权重。海量语料里反复出现的事实，会被压进参数空间。
- **隐式推理 (latent reasoning)**：要续写 "如果今天是周三，那么三天后是 ___"，模型得在内部完成一次加法/查表。为了降低 loss，它学会了可复用的运算"电路"。
- **语用与风格**：续写需要保持语气、格式、人物一致，这逼模型建模长程依赖。

从信息论看，交叉熵损失正是用 nats（或 bits）度量的**压缩率**：loss 越低，模型对文本的预测越确定，等价于用更少的比特就能编码这段文本。Scaling laws（Kaplan 2020、Hoffmann 2022 "Chinchilla"）实证表明，loss 随参数量、数据量、算力以幂律平滑下降——正是这种"简单目标 + 规模"的可预测性，让预训练成为大模型的核心引擎。能力不是被显式设计的，而是在压缩压力下**涌现 (emerge)** 的。

理论根基可追溯到 Bengio 等 (2003) 的神经语言模型，以及 Radford 等 (2018) 的 GPT（"Improving Language Understanding by Generative Pre-Training"）。

## 目标函数（数学）

预训练最小化全体 token 的平均负对数似然 (negative log-likelihood, NLL)，即下一个 token 的交叉熵：

$$ \mathcal{L}(\theta) = -\frac{1}{N} \sum_{t} \mathbb{1}[y_t \neq -100] \cdot \log p_\theta(y_t \mid x_{\lt t}) $$

其中：

- $x_{\lt t}$ —— 第 $t$ 个位置之前的全部 token（左侧上下文 / 前缀）。
- $y_t$ —— 第 $t$ 个位置的监督目标，也就是"下一个 token"。在纯预训练里 $y_t = x_{t+1}$，因此代码中 `labels` 等于 `input_ids`（内部会做一次 **causal shift**，把 logits 对齐到下一位）。
- $p_\theta(y_t \mid x_{\lt t})$ —— 模型在该位置对词表做 softmax 后，分配给正确 token $y_t$ 的概率。
- $\mathbb{1}[y_t \neq -100]$ —— 掩码指示函数：标为 `-100` 的位置（如 padding）不计入损失。
- $N = \sum_t \mathbb{1}[y_t \neq -100]$ —— 实际参与计算的 token 总数，用作归一化分母。

单个位置的损失就是 softmax 后正确类别的 $-\log$ 概率：

$$ \ell_t = -\log \frac{\exp(z_{t,\,y_t})}{\sum_{v=1}^{|V|} \exp(z_{t,\,v})} $$

其中 $z_{t,v}$ 是位置 $t$ 上词表第 $v$ 个 token 的 logit，$|V|$ 是词表大小。

一个直观的衍生量是 **困惑度 (perplexity, PPL)**，即损失的指数：

$$ \mathrm{PPL} = \exp(\mathcal{L}) $$

可以理解为"模型在每个位置平均要在多少个等概率候选里犹豫"。PPL 越接近 1 越好；随机初始化时它约等于词表大小 $|V|$。在 trainall 里，`compute_loss` 返回的 `metrics` 字典就带着 `ppl` 这个键。

## 数据长什么样

`CausalLMObjective` 消费一个 `trainall.types.Batch`，预训练只用到三个张量：

- `input_ids` —— `(B, T)`，long 型 token id。
- `attention_mask` —— `(B, T)`，1 表示真实 token，0 表示 padding。
- `labels` —— `(B, T)`，监督目标。**纯预训练时 `labels == input_ids`**（每个 token 都被预测）；`-100` 的位置被忽略。

如果你不传 `labels`，目标函数会默认用 `input_ids` 当 `labels`，所以纯 LM 训练里两者本就相同。

喂给 `Trainer` 时，最简单的方式是 `InMemorySource`，每条记录是一个**已分词**的字典 `{"input_ids": [...], "labels": [...]}`，默认 collate 会把它们 pad 成上面那批张量。一条记录长这样：

```python
{"input_ids": [12, 7, 40, 3, 21, ...], "labels": [12, 7, 40, 3, 21, ...]}  # labels == input_ids
```

实践中，原始语料会先被**打包 (packing)**：用 `trainall.data.pack_sequences` 把许多短文档拼接、切成等长块，让每个块都被 token 填满，几乎不浪费算力在 padding 上。

## 在 trainall 中怎么用

下面是一个可在 CPU 上跑通的最小预训练循环：一个极小的 `DecoderLM`、一个 `InMemorySource`、`build("pretrain")` 拿到目标函数，跑 3 步。

```python
import torch, trainall
from trainall.models import DecoderLM, ArchConfig
from trainall.data import InMemorySource
from trainall.training import Trainer, TrainerConfig
from trainall.types import Batch

# 1) A tiny decoder-only LM (CPU-friendly).
cfg = ArchConfig(vocab_size=64, dim=32, n_layers=2, n_heads=4,
                 n_kv_heads=2, ffn_dim=64, max_seq_len=64)
model = DecoderLM.from_config(cfg)

# 2) Pre-tokenised corpus. For pure next-token pretraining, labels == input_ids:
#    every token is a supervised target predicted from its left context.
torch.manual_seed(0)
items = [{"input_ids": (ids := torch.randint(0, 64, (16,)).tolist()),
          "labels": list(ids)} for _ in range(8)]
data = InMemorySource(items)

# 3) The next-token cross-entropy objective ("clm" is an alias).
objective = trainall.build("pretrain", category="objective")

# 4) Train 3 CPU steps.
tcfg = TrainerConfig(lr=1e-3, batch_size=4, max_steps=3,
                     device="cpu", bf16=False, log_every=1)
trained = Trainer(model, objective, data=data, config=tcfg).train()

# 5) Inspect the loss directly via compute_loss on one batch.
ids = torch.randint(0, 64, (2, 16))
batch = Batch.of(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())
loss, metrics = objective.compute_loss(trained, batch)
print("loss:", round(float(loss.detach()), 4), "| ppl:", round(metrics["ppl"], 3))
```

实际运行输出（loss 随 3 步训练下降，最终在单 batch 上为有限值）：

```
step 1 | loss=4.1715 ppl=64.8107 n_tokens=60.0000
step 2 | loss=4.1567 ppl=63.8614 n_tokens=60.0000
step 3 | loss=4.0940 ppl=59.9802 n_tokens=60.0000
loss: 4.1197 | ppl: 61.541
```

随机初始化时 PPL 约等于词表大小（这里 64），与上面 ~60 的数值吻合——一个不错的"健全性检查 (sanity check)"。

## 何时用 / 何时不用

**适合用预训练的场景：**

- 你要**从零造一个基座模型**，且手握 token 数以十亿/万亿计的通用语料。
- 你想引入一种全新的语言、模态或符号体系，已有 tokenizer/模型完全没见过它。

**不适合（这时别用全量预训练）：**

- 只是想让模型适应某个**领域**（医疗、法律、代码）——用**继续预训练 (CPT/DAPT)**，见 [02-continued-pretraining.md](02-continued-pretraining.md)，它在已有基座上小步走，便宜得多。
- 想让模型**听懂指令、按格式回答**——那是 **SFT** 的活，见 [03-sft.md](03-sft.md)。预训练出的 base model 只会续写。
- 想对齐人类偏好或提升特定任务正确率——用偏好优化 [04-preference-optimization.md](04-preference-optimization.md) 或 RLVR [06-rlvr-grpo.md](06-rlvr-grpo.md)。
- 数据量小、算力有限：从头预训练几乎注定欠拟合，远不如拿开源基座做精修。

一句话：**预训练是"造地基"，绝大多数人需要的是"装修"。**

## 常见陷阱与实践要点

- **数据质量 ≫ 数据数量**：脏数据、重复文档会直接把噪声压进权重。去重 (dedup)、质量过滤、去除 PII 是标配。重复样本还会让模型"背诵"而非泛化。
- **务必做序列打包 (packing)**：不打包就会在 padding 上浪费大量算力，且短样本的梯度信号被稀释。用 `pack_sequences`。
- **label 别错位**：纯预训练 `labels == input_ids`；causal shift 由目标函数内部完成，**不要**自己再手动右移一位，否则会"预测当前 token"而非下一个，loss 异常低却学不到东西。
- **PPL 当健全性检查**：训练开始时 PPL 应接近词表大小；若一上来就远低于词表大小，多半是 label 泄漏（模型偷看到了答案）。
- **学习率与 warmup**：预训练对大学习率敏感，需要 warmup 平稳启动，否则早期梯度爆炸。`TrainerConfig.warmup_ratio` 默认 0.03。
- **数值精度**：大规模训练用 `bf16`；本文档的 CPU 小例子用 `bf16=False` 以保证可复现且无精度告警。
- **`-100` 才是忽略标记**：要跳过某些位置（如 padding），把对应 `labels` 设为 `-100`，而不是删除或置 0。
- **base model ≠ chat model**：直接拿预训练产物去对话会失望——它会续写你的问题而不是回答。务必接 SFT。

## 相关

- [继续预训练 / CPT-DAPT](02-continued-pretraining.md) —— 在已有基座上做领域适配，预训练的"轻量续作"。
- [SFT 监督微调](03-sft.md) —— 在基座上教模型听指令；本质是带 prompt 掩码的同一个交叉熵。
- [偏好优化](04-preference-optimization.md) 与 [RLHF](05-rlhf.md) —— 对齐人类偏好。
- [架构详解](11-architectures.md) —— decoder-only Transformer、注意力变体、RoPE、MoE。
- 词表 / 术语见 [术语表](../GLOSSARY.md)：[perplexity](../GLOSSARY.md#perplexity)、[cross-entropy](../GLOSSARY.md#cross-entropy)、[self-supervised](../GLOSSARY.md#self-supervised)。
- 返回 [方法总览 README](README.md)。
