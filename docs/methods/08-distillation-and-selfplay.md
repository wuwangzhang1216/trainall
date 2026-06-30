<!-- nav -->
<p align="center">
  <a href="07-agentic-rl.md">← Agentic RL</a> ·
  <a href="README.md">索引</a> ·
  <a href="../GLOSSARY.md">术语表</a> ·
  <a href="en/08-distillation-and-selfplay.md">English</a> ·
  <a href="09-process-supervision.md">过程监督 →</a>
</p>
<!-- /nav -->

# 蒸馏与数据飞轮 (Distillation & the Synthetic-Data Flywheel)

> **用一个更强的"老师"或一个廉价的"裁判"去制造监督信号，让模型从它自己（或别人）能被验证的成功里反复学习——这就是蒸馏与数据飞轮。**

![蒸馏数据飞轮：propose → solve → verify → keep，配合 KD 与课程难度调节](../assets/distill_flywheel.png)

## 直觉：它到底在做什么

监督微调 (SFT) 的瓶颈从来不是算法，而是**数据**：高质量的"问题—答案"对又贵又少。本章讲的是两条绕开人工标注、让模型自己造数据/自己长本事的路线，它们常常组合成一个**数据飞轮 (data flywheel)**：

1. **知识蒸馏 (knowledge distillation, KD)**：有一个更强的 teacher 模型。与其只告诉 student "正确答案是 token #42"，不如把 teacher 在每个位置上的**完整概率分布**（soft targets，软标签）交给 student 去模仿。软标签里藏着 teacher 的"暗知识 (dark knowledge)"——它认为 42 之外哪些 token 也合理、哪些绝不可能。这比 one-hot 的硬标签信息量大得多。

2. **可验证数据飞轮 (verifiable data flywheel)**：你没有更强的 teacher，但你有一个**便宜的裁判 (verifier)**——一段能判对错的代码（数学答案比对、单元测试、JSON schema 校验）。于是让模型自己**采样很多答案**，用裁判把对的留下、错的扔掉，留下的对答案就成了新的 SFT 数据。模型在"自己偶尔做对的题"上微调，下一轮就更常做对——这就是飞轮转起来的样子。

这套思路有三个递进的工程化形态，trainall 里各对应一个组件：

- **Rejection Sampling（拒绝采样 / best-of-N）**：题目固定，对每题采 N 个答案，留下裁判通过的。
- **SyntheticDataEngine（合成数据引擎）**：连题目都让 proposer 自己出，形成 `propose → solve → verify → keep` 的完整闭环。
- **SelfPlayLoop（自我对弈循环）+ Curriculum（课程）**：多轮迭代，并根据当前通过率**自动调节题目难度**，把模型一直保持在"够得着但不轻松"的学习区。

一句话区分：**蒸馏是"向更聪明的人抄"，飞轮是"在能判分的题上反复自练"。** 二者经常串联——飞轮造出干净数据，再用 SFT 或 KD 喂回模型。

## 原理与架构（深度讲解）

### 蒸馏：为什么软标签比硬标签强

考虑一个分类（或下一个 token 预测）问题。硬标签的交叉熵 (cross-entropy, CE) 只约束**正确类别**的概率往 1 推，对其余所有错误类别一视同仁地往 0 压。但 teacher 知道更细的结构：在"猫 vs 狗 vs 汽车"里，把猫错认成狗远比错认成汽车合理。Teacher 输出的概率分布 $p_{\text{teacher}}=(0.9, 0.09, 0.01)$ 把这种**类间相似度结构**编码了进去，Hinton 等人 (2015) 称之为暗知识。

为了让这些"非最大概率"的信息显现出来，蒸馏引入**温度 (temperature)** $T$：把 logits 除以 $T$ 再 softmax。$T\gt 1$ 会"软化"分布，放大那些原本接近 0 的小概率之间的相对差异，让 student 能学到它们。Student 用**同样的** $T$ 去匹配 teacher 的软分布。训练完成后推理时 $T=1$。

关键细节：因为软标签的梯度量级随 $1/T^2$ 缩小，蒸馏损失要乘回 $T^2$，使其梯度量级与硬-CE 项可比、便于二者加权混合。这正是 trainall `DistillObjective` 里 `kd = (T*T) * masked_mean(kl_tok, mask)` 的来历。

**Forward vs Reverse KL（前向 / 反向 KL）——一个被低估的选择**：

- **Forward KL** $\mathrm{KL}(p_{\text{teacher}}\Vert p_{\text{student}})$ 是 **mass-covering（覆盖均值）** 的：凡是 teacher 给了概率的地方，student 都被迫覆盖，否则 $\log$ 项爆炸。结果是 student 倾向于"摊平"以覆盖 teacher 的所有模式，是经典离线蒸馏的默认选择。
- **Reverse KL** $\mathrm{KL}(p_{\text{student}}\Vert p_{\text{teacher}})$ 是 **mode-seeking（寻找众数）** 的：student 只要把自己的概率质量放在 teacher 也认可的地方就行，倾向于**锁定 teacher 的某个主模式**而忽略长尾。近年的 on-policy / 序列级蒸馏（如 MiniLLM）发现，对生成式语言模型，reverse KL 往往能避免 forward KL 那种"在 teacher 不会真正生成的区域也分配概率、导致语无伦次"的问题。

trainall 用 `kind="forward"`（默认）/ `kind="reverse"` 切换，并允许 `alpha` 在 KD 项与硬-CE 锚之间插值：`total = alpha*KD + (1-alpha)*CE`。保留一点 CE（`alpha<1`）等于给 student 一个"别忘了真正的 ground truth"的锚，常比纯蒸馏更稳。

### 飞轮：data → objective → algorithm 的视角

用本仓库一贯的三段式框架看飞轮：

- **data（数据从哪来）**：不再来自人工标注，而来自 `proposer/solver/verifier` 三个可调用对象的乘积。`proposer` 出题，`solver`（通常就是当前模型）采 N 个答案，`verifier` 判分。**验证比生成便宜**是整个范式成立的前提——数学题答案一比对就知道对错，代码跑一遍测试就知道，但生成正确解很难。这种"易验证、难生成"的非对称性 (asymmetry) 是飞轮的能量来源。
- **objective（学什么）**：留下来的 `(prompt, response)` 是**已被验证为正确**的轨迹，喂给标准 SFT 目标即可；若有 teacher logits 则可走 KD。模型实际学到的是"把自己偶尔做对的推理路径变成稳定行为"。
- **algorithm（怎么更新）**：full / LoRA / QLoRA 都行，飞轮只负责造数据，与参数更新算法正交。

这正是 **STaR (Zelikman et al., 2022)** 和 **拒绝采样微调 (Rejection-sampling Fine-Tuning, RFT)** 的核心：模型从自己能被验证的成功里 bootstrap 出训练集。它和 RLVR（见 [RLVR / GRPO](06-rlvr-grpo.md)）共享同一个可验证奖励信号，区别在于：飞轮把信号变成**离线 SFT 数据**（简单、稳定、可复用），而 RLVR 把信号变成**在线策略梯度**（样本效率更高但更难调）。很多团队两者并用：先飞轮冷启动，再 RLVR 精修。

### Best-of-N：为什么"只学成功"有效

对一个 base 模型，单次采样做对一道难题的概率 $p$ 可能很低；但采 N 次、至少做对一次的概率是 $1-(1-p)^N$，随 N 迅速上升。Rejection sampling 正是利用这一点：**用推理期的多次采样换取一条正确轨迹**，再把它蒸成训练数据。`keep="best"` 保留奖励最高的那条，`keep="all"` 保留所有通过的（数据更多但更同质），`keep="first"` 最省算力。注意只学正确轨迹会引入**幸存者偏差**——模型只见成功不见失败，因此飞轮通常配合一点原始数据回放，或交给 RLVR 去显式利用负样本。

### 课程与反崩溃：让飞轮持续转动

飞轮最危险的失效模式是**分布崩溃 (distribution collapse)**：proposer 反复出几乎一样的题，solver 反复给几乎一样的答案，数据多样性塌缩，模型在一小撮模式上过拟合、整体能力反而退化（即所谓 model collapse / 模式坍塌）。`Curriculum` 用两个机制对抗它：

1. **自适应难度（最近发展区, zone of proximal development）**：观察每轮通过率 `pass_rate`。高于 `target_high`（默认 0.8）说明题太简单 → 难度 `+step`；低于 `target_low`（默认 0.4）说明题太难、几乎没有正确轨迹可学 → 难度 `-step`；落在中间 → `hold`。把模型一直钉在"够得着但不轻松"的区间，是产出有效梯度的关键——题太易学不到新东西，题太难没有通过样本。
2. **多样性监控（anti-collapse）**：每轮统计 prompt 的唯一比例 `diversity = #unique / #total`，低于 `min_diversity` 就在 `history` 里记一个 `collapsed` 警告，提示你该给 proposer 加噪声、扩主题或注入外部种子。

`SelfPlayLoop` 把这些串起来：每轮按当前难度出 `tasks_per_round` 道题，每题采 `k` 个候选，验证、去重、保留，最后用本轮通过率 `update` 课程，进入下一轮。

## 目标函数（数学）

**蒸馏损失（`DistillObjective`）。** 设 student / teacher 在某 token 位置的 logits 为 $z^s, z^t$，温度 $T$。软分布为

$$
p^s_i = \frac{\exp(z^s_i / T)}{\sum_j \exp(z^s_j / T)}, \qquad
p^t_i = \frac{\exp(z^t_i / T)}{\sum_j \exp(z^t_j / T)}.
$$

KD 项是温度缩放后的 KL，再乘 $T^2$ 还原梯度量级：

$$
\mathcal{L}_{\text{KD}} = T^2 \cdot \mathrm{KL}\!\left(p^t \,\Vert\, p^s\right)
= T^2 \sum_i p^t_i \big(\log p^t_i - \log p^s_i\big) \quad (\text{forward}),
$$

或 reverse 变体（交换角色、寻找众数）：

$$
\mathcal{L}_{\text{KD}}^{\text{rev}} = T^2 \sum_i p^s_i \big(\log p^s_i - \log p^t_i\big) \quad (\text{reverse}).
$$

与硬标签 CE 混合（$y$ 为 ground-truth token，$q^s$ 为 $T=1$ 的 student 概率）：

$$
\mathcal{L} = \alpha\, \mathcal{L}_{\text{KD}} + (1-\alpha)\, \underbrace{\big(-\log q^s_{y}\big)}_{\mathcal{L}_{\text{CE}}}.
$$

符号说明：$T$ 温度（$T\gt 1$ 软化分布）；$\alpha\in[0,1]$ KD 与 CE 的权重（$\alpha=1$ 纯蒸馏，$\alpha=0$ 纯监督）；$\mathrm{KL}$ 在每个 token 上算、再按 `response_mask`（默认 attention mask）做 masked 平均，只蒸馏回答区域；$T^2$ 抵消温度对梯度量级的 $1/T^2$ 缩放。

**飞轮的"目标"——一个数据过滤算子。** 飞轮本身不是可微损失，而是一个保留算子。给定题 $x$、参考答案 $r$、裁判 $V$（$V(y,r)\in\{0,1\}$ 通过与否），从 solver $\pi$ 采 $N$ 个答案，保留集为

$$
\mathcal{D}_x = \big\{\, y^{(i)} \;:\; y^{(i)} \sim \pi(\cdot\mid x),\; V(y^{(i)}, r)=1,\; i=1\dots N \,\big\},
$$

它们随后进入 SFT/KD 目标。每题通过率 $\hat p_x = \tfrac{1}{N}\sum_i V(y^{(i)},r)$ 既用于打难度标签，也用于驱动课程：

$$
d \leftarrow
\begin{cases}
\min(1,\; d + \text{step}), & \bar p \gt  \text{target\_high} \\
\max(0,\; d - \text{step}), & \bar p \lt  \text{target\_low} \\
d, & \text{otherwise}
\end{cases}
\qquad \bar p = \frac{\sum_{\text{round}} \mathbf{1}[\text{pass}]}{\#\text{candidates}}.
$$

## 数据长什么样

飞轮组件吃**纯 Python 可调用对象**，不依赖 torch，完全可测：

- `proposer() -> task`：返回一道题。`task` 可以是 `str`、`(prompt, reference)` 元组、`{"prompt": ..., "reference": ...}` 字典，或 `Sample`。
- `solver(prompt) -> response | [responses]`：返回 1 个或多个候选答案（字符串）。
- `verifier(response, reference) -> VerifierResult | bool | float`：判分；trainall 的 `Verifier`（如 `MathVerifier`）也直接可用，内部会调 `.verify(response, reference, prompt=...)`。

输出统一是 `trainall.types.Sample` 列表，每个带 `prompt / response / reference / meta`，`meta` 里记录 `pass_rate`、`difficulty`、来源标志（`synthetic` / `rejection_sampled` / `self_play`）等。这些 `Sample` 可直接进 `InMemorySource` → SFT（见 [SFT](03-sft.md)）。

`DistillObjective` 吃的是张量 `Batch`（`trainall.types.Batch`）：

- `input_ids` `(B, T)`、`attention_mask` `(B, T)`，
- `labels` `(B, T)`（`-100` 表示忽略，给硬-CE 锚用；`alpha=1` 时可省略），
- **关键**：`batch.extra["teacher_logits"]` 形状 `(B, T, V)`，由一个冻结的 teacher 前向得到，
- 可选 `response_mask` `(B, T)` 指定只蒸馏回答区域（默认退化为 attention mask）。

## 在 trainall 中怎么用

下面是已实际运行通过的最小例子，全部 CPU、无需模型即可跑通飞轮三件套（`MathVerifier` 用 `\boxed{}` 答案比对）：

```python
from trainall.data import (
    RejectionSampler, SyntheticDataEngine, SelfPlayLoop, Curriculum, TaskProposer,
)
from trainall.verifiers import MathVerifier

# solver：解 "add a+b"，故意给两个对、一个错，模拟随机采样
def solver(prompt):
    a, b = prompt.replace("add ", "").split("+")
    s = int(a) + int(b)
    return [rf"\boxed{{{s}}}", r"\boxed{0}", rf"\boxed{{{s}}}"]

# 1) Rejection sampling：best-of-N，留下裁判通过的轨迹
rs = RejectionSampler(solver, MathVerifier(), n=3, keep="all")
kept = rs.run([{"prompt": "add 19+23", "reference": "42"}])
print("[RS] kept", len(kept), "| resp =", kept[0].response,
      "| pass_rate =", kept[0].meta["pass_rate"])

# 2) SyntheticDataEngine：proposer 自己出题，propose -> solve -> verify -> keep
counter = {"i": 0}
def proposer():
    counter["i"] += 1
    n = counter["i"]
    return {"prompt": f"add {n}+{n}", "reference": str(2 * n)}

engine = SyntheticDataEngine(proposer, solver, MathVerifier(), k=2, keep_per_task="first")
syn = engine.generate(3)
print("[SDE] generated", len(syn),
      "| difficulties =", [s.meta["difficulty"] for s in syn])

# 3) SelfPlayLoop + Curriculum：多轮迭代，按通过率自动调难度
loop = SelfPlayLoop(
    TaskProposer(lambda difficulty=0.5: {"prompt": "add 1+1", "reference": "2"}),
    lambda p: r"\boxed{2}",
    MathVerifier(),
    curriculum=Curriculum(difficulty=0.2, step=0.1),
    rounds=2, tasks_per_round=2, k=2,
)
sp = loop.run()
print("[SP] retained", len(sp),
      "| decisions =", [h["decision"] for h in loop.curriculum.history],
      "| final difficulty =", loop.curriculum.difficulty)
```

实际输出：

```
[RS] kept 2 | resp = \boxed{42} | pass_rate = 0.6666666666666666
[SDE] generated 3 | difficulties = ['medium', 'medium', 'medium']
[SP] retained 1 | decisions = ['harder', 'harder'] | final difficulty = 0.4
```

把 `kept` / `syn` / `sp` 里的 `Sample` 丢进 `InMemorySource` 就能接 `Trainer` 做 SFT。

蒸馏侧用 `DistillObjective`（需要 torch + teacher logits），已实测可前向 + 反向：

```python
import torch, trainall
from trainall.types import Batch
from trainall.models import DecoderLM, ArchConfig

cfg = ArchConfig(vocab_size=64, dim=32, n_layers=2, n_heads=4,
                 n_kv_heads=2, ffn_dim=64, max_seq_len=64)
student = DecoderLM.from_config(cfg)

obj = trainall.build("distill", category="objective", alpha=0.5, temperature=2.0, kind="forward")
ids = torch.randint(0, cfg.vocab_size, (2, 6))
batch = Batch.of(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone())
batch.extra["teacher_logits"] = torch.randn(2, 6, cfg.vocab_size)   # (B, T, V) 冻结 teacher
loss, metrics = obj.compute_loss(student, batch)
loss.backward()
print(f"[KD] loss={float(loss.detach()):.4f}  kd={metrics['kd']:.4f}  ce={metrics['ce']:.4f}")
# -> [KD] loss=2.3006  kd=0.4735  ce=4.1278   (随机种子会变)
```

真实场景里 `teacher_logits` 来自一个更大的冻结模型的前向；用 `kind="reverse"`、`alpha=1.0` 即切到纯反向-KL 的 on-policy 蒸馏。

## 何时用 / 何时不用

**用蒸馏当**：你有一个更强的 teacher（更大模型、ensemble、或带工具/CoT 的强 pipeline），想把它的能力压进一个更小/更快的 student；或想做模型压缩、加速推理、把多个专家合一。

**用飞轮当**：你有**可程序化验证**的任务（数学、代码、SQL、结构化输出），但缺人工标注；想低成本冷启动 SFT 数据；或想在 RLVR 之前先用稳定的离线数据把模型抬到一个合理起点。

**不要用蒸馏当**：没有比当前模型更强的 teacher（向同水平模型蒸馏只会传播其错误，得不偿失）；或 teacher 与 student 的 tokenizer / 词表不一致（logits 对不齐，需要序列级而非 token 级蒸馏）。

**不要用飞轮当**：任务**无法廉价验证**（开放式写作、主观偏好——这时该走偏好优化，见 [偏好优化](04-preference-optimization.md)）；裁判本身有系统性误判（会把错的当对的固化进数据）；或 proposer 多样性不足，几轮就崩塌。

## 常见陷阱与实践要点

- **裁判即上限**：飞轮数据质量被 verifier 的精度死死卡住。verifier 的假阳性 (false positive) 会把错误答案当正例固化，比缺数据更糟。上线前务必单测裁判（见 [verifiers 测试](../GLOSSARY.md#verifier)），宁可严格（漏掉真正例）也别宽松（放进假正例）。
- **分布崩溃要主动监控**：盯住 `Curriculum.history` 里的 `diversity` 和 `collapsed`。多样性一旦下滑，就给 proposer 加扰动、扩主题、混入外部种子题，或限制连续自训轮数。纯靠模型自产数据无限迭代几乎必然退化。
- **温度别和推理混淆**：蒸馏的 $T$ 只在训练时用来软化软标签，student 学的是 $T$ 缩放下的分布；推理时用 $T=1$。忘了乘 $T^2$ 会让 KD 项在大 $T$ 下梯度过小、几乎不更新。
- **保留一点硬-CE 锚**：纯蒸馏（`alpha=1`）容易让 student 漂向 teacher 的系统性偏差；混入 `(1-alpha)` 的 CE 把它锚回真实标签，通常更稳。
- **best-of-N 的幸存者偏差**：只学正确轨迹意味着模型从不见到失败模式。要么补一点原始混合数据，要么把负样本留给 RLVR 显式利用。
- **去重但别去多样性**：`dedup=True` 去掉完全相同的 `(prompt, response)` 是对的，但若你的 solver 总输出同一条正确解，留下的样本仍然同质——多样性靠采样温度和 proposer，而非去重。
- **N / k 是算力-质量旋钮**：N 越大、做对越多、数据越干净，但推理成本线性上涨。难任务调大 N，简单任务调小，或用课程难度间接控制 N 的有效收益。

## 相关

- [SFT](03-sft.md) — 飞轮产出的 `Sample` 最终通过它学习。
- [RLVR / GRPO](06-rlvr-grpo.md) — 与飞轮共享可验证奖励信号的在线对照路线。
- [Agentic RL](07-agentic-rl.md) — 多步环境里的自我对弈与轨迹收集。
- [过程监督 / PRM](09-process-supervision.md) — 用更细粒度的步骤奖励替代终局裁判。
- [偏好优化](04-preference-optimization.md) — 无法程序化验证时的替代方案。
- [LoRA / QLoRA](10-lora-qlora.md) — 在蒸馏/飞轮数据上做参数高效微调。
- 词表：[distill](../GLOSSARY.md#distill)、[verifier](../GLOSSARY.md#verifier)、[rejection-sampling](../GLOSSARY.md#rejection-sampling)、[self-play](../GLOSSARY.md#self-play)、[curriculum](../GLOSSARY.md#curriculum)。
- 返回总览：[README](README.md)。
