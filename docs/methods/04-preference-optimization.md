<!-- nav -->
<p align="center">
  <a href="03-sft.md">← SFT</a> ·
  <a href="README.md">索引</a> ·
  <a href="../GLOSSARY.md">术语表</a> ·
  <a href="en/04-preference-optimization.md">English</a> ·
  <a href="05-rlhf.md">RLHF →</a>
</p>
<!-- /nav -->

# 离线偏好优化 (Offline Preference Optimization)

> **不训练奖励模型、不跑在线采样，直接把"成对偏好 (pairwise preference)"当作一个分类/回归目标，让策略 (policy) 自己变成隐式奖励模型 (implicit reward model)。**

![DPO 偏好优化：策略与参考模型对 chosen/rejected 的对数比之差驱动损失](../assets/dpo_preference.png)

## 直觉：它到底在做什么

经典 RLHF (见 [05-rlhf.md](05-rlhf.md)) 是三段式：先用偏好数据训一个奖励模型 (reward model)，再用 PPO 在线采样、用奖励信号做强化学习。这条链路又长又脆——奖励模型会被 reward hacking、PPO 调参困难、还要同时在显存里放下策略、参考、奖励、价值四个网络。

**离线偏好优化 (offline preference optimization)** 的核心洞察是：在 RLHF 那个"最大化奖励 + 受 KL 约束不要偏离参考模型太远"的最优解里，**奖励本身可以被反解 (reparameterize) 成策略相对参考模型的对数概率比**。也就是说，奖励不需要单独训练——它就藏在策略里。于是整个 PPO 阶段塌缩成一个**对一批固定的 (chosen, rejected) 数据做监督学习**的过程：没有采样、没有奖励模型、没有价值网络，只有一个普通的反向传播。

一句话直觉：给模型看一对回答，告诉它"这个好、那个差"，模型要做的就是**抬高好回答的概率、压低差回答的概率**——但不是无脑地抬/压，而是相对一个"出发点" (参考模型或长度归一化基线) 去抬/压，避免模型把整个分布推飞。本文覆盖的 6 个成员都是这个思想的不同实现：DPO、IPO、KTO、ORPO、SimPO、CPO。

## 原理与架构（深度讲解）

### 从 RLHF 的最优解到 DPO 的反解

RLHF 要解的优化问题是（$r$ 是奖励，$\pi_{\text{ref}}$ 是 SFT 后的参考模型，$\beta$ 是 KL 强度）：

$$\max_{\pi}\ \mathbb{E}_{x,\,y\sim\pi}\big[r(x,y)\big]\ -\ \beta\,\mathrm{KL}\big(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)$$

这个带 KL 正则的最大化有**闭式最优解**：

$$\pi^{*}(y\mid x)=\frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\Big(\tfrac{1}{\beta}r(x,y)\Big)$$

把它反解出 $r$：

$$r(x,y)=\beta\log\frac{\pi^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x)$$

关键在于：DPO (Rafailov et al. 2023) 用 **Bradley-Terry 偏好模型**（reward model 与 Bradley-Terry 见 [05-rlhf.md](05-rlhf.md)）把偏好概率写成 $P(y_c\succ y_r)=\sigma\big(r(x,y_c)-r(x,y_r)\big)$。**相减时配分函数 $Z(x)$ 被消掉了**——它只依赖 $x$，对 chosen/rejected 是同一个值。于是奖励里那个难算的归一化项消失，只剩下策略与参考的对数比之差。这就是为什么 DPO 不需要奖励模型：**奖励差 = 隐式奖励边际 (implicit reward margin)**，可以直接从策略前向算出来。

### 模型到底学到了什么

定义隐式奖励 $\hat r(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$。DPO 的梯度（推导自 sigmoid 损失）形如：

$$\nabla_\theta\mathcal{L}_{\text{DPO}}\ \propto\ -\,\underbrace{\sigma\!\big(\hat r_r-\hat r_c\big)}_{\text{“模型还认为 rejected 更好”的程度}}\ \Big[\nabla_\theta\log\pi_\theta(y_c)-\nabla_\theta\log\pi_\theta(y_r)\Big]$$

两点机制值得记住：

1. **自适应加权**：那个 $\sigma(\hat r_r - \hat r_c)$ 是一个软权重。当模型已经把 chosen 排在前面时权重趋近 0，几乎不更新；只有当模型**排错**（仍偏向 rejected）时梯度才大。这就是隐式的难例挖掘 (hard-example mining)。
2. **相对参考而非绝对**：更新的是对数比 $\log\frac{\pi_\theta}{\pi_{\text{ref}}}$，参考模型充当一个"锚"。这正是 KL 约束的体现——它阻止模型为了拉开 chosen/rejected 而把整个语言分布推飞（否则会退化、复读、丢失通用能力）。

### data → objective → algorithm 的拆分

在 trainall 的三段抽象里，偏好优化全部落在 **objective** 这一层：

- **data**：一批 `(prompt, chosen, rejected)` 三元组，离线、静态、可复用。没有采样器、没有 environment。这是"离线 (offline)"二字的全部含义。
- **objective**：本文的 6 个损失。它们的差异只在于**如何定义隐式奖励**和**如何把"chosen 应当胜出"写成可微的标量**。
- **algorithm**：`full` / `lora` / `qlora`（见 [10-lora-qlora.md](10-lora-qlora.md)）。偏好优化和参数高效微调正交——DPO + LoRA 是最常见的低成本对齐配方。

### 六个变体的设计张力

所有变体都在回答两个问题，而它们的取舍构成了这一族的全貌：

**问题一：要不要参考模型？** 参考模型提供 KL 锚定、防退化，但要在显存里多放一份冻结权重、多跑一次前向。DPO/IPO/KTO 保留它；ORPO/SimPO/CPO 用各自的"代理"消掉它（odds-ratio 自带正则 / 长度归一化 / NLL 锚），换取一半显存和更短的流水线。

**问题二：用什么 link 把边际变成损失？** DPO 用 logistic（sigmoid），对易分样本饱和；IPO 用平方损失，把边际**回归**到一个有限目标值，避免在确定性偏好上把边际推到无穷（DPO 的过拟合病灶）；KTO 干脆放弃成对假设，用前景理论 (prospect theory) 的效用函数处理**单条**带标签样本。

下面逐个给出数学。

## 目标函数（数学）

记 $\log\pi(y)=\sum_t\log\pi(y_t\mid y_{\lt t},x)$ 为序列对数概率（trainall 中由 `sequence_logps(..., average=False)` 计算）；$\overline{\log\pi}(y)$ 为其**长度归一化**版本（`average=True`，即除以 token 数）。$y_c,y_r$ 分别是 chosen / rejected。

### 比较表

| key | 隐式奖励 / 核心量 | link / 形式 | 需要参考模型? |
|---|---|---|---|
| `dpo` | $\beta\big(\log\tfrac{\pi}{\pi_{\text{ref}}}\big)$，求 chosen−rejected 边际 | logistic（可选 cDPO 平滑 / hinge） | 是 |
| `ipo` | 同 DPO，但用**长度归一化**对数比 | 平方损失，回归到 $\tfrac{1}{2\beta}$ | 是 |
| `kto` | $\log\pi-\log\pi_{\text{ref}}$ 相对 KL 基线 $z$ | 前景理论效用（**非成对**） | 是 |
| `orpo` | 几率比 $\log\tfrac{\text{odds}(y_c)}{\text{odds}(y_r)}$ | logistic + SFT，无参考 | 否 |
| `simpo` | $\beta\,\overline{\log\pi}$（**无参考**） | logistic，减去目标边际 $\gamma$ | 否 |
| `cpo` | $\beta\big(\log\pi(y_c)-\log\pi(y_r)\big)$ | logistic + NLL 锚，无参考 | 否 |

> 注意：在 trainall 的实现里 `kto` 的 `requires_reference_model = True`（它仍需要参考对数概率来构造对数比），这与某些"KTO 免参考"的口径不同——本文以仓库实现为准。

### DPO (Direct Preference Optimization)

令隐式奖励边际

$$\Delta=\Big(\log\pi_\theta(y_c)-\log\pi_{\text{ref}}(y_c)\Big)-\Big(\log\pi_\theta(y_r)-\log\pi_{\text{ref}}(y_r)\Big)$$

默认 **sigmoid** 损失（含 conservative-DPO 标签平滑 $\varepsilon$，$\varepsilon=0$ 即标准 DPO）：

$$\mathcal{L}_{\text{DPO}}=-(1-\varepsilon)\,\log\sigma(\beta\Delta)\;-\;\varepsilon\,\log\sigma(-\beta\Delta)$$

可选 **hinge** 损失（SLiC 风格）：$\ \mathcal{L}=\max\!\big(0,\ 1-\beta\Delta\big)$。

- $\beta$：温度，控制对参考模型的偏离强度（越小越保守）。典型 $0.1$。
- $\sigma$：logistic sigmoid；$\varepsilon$：cDPO 平滑系数，承认偏好标签有噪声、防止把边际推到无穷。

### IPO (Identity Preference Optimization)

用**长度归一化**对数比 $h=\big(\overline{\log\pi_\theta}(y_c)-\overline{\log\pi_{\text{ref}}}(y_c)\big)-\big(\overline{\log\pi_\theta}(y_r)-\overline{\log\pi_{\text{ref}}}(y_r)\big)$，损失为

$$\mathcal{L}_{\text{IPO}}=\Big(h-\tfrac{1}{2\beta}\Big)^{2}$$

平方损失把边际**回归**到有限目标 $\tfrac{1}{2\beta}$，根治 DPO 在确定性/无噪偏好上把 $\Delta\to\infty$ 的过拟合 (Azar et al. 2023)。

### KTO (Kahneman-Tversky Optimization)

放弃成对假设：每条样本单独带 desirable/undesirable 标签。令对数比 $r=\log\pi_\theta-\log\pi_{\text{ref}}$，共享 KL 基线 $z$（trainall 简化实现：$z=\mathrm{clip}_{\ge 0}\big(\overline{r}\big).\text{detach}()$）：

$$\mathcal{L}_{\text{KTO}}=
\begin{cases}
w_d\,\big(1-\sigma(\beta(r-z))\big), & \text{desirable}\\[4pt]
w_u\,\big(1-\sigma(\beta(z-r))\big), & \text{undesirable}
\end{cases}$$

$w_d,w_u$ 为期望/非期望权重，可用来纠正正负样本不均衡。形状取自前景理论：相对一个参照点 $z$ 衡量收益/损失 (Ethayarajh et al. 2024)。

### ORPO (Odds Ratio Preference Optimization)

**免参考**。令长度归一化对数概率 $\ell_c=\overline{\log\pi_\theta}(y_c),\ \ell_r=\overline{\log\pi_\theta}(y_r)$，几率比项

$$\log\text{OR}=\Big(\ell_c-\log\big(1-e^{\ell_c}\big)\Big)-\Big(\ell_r-\log\big(1-e^{\ell_r}\big)\Big)$$

总损失 = SFT 负对数似然 + 几率比惩罚：

$$\mathcal{L}_{\text{ORPO}}=\underbrace{-\,\overline{\log\pi_\theta}(y_c)}_{\text{SFT}}\;+\;\lambda\,\big(-\log\sigma(\log\text{OR})\big)$$

SFT 项让模型学会 chosen 的内容，几率比项把 rejected 推开；$\log(1-e^{\ell})$ 用数值稳定的 `log1mexp` 计算 (Hong et al. 2024)。

### SimPO (Simple Preference Optimization)

**免参考**，隐式奖励就是长度归一化平均对数概率 $r(y)=\beta\,\overline{\log\pi_\theta}(y)$，并引入目标边际 $\gamma$：

$$\mathcal{L}_{\text{SimPO}}=-\log\sigma\big(\beta\,\overline{\log\pi_\theta}(y_c)-\beta\,\overline{\log\pi_\theta}(y_r)-\gamma\big)$$

长度归一化天然消除 DPO 偏爱长回答的 length bias；$\gamma\gt 0$ 要求 chosen 至少领先一个安全边际 (Meng et al. 2024)。注意此处默认 $\beta=2.0,\ \gamma=0.5$。

### CPO (Contrastive Preference Optimization)

**免参考**，用 NLL 锚替代参考 KL。基于**未归一化**的序列对数概率：

$$\mathcal{L}_{\text{CPO}}=\underbrace{-\log\sigma\big(\beta(\log\pi_\theta(y_c)-\log\pi_\theta(y_r))\big)}_{\text{contrastive}}\;+\;\lambda\,\underbrace{\big(-\,\overline{\log\pi_\theta}(y_c)\big)}_{\text{NLL anchor}}$$

对比项拉开 chosen/rejected，NLL 锚（行为克隆）防止模型为了拉开边际而牺牲 chosen 的绝对似然 (Xu et al. 2024)。

## 数据长什么样

偏好目标消费一个**双侧** `Batch`（`trainall.types.Batch`），chosen 与 rejected 各一组张量：

```
chosen_input_ids       (B, T)   chosen 回答的 token id（含 prompt）
chosen_attention_mask  (B, T)
chosen_labels          (B, T)   -100 处忽略；prompt 段通常掩掉
rejected_input_ids     (B, T)
rejected_attention_mask(B, T)
rejected_labels        (B, T)
```

参考模型有两种供给方式（DPO/IPO/KTO 需要）：

- **在线计算**：把一份冻结的策略深拷贝放进 `batch.extra["ref_model"]`，目标会在 `torch.no_grad()` 下跑它一次前向。
- **预计算**：直接给 `batch.tensors["ref_chosen_logps"]` / `ref_rejected_logps`（一维 `(B,)`），跳过参考前向、省显存省算力。`dpo` 缺二者会 `raise ValueError`。

KTO 额外需要 `batch.extra["labels"]`：一个长度 `B` 的 bool 张量，`True == desirable`；它只用 `chosen_*` 字段承载每条样本，`rejected_*` 被忽略。

ORPO/SimPO/CPO 是 reference-free，**不需要** `ref_model`。

## 在 trainall 中怎么用

下面是一段可在 CPU 上直接跑通的最小示例：搭一个极小 `DecoderLM` 当策略、深拷贝一份冻结的参考放进 `batch.extra`，构造一个 tiny 偏好 `Batch`，分别跑 `dpo`（需参考）与 `simpo`（免参考）的 `compute_loss`。

```python
import copy, torch
import trainall
from trainall.models import DecoderLM, ArchConfig
from trainall.types import Batch

torch.manual_seed(0)

# 1) 一个极小的策略模型 (policy)，CPU 即可
cfg = ArchConfig(vocab_size=37, dim=16, n_layers=2, n_heads=4,
                 n_kv_heads=2, ffn_dim=32, max_seq_len=32)
policy = DecoderLM.from_config(cfg)

# 2) 冻结的参考模型 (reference) = 策略的深拷贝
ref = copy.deepcopy(policy).eval()
for p in ref.parameters():
    p.requires_grad_(False)

# 3) 一个偏好 Batch：chosen_* 与 rejected_* 两侧
def ids(b=3, t=6, v=37): return torch.randint(0, v, (b, t))
cids, rids = ids(), ids()
batch = Batch(tensors=dict(
    chosen_input_ids=cids,   chosen_attention_mask=torch.ones_like(cids),
    chosen_labels=cids.clone(),
    rejected_input_ids=rids, rejected_attention_mask=torch.ones_like(rids),
    rejected_labels=rids.clone(),
))
batch.extra["ref_model"] = ref          # 参考模型放进 extra（DPO/IPO/KTO 需要）

# 4) 构建 DPO 目标并计算损失
dpo = trainall.build("dpo", beta=0.1)    # 也可 category="objective"
loss, metrics = dpo.compute_loss(policy, batch)
loss.backward()                          # 梯度流入 policy
print("DPO   loss =", metrics["loss"], "acc =", metrics["reward_acc"])

# 5) reference-free 变体（SimPO）：同一个 batch，无需 ref_model
simpo = trainall.build("simpo", beta=2.0, gamma=0.5)
loss2, m2 = simpo.compute_loss(policy, batch)
print("SimPO loss =", m2["loss"], "margin =", round(m2["reward_margin"], 4))
assert torch.isfinite(loss) and torch.isfinite(loss2)
```

实跑输出（`PYTHONPATH=src python3`）：

```
DPO   loss = 0.6931471824645996 acc = 0.0
SimPO loss = 0.9152031540870667 margin = 0.0973
```

DPO 初始 loss 恰为 $-\log\sigma(0)=\log 2\approx 0.6931$：因为 policy 与 ref 是同一份权重，边际 $\Delta=0$，符合预期。要落到完整训练，把目标交给 `Trainer`：`Trainer(model, objective=dpo, algorithm=trainall.build("lora", r=8), data=...).train()`，参考模型的注入由偏好配方负责。其它 key 替换 `build("ipo"/"kto"/"orpo"/"cpo")` 即可（KTO 记得再塞 `batch.extra["labels"]`）。

## 何时用 / 何时不用

**适合：**
- 你已经有静态的成对偏好数据（人类标注或 LLM-as-judge 产出），想要稳定、低成本地对齐——**DPO 是默认首选**。
- 显存或流水线吃紧、不想维护参考模型：用 **SimPO / ORPO / CPO**（reference-free）。ORPO 还能把 SFT 与对齐合成一步，省掉单独的 SFT 阶段。
- 偏好标签有噪声、或担心 DPO 过拟合确定性偏好：用 **IPO**（平方损失更温和）。
- 数据天然是"单条好/坏"而非成对（如点赞/点踩日志、人工审核通过/拒绝）：用 **KTO**，它不要求配对。

**不适合：**
- 你需要从**可验证奖励**（数学对错、单测通过、SQL 执行结果）学习：那是 RLVR 的领地，用 GRPO（见 [06-rlvr-grpo.md](06-rlvr-grpo.md)），离线偏好对此无能为力。
- 你需要在线探索 / 多步交互 / 工具调用：用在线 RL（[05-rlhf.md](05-rlhf.md) PPO、[07-agentic-rl.md](07-agentic-rl.md)）。离线方法只能在固定数据分布内重排概率。
- 偏好数据极少或质量极差：离线方法会忠实地拟合噪声，"garbage in, garbage out"比在线 RL 更直接。

## 常见陷阱与实践要点

- **参考模型必须是 SFT 后的同一权重**：DPO/IPO/KTO 的 KL 锚定假设 $\pi_{\text{ref}}=\pi_{\text{sft}}$。拿一个未经 SFT 的底座当参考会让对数比失去意义。先 SFT（[03-sft.md](03-sft.md)）再 DPO。
- **prompt 段要在 labels 里掩成 -100**：偏好损失基于回答的序列对数概率，把 prompt 算进去会污染信号（本文 tiny 示例为简洁未掩，真实数据务必掩）。
- **DPO 的隐式 length bias**：sigmoid + 未归一化对数概率会让模型偏爱更长的 chosen。若观察到回答变长、变啰嗦，换 **SimPO**（长度归一化）或在 DPO 里改用平均对数概率。
- **$\beta$ 不是越大越好**：$\beta$ 太大 → 强烈偏离参考、可能退化/复读；太小 → 几乎不更新。从 $0.1$ 起调；SimPO 的 $\beta$ 量纲不同（默认 $2.0$），不要照搬 DPO 的值。
- **`reward_acc` 是最有用的早期信号**：它是边际为正的样本比例（模型把 chosen 排在前的比例）。健康训练应从 ~0.5 稳步上升；若停在 0.5 附近，多半是 prompt 没掩、参考模型不对、或学习率过小。
- **预计算参考对数概率省一半算力**：参考模型对固定数据的输出是常量，可离线算好存进 `ref_chosen_logps`/`ref_rejected_logps`，训练时跳过参考前向。
- **过拟合确定性偏好**：若你的偏好对几乎总是单边压倒（margin 极大），DPO 会把边际推向无穷、损害校准。换 **IPO** 或加 cDPO 标签平滑 $\varepsilon\in(0,0.5)$。

## 相关

- 上游：[SFT (Supervised Fine-Tuning)](03-sft.md) —— 偏好优化之前的必经一步，提供参考模型。
- 对照：[RLHF / PPO](05-rlhf.md) —— 在线、带显式奖励模型的对齐范式；偏好优化是它的离线简化。
- 进阶：[RLVR / GRPO](06-rlvr-grpo.md) —— 用可验证奖励替代偏好；[Agentic RL](07-agentic-rl.md) —— 多步交互对齐。
- 配方：[LoRA / QLoRA](10-lora-qlora.md) —— 与偏好优化正交、最常见的低成本组合。
- 词表：[DPO](../GLOSSARY.md#dpo) · [Reward Model](../GLOSSARY.md#reward-model) · [Bradley-Terry](../GLOSSARY.md#bradley-terry) · 返回 [方法索引](README.md)。
