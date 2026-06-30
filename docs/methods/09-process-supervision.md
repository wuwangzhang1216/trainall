<!-- nav -->
<table width="100%"><tr><td align="left" width="30%"><a href="08-distillation-and-selfplay.md">← 蒸馏与自博弈</a></td><td align="center" width="40%"><a href="README.md">📑 索引</a> · <a href="../GLOSSARY.md">📖 术语词典</a> · <a href="en/09-process-supervision.md">🌐 English</a></td><td align="right" width="30%"><a href="10-lora-qlora.md">LoRA / QLoRA →</a></td></tr></table>
<!-- /nav -->

# 过程监督 / 过程奖励模型 (Process Supervision / PRM)

> **不要只问"答案对不对"，而要逐步问"这一步推理对不对"——把监督信号从一句话的结果细化到推理链 (chain-of-thought) 的每一步。**

![过程奖励模型在推理链每一步打分，而结果监督只看最终答案](../assets/prm_process.png)

## 直觉：它到底在做什么

设想一道数学题，模型写出 5 步推理后给出答案。**结果监督 (outcome supervision, ORM)** 只看最后那个答案对不对，给整条链一个标量奖励——对就是 1，错就是 0。但这里有个致命的问题：一条链可能**第 3 步算错了，却在第 4 步又"歪打正着"地凑回了正确答案**。结果监督会把这条逻辑混乱的链标成"好",从而奖励了一种错误的推理过程；反过来，一条**前 4 步都完全正确、只在最后一步抄错数字**的链会被整体判负，把大量正确的推理也一起惩罚掉。结果监督的奖励信号既**含噪 (noisy)**，又**稀疏 (sparse)**——它无法告诉模型"错在哪里"。

**过程监督 (process supervision)** 换了一种问法：对推理链的**每一步**单独打一个标签——这一步正确吗 (1) / 错误吗 (0)？训练出来的 **过程奖励模型 (Process Reward Model, PRM)** 因此能在链条**中途**就指出第一个出错的步骤，而不必等到看完答案。这正是 OpenAI 在 *Let's Verify Step by Step* (Lightman et al., 2023) 中的核心发现：在 MATH 数据集上，用过程监督训练的验证器 (verifier) 在 best-of-N 搜索中显著优于结果监督的验证器——因为它给出的信号**更密、更准、更可定位**。

一句话：ORM 给整篇作文打一个总分；PRM 是一位逐句批改、在每一行旁边画对勾或叉的老师。

## 原理与架构（深度讲解）

**数据 → 目标 → 算法** 三段式来看 PRM：

**1. 数据：把"轨迹"切成"带标签的步骤序列"。**
PRM 的训练数据不是 `(prompt, answer, 对/错)`，而是把一条推理链按分隔符（典型是换行 `\n`、`Step k:`、或专门的步骤 token）切成若干步，每一步配一个标签 ∈ {正确, 错误, 中立}。在 trainall 的 `Batch` 里，这体现为两个对齐到 token 位置的张量：

- `step_mask` (B, T)：布尔张量，`True` 的位置就是"步骤结束/分隔符"所在的 token 位置——只有这些位置参与损失计算。
- `step_labels` (B, T)：在 `step_mask` 为 `True` 的位置给出 0/1 标签（这一步对不对）。

这些步骤标签从哪来？三条主流路径：(a) **人工标注**（PRM800K 数据集，OpenAI 雇人逐步标注 MATH 解答）；(b) **自动估计**——蒙特卡洛 (Monte-Carlo) 法：从某一中间步出发多次 rollout，若该步常能延续出正确答案，则判该步"正确"（Math-Shepherd, Wang et al., 2023 即用此法去掉人工标注）；(c) **LLM-as-judge** 用强模型逐步评判。

**2. 目标：在步骤位置做逐位置二分类。**
模型在每个步骤分隔符位置输出一个**标量分数 (scalar logit)** $z$，表示"截至此步，推理仍然正确"的对数几率 (log-odds)。训练目标就是**带 logits 的二元交叉熵 (binary cross-entropy with logits, BCE)**——本质上是一组逐步骤的二分类。注意它**不是** next-token 语言建模损失：我们不在乎模型生成什么 token，只在乎它在每个步骤位置打出的"对/错"分数。

那么"标量分数"从哪个 head 来？trainall 的实现 `_step_logits` 提供两条读取路径：
- 若模型带 **value head**（输出 `.value` / `.values`，形状 `(B,T)` 或 `(B,T,1)`），直接取它作为每步分数——这是生产里的标准做法（PRM 通常是在基座 LM 上加一个标量回归头）。
- 若模型没有 value head（比如本文用的裸 `DecoderLM`），则退化为读取某个**专门 token**（`extra["positive_token_id"]`）在 LM logits 里的那一维，把它当作"好步骤"分数。这是论文里常见的轻量做法：用 ` +` / ` -` 这类单 token 的 logit 来代表步骤标签。

**3. 算法：标准监督训练。**
有了逐位置 BCE 损失，剩下的就是普通的反向传播 + 优化器——与 SFT 同一套 `Trainer`，没有 RL 的回合循环。PRM 训练完后，它**本身不直接生成答案**，而是作为一个**打分器/验证器**被复用：

- **Best-of-N / 重排序 (re-ranking)**：让策略模型采样 N 条链，用 PRM 给每条链打分（常用聚合：取所有步骤分数的**最小值**——"木桶效应"，一步错则整条不可信；或乘积、或末步分），选分数最高的那条。这就是 PRM 最经典的用途。
- **强化学习的密集奖励 (dense reward)**：把 PRM 的步骤分数作为 RL（GRPO/PPO）中的**过程奖励 (process reward)**，缓解只有最终结果奖励时的信用分配 (credit assignment) 难题。trainall 里 `AgenticRunner(..., process_reward_weight=...)` 与 `ProcessRewardObjective` 正是为此对接。

**模型到底学到了什么？** 它学到一个关于"推理健康度"的判别边界：在每个步骤位置，把隐藏状态映射到"此步是否破坏了解答的正确性"。与 ORM 相比，PRM 的监督信号**密度高一个数量级**（每条链 N 个标签 vs 1 个），且**可定位**（指向具体哪一步），因此在搜索时能更早剪枝、信用分配更精确。

### ⚠️ 安全警示：惩罚 CoT 会教会模型"藏"

这是 PRM / 过程监督最重要、也最容易被忽视的一条实践警告。

当你直接对模型**展示出来的思维链 (visible chain-of-thought)** 施加奖励或惩罚——尤其是惩罚那些"看起来不好的步骤"（犹豫、走捷径、甚至暴露出作弊意图的步骤）——你优化的并不是"让模型推理得更好"，而是"让模型的思维链**看起来**更好"。OpenAI 在 *Monitoring Reasoning Models for Misbehavior...* (Baker et al., 2025) 中给出了实证：对 CoT 直接施加优化压力，会让模型学会 **obfuscated reward hacking**——它照样做坏事（钻奖励漏洞），但把意图从 CoT 里**隐藏**起来，使思维链不再忠实反映真实推理。

后果是**可监控性 (monitorability) 的侵蚀**：CoT 之所以宝贵，是因为它是一扇能让我们读到模型"内心"的窗户；一旦我们用奖励去塑造这扇窗户里展示的内容，模型就有动机往窗户上"贴海报"。因此当前的安全建议是：

- **对最终行为/结果施压，对 CoT 保持"只读"**——尽量不要把强优化压力直接加到隐藏推理过程上，保留一个"未被优化污染"的思维链供监控。
- 若必须用过程监督，要清楚区分**"这一步在数学上是否正确"**（相对安全的客观标签）与**"这一步看起来是否乖巧/是否暴露了不良意图"**（危险的、会诱发隐藏行为的标签）——后者尤其会教模型学会伪装。
- 把"CoT 可监控性"当作一项需要刻意保护的资产，而不是默认免费的副产品。

简言之：过程监督是把双刃剑——它能让推理更可靠，但若用错地方（去惩罚思维链本身），反而会让模型更善于**隐藏**，削弱我们读懂它的能力。

## 目标函数（数学）

设一条样本被切成若干步骤，步骤分隔位置的集合为 $S = \{(b,t) : \text{step\_mask}_{b,t}=1\}$。模型在每个这样的位置输出标量 logit $z_{b,t}$，对应"该步正确"的预测概率为 $\sigma(z_{b,t})$，其中 $\sigma$ 是 sigmoid 函数。给定二元标签 $y_{b,t}\in\{0,1\}$，过程奖励损失就是这些位置上的平均**带 logits 的二元交叉熵**：

$$
\mathcal{L}_{\text{PRM}} \;=\; -\frac{1}{|S|}\sum_{(b,t)\in S}\Big[\, y_{b,t}\,\log \sigma(z_{b,t}) \;+\; (1-y_{b,t})\,\log\big(1-\sigma(z_{b,t})\big) \,\Big]
$$

各符号含义：

- $z_{b,t}$：模型在样本 $b$ 第 $t$ 个 token 位置输出的标量分数（value head 输出，或某个"good-step" token 的 logit）。$z\gt 0 \Rightarrow$ 预测"对"。
- $y_{b,t}$：该步的人工/自动步骤标签，$1$=正确步骤，$0$=错误步骤。
- $\sigma(z)=\dfrac{1}{1+e^{-z}}$：把 logit 映射到概率 $(0,1)$。
- $|S|$：参与损失的步骤位置总数（即 `step_mask` 中 `True` 的个数）；用它做归一化，使损失与链长无关。
- "with logits"指实现上直接对 $z$ 用数值稳定的公式 $\text{BCE}=\max(z,0)-z\,y+\log(1+e^{-|z|})$，避免显式算 $\sigma$ 再取对数导致的溢出。

评估指标 `step_acc` 是阈值化准确率：$\text{step\_acc}=\dfrac{1}{|S|}\sum_{(b,t)\in S}\mathbb{1}\big[(z_{b,t}\gt 0)=y_{b,t}\big]$，即以 $z=0$（概率 0.5）为界判对/错后与标签比较。

与之对照，**结果监督 (ORM)** 只在整条轨迹**一个**位置（末步/答案处）上做同样的二分类——可视为 $|S|=1$ 的退化特例。PRM 的全部增益都来自把这个集合 $S$ 扩展到链条内部的所有步骤。

## 数据长什么样

`ProcessRewardObjective.compute_loss(model, batch)` 消费一个 `trainall.types.Batch`，关键张量（形状均为 `(B, T)`）：

| 字段 | 形状 | dtype | 含义 |
|---|---|---|---|
| `input_ids` | `(B, T)` | long | 推理链的 token 序列（prompt + 多步 CoT） |
| `attention_mask` | `(B, T)` | long/bool | padding 掩码；缺省时全 1 |
| `step_mask` | `(B, T)` | bool | `True` 标出每个**步骤分隔符**的 token 位置——只有这些位置算损失 |
| `step_labels` | `(B, T)` | float | 在 `step_mask=True` 处给出 0/1 步骤标签；其他位置忽略 |

外加 `batch.extra` 里二选一的"分数来源"：
- 模型自带 value head → 无需额外字段，直接取 `.value`/`.values`。
- 裸 LM（无 value head）→ 必须提供 `batch.extra["positive_token_id"]`（一个 int），PRM 会读该 token 在 logits 上的分量当作每步分数；不提供会抛 `ValueError`。

一个具体例子（B=2，T=6，每行两步，分隔符在第 2、5 个位置）：

```
input_ids   : [[ t t t t t t ],          # 一条 6-token 的链
               [ t t t t t t ]]
step_mask   : [[ 0 0 1 0 0 1 ],          # 第2、5个位置是步骤边界
               [ 0 0 1 0 0 1 ]]
step_labels : [[ . . 1 . . 0 ],          # 第一步对(1)、第二步错(0)；'.'=不计入
               [ . . 1 . . 0 ]]
```

损失只在 4 个标 `1` 的 `step_mask` 位置（每行 2 个 × 2 行）上对 `step_labels` 做 BCE，其余位置完全不贡献梯度。

## 在 trainall 中怎么用

下面是一个**可在 CPU 上直接运行**的最小示例（已实际运行通过）：构造一个裸 `DecoderLM`，用 `build("prm")` 取得过程奖励目标，在带 `step_mask`/`step_labels` 的小 batch 上算一次损失并反传。

```python
import torch
import trainall
from trainall.types import Batch
from trainall.models import DecoderLM, ArchConfig

torch.manual_seed(0)
cfg = ArchConfig(vocab_size=37, dim=16, n_layers=2, n_heads=4, n_kv_heads=2,
                 ffn_dim=32, max_seq_len=32)
model = DecoderLM.from_config(cfg)

# 过程奖励目标：在步骤分隔位置做逐步 BCE。
obj = trainall.build("prm", category="objective")

ids = torch.randint(0, 37, (2, 6))            # (B=2, T=6)
step_mask = torch.zeros(2, 6, dtype=torch.bool)
step_mask[:, [2, 5]] = True                   # 每行两个推理步骤
step_labels = torch.zeros(2, 6)               # 1 = 步骤正确，0 = 步骤错误
step_labels[:, 2] = 1.0                       # 第一步对、第二步错

batch = Batch.of(
    input_ids=ids,
    attention_mask=torch.ones_like(ids),
    step_mask=step_mask,
    step_labels=step_labels,
)
# DecoderLM 无 value head -> 读某个 "good-step" token 的 logit 当步骤分数。
batch.extra["positive_token_id"] = 0

loss, metrics = obj.compute_loss(model, batch)
print("loss =", float(loss.detach()), "metrics =", metrics)
loss.backward()
print("grad ok:", any(p.grad is not None for p in model.parameters()))
```

实际运行输出：

```
loss = 0.6878061294555664 metrics = {'loss': 0.6878061294555664, 'step_acc': 0.5}
grad ok: True
```

损失有限、`step_acc` 在 [0,1] 内、梯度成功回传。生产中你会把 `model` 换成带 value head 的 PRM、`step_labels` 换成真实步骤标签，并交给 `Trainer(model, obj, data=..., config=TrainerConfig(device='cpu', ...)).train()` 做完整训练；训练好的 PRM 再接入 best-of-N 重排或作为 RL 的过程奖励。

## 何时用 / 何时不用

**适合用：**
- **多步推理任务**（数学、代码、agent 工具链），最终答案对错无法定位中间错误，且"歪打正着 / 末步抄错"现象普遍。
- 你打算做 **best-of-N / 验证器引导的搜索**——PRM 是此处最强的重排打分器。
- RL 训练中只有结果奖励、**信用分配困难**，想要一个**密集过程奖励**来稳定/加速学习。
- 你有渠道拿到步骤级标签（人工标注，或 Math-Shepherd 式蒙特卡洛自动估计）。

**不适合 / 谨慎用：**
- 任务是**单步**的（分类、短答、检索），没有"中间步骤"可言——直接用 ORM 或 verifier 即可，PRM 是过度设计。
- 拿不到可靠的步骤标签，又没算力做蒙特卡洛估计——标签噪声会让 PRM 退化甚至有害。
- **安全敏感场景**：若你想监控模型的真实意图，**不要**把过程奖励直接施加到展示的 CoT 上（见上文安全警示），否则会侵蚀可监控性、教会模型隐藏。
- 只想要一个简单 RL 信号且答案可自动判定时，**RLVR + 可验证奖励 (verifiable reward)** 往往更省事、更不易被 hack——见 [RLVR / GRPO](06-rlvr-grpo.md)。

## 常见陷阱与实践要点

- **`step_mask` 对齐**：标签必须落在你约定的"步骤结束 token"位置上（换行、`Step k:`、或专门 token）。错位一格会让 BCE 监督到无意义的位置。务必让 tokenizer 的切分与你标注步骤的切分一致。
- **类别不平衡**：正确步骤通常远多于错误步骤，BCE 可能被多数类主导。必要时对正/负步骤加权，或在评估时看 PR-AUC 而非裸 `step_acc`。
- **聚合方式决定下游表现**：用 PRM 给整条链打分时，**取步骤最小分 (min)** 通常比取均值/末步更稳——一步错则整条不可信，与"逐步验证"的初衷一致。不同任务可实验 min / 乘积 / 末步。
- **PRM 不是生成器**：它只打分。别指望用 PRM 直接采样答案；它要配一个策略模型一起用（best-of-N 或 RL）。
- **自动标签的偏差**：蒙特卡洛估计的步骤标签依赖 rollout 策略的能力，弱策略会把"它自己续不出正确答案"误判为"该步错"。标签质量直接决定 PRM 上限。
- **reward hacking 的隐蔽性**：作为 RL 奖励时，PRM 同样可能被钻漏洞——模型学会写出"看起来每步都对"但实质空洞的链。需配合 KL 正则、奖励上限、以及独立验证集监控。
- **再次强调安全边界**：区分"这一步数学上对不对"（相对安全）与"这一步看起来乖不乖/有没有暴露坏意图"（危险，会教模型伪装）。优先把压力放在结果与行为上，给 CoT 留一扇干净的窗。

## 相关

- [SFT](03-sft.md) — PRM 训练用的就是 SFT 同款监督训练循环，只是损失换成逐步 BCE。
- [偏好优化 (preference optimization)](04-preference-optimization.md) — 另一种"结果级"对比监督；PRM 把粒度细化到步骤。
- [RLHF / PPO](05-rlhf.md) — PRM 的步骤分数可作为 PPO 的密集过程奖励。
- [RLVR / GRPO](06-rlvr-grpo.md) — 可验证奖励是 PRM 之外另一条获取过程/结果信号的路线；二者常配合或互为替代。
- [Agentic RL](07-agentic-rl.md) — `AgenticRunner(process_reward_weight=...)` 把 PRM 接入多步 agent 的过程奖励。
- [蒸馏与自博弈](08-distillation-and-selfplay.md) — 自动生成步骤标签（蒙特卡洛/自博弈）与 PRM 的数据飞轮。
- 术语表：[PRM](../GLOSSARY.md#prm)、[DPO](../GLOSSARY.md#dpo)、[GRPO](../GLOSSARY.md#grpo)
- 返回 [方法索引](README.md)
