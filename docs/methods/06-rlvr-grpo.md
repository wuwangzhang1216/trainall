<!-- nav -->
<table width="100%"><tr><td align="left" width="30%"><a href="05-rlhf.md">← RLHF</a></td><td align="center" width="40%"><a href="README.md">📑 索引</a> · <a href="../GLOSSARY.md">📖 术语词典</a> · <a href="en/06-rlvr-grpo.md">🌐 English</a></td><td align="right" width="30%"><a href="07-agentic-rl.md">Agentic RL →</a></td></tr></table>
<!-- /nav -->

# RLVR + GRPO (Reinforcement Learning from Verifiable Rewards + Group Relative Policy Optimization)

> **用一个能自动判对错的"裁判 (verifier)"取代学出来的奖励模型，再用"同一道题采样一组答案、组内比好坏"的方式估计优势，从而扔掉价值网络 (value network) ——这就是 DeepSeek-R1 把推理能力做上去的那条路。**

![RLVR + GRPO 数据流：一个 prompt 采样一组答案，verifier 打分，组内归一化成优势，clipped policy gradient 更新策略](../assets/rlvr_grpo.png)

## 直觉：它到底在做什么

经典 RLHF（见 [RLHF/PPO](05-rlhf.md)）的奖励来自一个**学出来的奖励模型 (reward model)**：先用人类偏好数据训一个打分器，再让策略去最大化它的分数。这条路有两个老毛病：奖励模型本身会被"刷分 (reward hacking)"，而且它只是人类偏好的一个有噪声的代理 (proxy)，分数高不代表答案真的对。

RLVR (Reinforcement Learning from Verifiable Rewards) 换了个思路：**对那些答案能被程序自动判定对错的任务（数学、代码、SQL、JSON 格式……），直接用一个确定性的 verifier 当奖励**。`\boxed{42}` 和参考答案 `42` 相等就给 1，不等就给 0；代码跑过单元测试就 1，挂了就 0。奖励不再是"学"出来的，而是"算"出来的——它不会被刷分，因为它就是 ground truth。

GRPO (Group Relative Policy Optimization) 是配套的优化算法。PPO 需要一个**价值网络**来估计基线 (baseline)，从而把奖励变成优势 (advantage)。GRPO 发现：既然我可以对**同一个 prompt** 一次采样一整组（比如 8 个）答案，那这组答案的**平均奖励**就是个天然的、无偏的基线——比平均好的答案优势为正，比平均差的为负。于是价值网络被彻底删掉，省一半显存、少一个要调的网络。

一句话：RLVR 解决"奖励从哪来"，GRPO 解决"优势怎么估"。两者合在一起，就是 2024–2025 年推理模型（DeepSeek-R1、Tulu 3 等）的训练主干。

## 原理与架构（深度讲解）

### data → reward → objective → algorithm 的拆解

把这条管线拆成四层，每一层在 trainall 里都是一个独立可替换的组件：

1. **data（任务 + 参考答案）**：每条样本是 `(prompt, reference)`。注意 reference **不是**一段示范文本（那是 [SFT](03-sft.md) 的监督信号），而是用来**判定**的依据——一个数字、一段断言、一个期望的 SQL 结果集。
2. **rollout（采样一组）**：对每个 prompt，用当前策略采样 `group_size` 个答案，它们共享同一个 `group_id`。这一步对应 `Rollout(...).group_sample(...)`。
3. **reward（verifier 打分）**：`VerifierReward` 对每条 trajectory 跑 verifier，把 `[0,1]` 的标量奖励写回去。这是 RLVR 的核心——奖励是**确定性、无参数**的。
4. **objective + algorithm（GRPO 更新）**：组内把奖励归一化成优势，套上 PPO 的 clipped 替代目标 (surrogate)，对 response token 做 policy gradient。

### 为什么"组内归一化"能替代价值网络

policy gradient 的方差主要来自奖励的绝对尺度。REINFORCE 的梯度是 $\nabla \log \pi \cdot r$，如果所有答案奖励都在 0.8 附近，那即使最差的答案也会被推高——因为它的 $r>0$。**减去一个基线** $b$ 不改变梯度的期望（无偏），却能大幅降方差。PPO 用价值网络学这个 $b$；GRPO 直接用**组内均值** $\bar r_g$ 当 $b$，再除以组内标准差做归一化（z-score）。

这个替代之所以成立，关键在于**同组的答案来自同一个 prompt**——它们面对的是同一道题，奖励直接可比。组均值是该 prompt 难度的一个即时、无偏估计；不需要任何额外参数去拟合它。代价是：你必须对每个 prompt 采样一整组（`group_size` 倍的采样成本），而且**当一组答案全对或全错时，组内方差为 0，优势全为 0，这一组不产生任何梯度信号**——这正是为什么 GRPO 需要难度适中的 prompt（curriculum）才高效。

### 模型真正学到了什么：RLVR 放大什么、不创造什么

这是最容易被误解的一点。RLVR 的梯度只会**提高已经能产出正确答案的那些 rollout 的概率，压低错误 rollout 的概率**。也就是说：

- 如果一个 prompt 在采样 `k` 次里**从没**碰对过，组内全 0，优势全 0，**学不到任何东西**。
- RLVR **放大 (amplify)** 的是模型采样分布里**本就存在但概率不高**的正确推理路径——它把 base/SFT 模型已经"会一点点"的能力，通过反复强化采样到的成功轨迹，变成"稳定会"。
- 它基本**不创造 (create)** 全新的、base 模型采样空间里压根没有的能力。pass@1 提升往往伴随 pass@k（大 k）几乎不变甚至略降——说明分布被"集中"到了已有的好模式上，而非长出新模式。

理解这点能解释很多现象：为什么 RLVR 前通常要先做高质量 SFT（把正确路径的初始概率抬起来）；为什么"涌现"的长 CoT、自我反思（"wait, let me reconsider..."）其实是把 base 模型偶尔才采到的反思 token 序列强化成了稳定行为，而不是凭空学会反思。

### DeepSeek-R1：把这条路走到极致

DeepSeek-R1（2025）证明了一个激进版本：**从 base 模型直接上 RLVR、连 SFT 冷启动都先不做**（R1-Zero），仅用 GRPO + 规则奖励（答案对不对 + 格式对不对），模型就自发涌现出长链推理、自检、回溯。R1 正式版在此之上加了少量冷启动 SFT 来稳定可读性。它的奖励几乎全是**可验证的**：数学有标准答案、代码有测试。这从工程上印证了 RLVR 的论点——**当奖励可验证时，你不需要奖励模型，规则就够，而且更不容易被刷分**。

### verifier zoo：奖励信号的"裁判团"

RLVR 的天花板由 verifier 的覆盖面决定。trainall 提供一组可组合的 verifier（`category='verifier'`）：

| key | 判什么 | reference 是什么 |
| --- | --- | --- |
| `math` | 数值/符号等价（抽 `\boxed{}` 或最后一个数，带容差） | 标准答案字符串，如 `"42"`、`"1/3"` |
| `code` | 抽出代码块跑参考断言，看进程退出码 | `assert ...` 测试串 |
| `sql` | 在 sqlite 里建表灌数据，比对查询结果集 | `{schema, seed, expected_sql/expected_rows}` |
| `json` | 是否合法 JSON；可选 schema 校验 | `None` 或 JSON schema |
| `format` | 是否含必需标签/字段（如 `<think>`/`<answer>`） | 由 verifier 构造参数指定 |
| `regex` | 是否匹配正则 | 模式串（可 per-call 覆盖） |
| `citation` | 引文是否真出自给定来源（防编造） | 来源文本列表 |
| `composite` | 把多个 verifier 加权/逻辑组合 | 透传给子 verifier |

`composite` 尤其重要：真实 RLVR 奖励常是**正确性 × 格式**的组合——R1 的奖励就是"答案对"加"放在 `<answer>` 里"。把 `format` 和 `math` 用 `CompositeVerifier(mode='weighted')` 拼起来，就能同时塑造正确性和可读性。

## 目标函数（数学）

**第一步：组内相对优势 (group-relative advantage)。** 对 prompt $q$ 采样的一组 $G$ 个答案 $\{o_1,\dots,o_G\}$，每个由 verifier 打分 $r_i$，优势为组内 z-score：

$$A_i = \frac{r_i - \operatorname{mean}(\{r_1,\dots,r_G\})}{\operatorname{std}(\{r_1,\dots,r_G\}) + \varepsilon}$$

- $r_i$：第 $i$ 个答案的 verifier 奖励，$\in[0,1]$。
- $\operatorname{mean},\operatorname{std}$：**只在同一组内**统计（trainall 用 `unbiased=False` 的有偏 std）；$\varepsilon=10^{-6}$ 防除零。
- $A_i$：组内归一化优势。比平均好 → 正；差 → 负；一组全相同 → 全 0。同一个 $A_i$ 会广播到该答案的所有 response token 上。

**第二步：clipped policy gradient（与 PPO 同形）。** 设当前策略对第 $i$ 个答案第 $t$ 个 token 的对数概率为 $\log\pi_\theta$，采样时的旧策略为 $\log\pi_{\text{old}}$，比率 $\rho_{i,t}=\exp(\log\pi_\theta - \log\pi_{\text{old}})$。逐 token 损失：

$$\mathcal{L}^{\text{PG}} = -\,\mathbb{E}_{i,t}\Big[\min\big(\rho_{i,t}\,A_i,\;\operatorname{clip}(\rho_{i,t},\,1-\epsilon,\,1+\epsilon)\,A_i\big)\Big]$$

- $\epsilon$：clip 半径（`clip_range`，默认 0.2），限制单步更新幅度，防止策略一步走太远。
- $\operatorname{clip}$ 与 $\min$ 一起构成 PPO 的悲观下界：优势为正时上限被 $1+\epsilon$ 卡住，为负时下限被 $1-\epsilon$ 卡住。
- 期望 $\mathbb{E}_{i,t}$ 在 **response token**（由 `response_mask` 选出）上取均值，prompt token 不计入。
- 当没有提供 `old_logps`（纯 on-policy 单步），$\rho\equiv 1$，目标退化为 REINFORCE：$\mathcal{L}=-\mathbb{E}[A_i \log\pi_\theta]$，梯度仍通过 $\log\pi_\theta$ 正常回传。

**第三步：KL 惩罚（k3 估计量，可选）。** 为防策略漂离参考策略 $\pi_{\text{ref}}$ 太远，加一项逐 token KL，trainall 用 Schulman 的 **k3 无偏低方差估计**：

$$\mathcal{L} = \mathcal{L}^{\text{PG}} + \beta\,\mathbb{E}_{i,t}\big[\,e^{d_{i,t}} - d_{i,t} - 1\,\big],\qquad d_{i,t}=\log\pi_{\text{ref}} - \log\pi_\theta$$

- $\beta$：`kl_coef`，KL 惩罚权重（默认 0，即不加 KL）。
- $e^d - d - 1$：k3 估计量，恒 $\ge 0$，是 $\mathrm{KL}(\pi_\theta\Vert\pi_{\text{ref}})$ 的无偏估计，方差远小于朴素的 $-d$。当 $\pi_\theta=\pi_{\text{ref}}$ 时 $d=0$，该项为 0。

## 数据长什么样

GRPO 消费的是 `trainall.types.Batch`，policy-gradient 约定的张量布局：

- `input_ids` `(B,T)`：prompt + response 拼接的 token id。
- `attention_mask` `(B,T)`：padding 掩码。
- `response_mask` `(B,T)`：1 标记 response token、0 标记 prompt token（损失只在 response 上算）。
- `rewards` `(B,)`：每条 trajectory 的 verifier 标量奖励。
- `group_ids` `(B,)`：同一个 prompt 的若干采样共享同一个 group id（优势在组内归一化的依据）。
- 可选 `old_logps` `(B,T)`：采样时旧策略的 token 对数概率（多步 off-policy 才需要）；不提供则比率恒为 1。
- 可选 `ref_logps` `(B,T)`：参考策略对数概率，配合 `kl_coef>0` 才生效。

`GRPOObjective.prepare_batch` 会在内部从 `rewards`+`group_ids` 自动算出 `advantages`，无需手动填。上游的 `Trajectory`（`prompt, response, reward, group_id, advantage, meta`）则是 rollout 阶段的载体——`reference` 放在 `traj.meta["reference"]` 里供 `VerifierReward` 取用。

## 在 trainall 中怎么用

下面三段对应管线的三个关键环节：(a) verifier 给可验证奖励；(b) 组内算优势；(c) GRPO 在分组 batch 上算 loss + 反传。全部已在 CPU 上跑通。

```python
import torch
import trainall
from trainall.types import Batch, Trajectory
from trainall.rl import compute_group_advantages
from trainall.rewards import VerifierReward

# (a) verifiable reward: 对/错由确定性 verifier 判定（不是学出来的奖励模型）
v = trainall.build("math", category="verifier")
ok  = v.verify(r"The final answer is \boxed{42}.", reference="42")
bad = v.verify(r"\boxed{41}", reference="42")
print("math correct :", ok.reward, ok.passed)   # 1.0 True
print("math wrong   :", bad.reward, bad.passed)  # 0.0 False

# VerifierReward 给一组 Trajectory 打分（reference 放在 meta 里）
vr = VerifierReward("math")
trajs = [
    Trajectory(prompt="2+2?", response=r"\boxed{4}", group_id=0, meta={"reference": "4"}),
    Trajectory(prompt="2+2?", response=r"\boxed{5}", group_id=0, meta={"reference": "4"}),
]
scores = vr.score(trajs)
print("reward scores:", scores)                  # [1.0, 0.0]

# (b) group-relative advantage: 组内 z-score，无需价值网络
for t, r in zip(trajs, scores):
    t.reward = r
compute_group_advantages(trajs)
print("advantages   :", [round(t.advantage, 4) for t in trajs])  # [1.0, -1.0]

# (c) GRPOObjective.compute_loss on a tiny grouped Batch
from trainall.models import DecoderLM, ArchConfig
torch.manual_seed(0)
cfg = ArchConfig(vocab_size=37, dim=16, n_layers=2, n_heads=4,
                 n_kv_heads=2, ffn_dim=32, max_seq_len=32)
model = DecoderLM.from_config(cfg)

obj = trainall.build("grpo", category="objective", clip_range=0.2, kl_coef=0.0)
ids = torch.randint(0, 37, (4, 6))
response_mask = torch.ones(4, 6)
response_mask[:, :2] = 0                          # 前 2 个 token 是 prompt
batch = Batch.of(
    input_ids=ids,
    attention_mask=torch.ones_like(ids),
    response_mask=response_mask,
    rewards=torch.tensor([1.0, 0.0, 1.0, 0.0]),
    group_ids=torch.tensor([0, 0, 1, 1]),        # 两个 prompt，各 2 个采样
)
loss, metrics = obj.compute_loss(model, batch)
print("grpo metrics :", {k: round(v, 4) for k, v in metrics.items()})
loss.backward()
print("has grad     :", any(p.grad is not None for p in model.parameters()))
```

实际输出：

```
math correct : 1.0 True
math wrong   : 0.0 False
reward scores: [1.0, 0.0]
advantages   : [1.0, -1.0]
grpo metrics : {'loss': 0.0, 'kl': 0.0, 'reward_mean': 0.5, 'adv_std': 1.1547}
has grad     : True
```

注意一个反直觉但正确的细节：上面打印的 `loss` 是 **0.0**，但梯度照样非零（实测参数梯度绝对值之和 ≈ 27.4）。原因是没传 `old_logps` 时比率 $\rho\equiv 1$，逐 token 损失就等于 $-A_i$；优势在每组内零和（$+1$ 与 $-1$），逐 token 均值正好抵成 0。但 `loss = -mean(A_i * logp_detached_ratio)` 的梯度路径仍通过 $\log\pi_\theta$——正梯度推高正优势答案、压低负优势答案的 token 概率。**loss 数值本身不是训练是否在进行的可靠指标，要看梯度和奖励均值。**

## 何时用 / 何时不用

**适合用 RLVR + GRPO：**
- 任务的正确性**可被程序判定**：数学、竞赛题、代码（有测试）、SQL、结构化抽取、可校验格式。这是充要条件。
- 已有一个**像样的 SFT/base 模型**，它对目标任务至少偶尔能采到正确答案（否则组内全错、学不动）。
- 想避免奖励模型的刷分和训练成本，追求 DeepSeek-R1 式的推理能力提升。

**不适合 / 慎用：**
- 奖励**无法自动验证**：开放式写作、对话风格、"哪个回答更有帮助"——这些是偏好/RLHF 的地盘（见 [偏好优化](04-preference-optimization.md)、[RLHF](05-rlhf.md)）。
- prompt 难度全部"太难"（pass@k≈0）或"太易"（pass@k≈1）：前者无梯度，后者组内方差为 0，两头都白采。
- 算力极紧：GRPO 每个 prompt 要采一整组，采样成本是单样本的 `group_size` 倍。
- 只想做基础能力注入而非强化已有能力：那应该先做 [SFT](03-sft.md) 或 [继续预训练](02-continued-pretraining.md)。

## 常见陷阱与实践要点

- **全对/全错组 = 零梯度。** 组内方差为 0 时优势全 0，这一组完全浪费。要么用 curriculum 控制难度（`Curriculum` 把通过率维持在 `target_low~target_high`），要么直接丢弃零方差组。
- **loss 数值会误导。** 如上所示，平衡组内 loss 可能恒为 0 但训练正常。监控 `reward_mean`（应随训练上升）、`adv_std`、以及外部评测的 pass@1，别盯着 loss。
- **verifier 必须严格且无副作用。** 宽松的 verifier 会被策略钻空子——比如 `math` 若只抽"最后一个数字"，模型可能学会在末尾堆答案而不真推理；`code` verifier 必须 sandbox/超时，否则恶意代码会拖垮训练。优先用 `composite` 把格式约束也纳入奖励。
- **RLVR 放大而非创造。** 若 base 模型在某能力上 pass@k（大 k）就是 0，RLVR 帮不上忙——先用 SFT 把正确路径的概率抬起来。RLVR 是"锐化"，不是"无中生有"。
- **KL 是稳定器，不是必需品。** R1-Zero 甚至不加 KL。但若发现策略输出退化（重复、乱码、语言漂移），调高 `kl_coef` 把它拉回参考策略附近。trainall 用 k3 估计量，比朴素 KL 方差小。
- **response_mask 别算错。** 只在 response token 上计损失；把 prompt token 算进去会污染梯度。Batch 里 `response_mask[:, :prompt_len] = 0`。
- **group_size 的权衡。** 太小（如 2）则基线估计噪声大；太大则采样贵。实践常用 8–16。
- **奖励尺度。** verifier 输出最好落在 `[0,1]`；组内 z-score 会自动归一化尺度，但极端稀疏（几乎全 0）的奖励仍会让多数组退化。

## 相关

- [RLHF / PPO](05-rlhf.md)：用**学出来的**奖励模型 + 价值网络的经典路线；GRPO 正是删掉价值网络、RLVR 正是换掉奖励模型的演进。
- [偏好优化 (DPO 等)](04-preference-optimization.md)：当奖励不可验证、只有成对偏好时的替代方案。
- [Agentic RL](07-agentic-rl.md)：把单步 verifier 奖励扩展到多步工具调用环境（`AgenticRunner`、`MultiStepEnv`）。
- [过程监督 (PRM)](09-process-supervision.md)：给推理**每一步**而非最终答案打分的稠密奖励信号。
- [SFT](03-sft.md)：RLVR 之前抬高正确路径初始概率的冷启动。
- [蒸馏与自博弈](08-distillation-and-selfplay.md)：用 `RejectionSampler`/`SelfPlayLoop` 配合 verifier 自动造 RLVR 训练数据。
- 词表：[GRPO](../GLOSSARY.md#grpo) · [RLVR](../GLOSSARY.md#rlvr) · [advantage](../GLOSSARY.md#advantage) · [KL penalty](../GLOSSARY.md#kl-penalty) · [verifier](../GLOSSARY.md#verifier)
- 返回 [方法索引](README.md)
