<!-- nav -->
<p align="center">
  <a href="04-preference-optimization.md">← 偏好优化</a> ·
  <a href="README.md">索引</a> ·
  <a href="../GLOSSARY.md">术语表</a> ·
  <a href="en/05-rlhf.md">English</a> ·
  <a href="06-rlvr-grpo.md">RLVR+GRPO →</a>
</p>
<!-- /nav -->

# RLHF / RLAIF（基于人类/AI 反馈的强化学习）

> **当"好"无法写成 loss 时，就先学一个奖励模型 (reward model) 把人类偏好压成一个标量分数，再用 PPO 把策略往高分方向推，同时用 KL 拴住别跑偏。**

![RLHF 三阶段：偏好数据 -> 奖励模型 -> PPO](../assets/rlhf_ppo.png)

## 直觉：它到底在做什么

SFT（监督微调）教模型"模仿"——给一条参考答案，让模型逐 token 复现它。但很多我们真正想要的东西**写不出参考答案**：什么叫"更有帮助"、"更诚实"、"语气更得体"、"不说教"？这些目标没有唯一正确的字符串，只有"A 比 B 好"这种**相对判断**。

RLHF (Reinforcement Learning from Human Feedback) 的核心洞察是：与其要求人类**写出**完美答案，不如让人类**在两个模型输出之间挑一个更好的**。挑选比创作便宜得多、一致性也高得多。于是整个流程分成三步（Ouyang et al., 2022, *InstructGPT*；Christiano et al., 2017）：

1. **收集偏好**：对同一个 prompt 采样多个回答，让人类（或一个更强的模型——这就是 RLAIF, RL from AI Feedback）标注 "chosen ≻ rejected"。
2. **训练奖励模型 (reward model, RM)**：拟合一个标量函数 $r_\phi(x, y)$，使得被偏好的回答得分更高。这一步把"人类品味"蒸馏进了一个可微的打分器。
3. **PPO 强化学习**：把语言模型当作策略 (policy)，生成回答、用 RM 打分作为奖励，用 PPO 把策略往高分方向更新——但加一条 KL 缰绳，防止它为了刷高 RM 分数而跑成一个连自己 SFT 起点都不认识的怪物。

一句话：**RM 学会"什么是好"，PPO 学会"怎么变好"，KL 负责"别变疯"。**

## 原理与架构（深度讲解）

### 为什么需要一个"学出来的奖励"

强化学习需要一个奖励信号 $r$。在围棋里奖励是显然的（赢/输），在代码题里可以跑单测（见 [RLVR](06-rlvr-grpo.md)）。但对话质量没有这种可程序化判定的 oracle。RLHF 的关键工程妥协是：**用人类的成对偏好去拟合一个奖励函数，然后把这个函数当 oracle 用。**

奖励模型通常就是一个语言模型骨干，去掉 LM head、换上一个**标量头 (scalar/value head)**：读入 $(x, y)$，在最后一个非 pad token 的隐状态上输出一个实数 $r_\phi(x, y)$。它不预测下一个 token，只预测"这条回答有多好"。

### 数据 → 目标 → 算法 的三段映射

- **数据**：第一阶段是偏好对 `(prompt, chosen, rejected)`；第二阶段是 prompt 集合（PPO 在线采样自己的回答，不需要参考答案）。
- **目标**：RM 阶段是 **Bradley-Terry 成对损失**；PPO 阶段是 **clipped 策略梯度代理目标 + KL 惩罚**。
- **算法**：RM 是普通的监督学习（一个二分类的对数似然）；PPO 是 on-policy 的 actor-critic，需要反复"采样 → 打分 → 更新"的循环。

### Bradley-Terry：把成对偏好变成可微损失

Bradley-Terry 模型（Bradley & Terry, 1952）假设：给定两个回答的潜在分数 $r_c, r_r$，人类偏好 chosen 的概率是

$$P(c \succ r) = \sigma(r_c - r_r), \qquad \sigma(z) = \frac{1}{1+e^{-z}}.$$

直觉很干净：**只有分差 $r_c - r_r$ 决定偏好概率**，绝对分值无所谓（整体平移一个常数不改变任何偏好概率，所以 RM 学到的奖励是"无标度的"——这一点后面会咬人）。训练就是对这个伯努利似然做最大似然，等价于最小化 $-\log\sigma(r_c - r_r)$。当模型把 chosen 排在前面时 margin 为正、loss 小；排错时 loss 大。

### PPO：在 RM 这座"山"上往上爬，但拴着缰绳

有了 $r_\phi$，把语言模型 $\pi_\theta$ 当策略：对 prompt $x$ 采样回答 $y$，奖励是

$$R(x, y) = r_\phi(x, y) - \beta\, \mathrm{KL}\!\left[\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\right].$$

第一项推高 RM 分数；第二项是**KL 缰绳 (KL leash)**，把策略拴在参考模型 $\pi_{\text{ref}}$（一般就是 SFT 模型）附近。

PPO（Schulman et al., 2017）不直接做朴素策略梯度，而是优化一个**裁剪过的代理目标 (clipped surrogate)**。它的设计目的：一批采样数据可以**复用做多步梯度更新**，而不会因为重要性权重 (importance ratio) 把策略一步推太远。每个 response token 上定义比值 $\rho_t = \pi_\theta(a_t)/\pi_{\text{old}}(a_t)$，优化 $\min(\rho_t A_t,\ \mathrm{clip}(\rho_t, 1{-}\epsilon, 1{+}\epsilon) A_t)$。当 $\rho$ 偏离 1 太远时，clip 让梯度归零——相当于一个"软信任域 (trust region)"。

优势 $A_t$ 由 **GAE (Generalized Advantage Estimation)** 计算（Schulman et al., 2016），需要一个 critic（value head）去估计每个位置的期望回报，从而把"这个 token 比平均好多少"算出来、降低方差。在 trainall 里 `compute_gae(rewards, values, gamma, lam, mask)` 就做这件事，`PPOObjective` 在 batch 里提供了 `values`/`returns` 时会额外加上 value 回归损失 $\tfrac12(V-G)^2$。

### 模型到底学到了什么

- **RM 学到一个"品味排序器"**：它内化的不是某个标准答案，而是"在这些维度上 A 通常比 B 好"的统计规律。它泛化得好不好，直接决定 PPO 阶段会被优化成什么。
- **策略学到"如何最大化 RM 的偏好"**——注意，是 RM 的偏好，不是人类的偏好。两者的差距就是下面"陷阱"一节的全部故事。

### 奖励黑客 (reward hacking) 与 KL 缰绳

RM 只是真实偏好的**代理 (proxy)**，在训练分布外它会犯错、会有可被利用的盲区。PPO 是个无情的优化器：只要某种文本模式能骗到高 RM 分（比如一律写得超长、堆砌"当然！我很乐意帮你"、用 markdown 标题装腔），它就会**朝那个漏洞坍缩**。这就是**奖励黑客 / reward over-optimization**（Gao et al., 2023 给出了过优化的标度律）。

KL 惩罚是第一道防线：它惩罚策略偏离 SFT 起点太远，相当于说"你可以变好，但不能变成另一个分布"。$\beta$ 太小 → 自由过度 → 奖励黑客；$\beta$ 太大 → 策略几乎不动 → 学不到东西。调 $\beta$（或等价的 target-KL 自适应控制器）是 PPO-RLHF 工程的核心难点之一。

### 为什么很多团队转向了 DPO

PPO-RLHF 要同时在显存里放四个模型（policy、reference、reward、critic），还要维护一个 on-policy 的采样-打分-更新循环，超参（$\beta$、clip $\epsilon$、GAE 的 $\lambda$、学习率、KL 控制器）多且敏感，复现性差。[DPO](04-preference-optimization.md) (Direct Preference Optimization) 证明了：**在 RM 用 Bradley-Terry、奖励里带 KL 这套假设下，最优策略有闭式解**，可以把"RM + PPO"两步合并成一个**直接在偏好对上做的监督式损失**——不用训 RM、不用在线采样、不用 critic。它把同样的 Bradley-Terry 思想"折叠"进策略本身，把 $r_c - r_r$ 替换成 $\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ 的对数比值差。代价是 DPO 是 off-policy 的、用的是固定偏好数据，**不能在线探索**，理论上限可能低于一个调好的 PPO。

实务取舍：要快、要稳、数据是静态偏好对 → DPO；要榨干性能、能在线生成、有能力调 PPO、或要做更复杂的奖励塑形 (reward shaping) → PPO-RLHF。两者共享同一个 Bradley-Terry 内核，理解 RM 损失对理解 DPO 也是必修课。

## 目标函数（数学）

**第二阶段 · Bradley-Terry 奖励模型损失**。对一个偏好对 $(x, y_c, y_r)$：

$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_c, y_r)}\Big[\log \sigma\big(r_\phi(x, y_c) - r_\phi(x, y_r)\big)\Big].$$

- $r_\phi(x, y)$：奖励模型对回答 $y$ 给出的标量分数（取最后一个非 pad token 隐状态过标量头）。
- $\sigma$：sigmoid。$\sigma(r_c - r_r)$ 即 chosen 胜出的概率。
- 损失只依赖**分差**，所以 $r_\phi$ 的绝对零点是任意的（无标度）。
- 监控指标：pairwise accuracy $= \mathbb{E}[\mathbb{1}(r_c \gt  r_r)]$，以及 reward margin $\mathbb{E}[r_c - r_r]$。

**第三阶段 · PPO 裁剪代理目标**。记 response token 上的重要性比值 $\rho_t = \dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)} = \exp(\log p_\theta - \log p_{\text{old}})$，优势 $A_t$ 由 GAE 给出，则

$$\mathcal{L}_{\text{PPO}}(\theta) = -\,\mathbb{E}_t\Big[\min\big(\rho_t A_t,\ \operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,A_t\big)\Big] \;+\; c_v\,\underbrace{\tfrac12\,\mathbb{E}_t\big[(V_\theta(s_t)-G_t)^2\big]}_{\text{value 回归}} \;-\; c_e\,\mathbb{E}_t[\mathcal{H}_t] \;+\; \beta\,\mathbb{E}_t\big[\mathrm{KL}_t\big].$$

- $\epsilon$：clip 范围（`clip_range`，默认 0.2）。比值跑出 $[1-\epsilon, 1+\epsilon]$ 时该项梯度被截断，限制单步步长。
- $A_t$：GAE 优势，$A_t = \sum_{l\ge0}(\gamma\lambda)^l\,\delta_{t+l}$，$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。
- $G_t = A_t + V(s_t)$：value head 的回归目标（returns）。$c_v$ = `vf_coef`。
- $\mathcal{H}_t$：策略熵，鼓励探索，$c_e$ = `ent_coef`。
- $\beta$ = `kl_coef`：对参考策略的 KL 惩罚（缰绳）。
- 奖励来源 $r_\phi$ 经由 GAE 进入 $A_t$；在最简单的 bandit 设定里也可把整条回答的 RM 分作为序列级优势直接广播到所有 response token。

## 数据长什么样

**RM 阶段** 消费一个偏好 `Batch`（`trainall.types.Batch`），左右两路各自一组 token：

```python
import torch
from trainall.types import Batch

B, T = 4, 6
cids = torch.randint(0, 37, (B, T))
rids = torch.randint(0, 37, (B, T))
batch = Batch(tensors=dict(
    chosen_input_ids=cids,                       # (B, T) 偏好回答
    chosen_attention_mask=torch.ones_like(cids), # (B, T)
    rejected_input_ids=rids,                     # (B, T) 非偏好回答
    rejected_attention_mask=torch.ones_like(rids),
))
```

`BradleyTerryObjective` 会分别对 chosen / rejected 取**最后一个非 pad token** 的分数。若骨干没有 value head（`DecoderLM` 就没有），通过 `batch.extra["scalar_head"]` 传入一个把池化隐状态映射到标量的线性头。

**PPO 阶段** 消费的是 policy-gradient 形态的 `Batch`：`input_ids`、`attention_mask`、`response_mask`（标出哪些位置是模型生成的回答、哪些是 prompt——只有 response token 计入损失）、`advantages`（或 `rewards`+`group_ids` 由算法换算），可选 `old_logps`（行为策略的对数概率，用于重要性比值）、`values`/`returns`（用 critic 时）、`ref_logps`（算 KL 时）。

## 在 trainall 中怎么用

下面是一段**真实运行过**的最小例子：用注册表拿到 Bradley-Terry 目标，在一个 chosen/rejected batch 上算 RM 损失并反传；末尾展示第三阶段 `build("ppo")` 的入口（PPO 的完整 rollout 见 [RLVR / GRPO](06-rlvr-grpo.md) 的策略梯度示例）。

```python
import torch
import torch.nn as nn
import trainall
from trainall.models import DecoderLM, ArchConfig
from trainall.types import Batch

torch.manual_seed(0)

# 1) 一个极小的策略模型，复用为 reward backbone
cfg = ArchConfig(vocab_size=37, dim=16, n_layers=2, n_heads=4, n_kv_heads=2,
                 ffn_dim=32, max_seq_len=32)
model = DecoderLM.from_config(cfg)

# 2) Bradley-Terry 奖励模型目标（reward_model / 别名 bt、rm 都可）
obj = trainall.build("reward_model", category="objective")

# 3) 一个 chosen/rejected 偏好 batch
def ids(b=4, t=6, v=37):
    return torch.randint(0, v, (b, t))

cids, rids = ids(), ids()
batch = Batch(
    tensors=dict(
        chosen_input_ids=cids,
        chosen_attention_mask=torch.ones_like(cids),
        rejected_input_ids=rids,
        rejected_attention_mask=torch.ones_like(rids),
    ),
    # DecoderLM 没有 value head -> 提供一个标量头，把隐状态映射到一个分数
    extra={"scalar_head": nn.Linear(model.config.vocab_size, 1)},
)

loss, metrics = obj.compute_loss(model, batch)
print("BT loss      =", float(loss.detach()))
print("pairwise acc =", metrics["acc"])
print("reward_margin=", metrics["reward_margin"])
loss.backward()
print("grad ok      =", any(p.grad is not None for p in model.parameters()))

# 第三阶段 PPO 在同一注册表里：trainall.build("ppo", category="objective")
ppo = trainall.build("ppo", category="objective", clip_range=0.2)
print("ppo objective=", type(ppo).__name__)
```

运行输出（CPU）：

```
BT loss      = 0.6856727004051208
pairwise acc = 0.75
reward_margin= 0.015163160860538483
grad ok      = True
ppo objective= PPOObjective
```

`PPOObjective` 在 batch 里提供 `advantages` / `response_mask` 时即可 `compute_loss`；典型完整链路是 `Rollout` 采样 → `VerifierReward`/RM 打分 → `compute_group_advantages` 或 GAE 算优势 → `Trainer` 用 `ppo` 目标更新。注册表里也直接有 `recipe` 形态：`trainall.build("rlvr")` / `trainall.build("frontier")`。

## 何时用 / 何时不用

**适合用 RLHF（RM + PPO）：**
- 目标是**主观/相对**的（有用性、安全、风格、诚实），写不出参考答案，但人类能可靠地"二选一"。
- 你能负担在线采样循环、有调 PPO 的工程能力，并希望通过**在线探索**拿到比静态偏好数据更高的上限。
- 需要灵活的奖励塑形（多个 RM 加权、规则奖励混合、安全惩罚项）。

**不适合 / 优先考虑别的：**
- 答案有**可验证 oracle**（数学、代码、SQL）→ 直接用 [RLVR / GRPO](06-rlvr-grpo.md)，免去训 RM 与奖励黑客。
- 只有**静态偏好对**、要快要稳要复现 → 用 [DPO 等偏好优化](04-preference-optimization.md)，省掉 RM 与在线 PPO。
- 还没做扎实的 [SFT](03-sft.md) → 先 SFT。RLHF 是在一个像样的 SFT 模型之上做对齐，不是从头教任务。

## 常见陷阱与实践要点

- **奖励黑客**是默认结局而非意外。务必盯住 KL、回答长度、以及一组"留出 prompt"的人工/强模型评测，而不只看 RM 分数上涨——RM 分涨而真实质量降是过优化的典型信号。
- **KL 缰绳 $\beta$ 要调**。常用做法是 target-KL 自适应控制器：设定每步目标 KL，动态调 $\beta$。$\beta$ 太小炸成黑客，太大几乎不动。
- **RM 是上限**。PPO 不可能优化出 RM 不认识的"好"。RM 的覆盖面、标注一致性、抗分布漂移能力决定一切；RM 弱时 PPO 越优化越糟。
- **奖励标度无意义**。Bradley-Terry 只约束分差，RM 的绝对值会漂移；做奖励白化 (whitening)/归一化能稳住 PPO 的优势估计。
- **长度偏置**。RM 常把"更长"误当"更好"，PPO 随即把回答越写越长。可加长度惩罚或在 RM 训练里做长度去偏。
- **四模型显存**。policy + reference + reward + critic 同时在显存里；用 LoRA（见 [LoRA / QLoRA](10-lora-qlora.md)）、共享骨干、或换 [GRPO](06-rlvr-grpo.md)（去掉 critic）来省显存。
- **RLAIF 的偏置会被继承**。用 AI 标注偏好省钱，但 judge 模型的口味/偏见会原样灌进 RM，再被 PPO 放大。

## 相关

- [偏好优化 / DPO](04-preference-optimization.md) —— 把 RM+PPO 折叠成一步的直接偏好损失。
- [RLVR / GRPO](06-rlvr-grpo.md) —— 用可验证奖励替代学出来的 RM；GRPO 去掉 critic。
- [Agentic RL](07-agentic-rl.md) —— 多步、带工具的策略优化。
- [过程监督 / PRM](09-process-supervision.md) —— 奖励到步骤级，缓解结果奖励的稀疏与黑客。
- [SFT](03-sft.md) —— RLHF 的起点与 reference 模型来源。
- [LoRA / QLoRA](10-lora-qlora.md) —— 缓解多模型显存压力。
- 术语表：[DPO](../GLOSSARY.md#dpo) · [PPO](../GLOSSARY.md#ppo) · [RLHF](../GLOSSARY.md#rlhf) · [KL](../GLOSSARY.md#kl)
- 返回[方法索引](README.md)。

> 参考文献：Christiano et al. 2017 (*Deep RL from Human Preferences*)；Ouyang et al. 2022 (*InstructGPT*)；Schulman et al. 2017 (*PPO*)；Schulman et al. 2016 (*GAE*)；Bradley & Terry 1952；Gao et al. 2023 (*Scaling Laws for Reward Model Overoptimization*)；Rafailov et al. 2023 (*DPO*)。
