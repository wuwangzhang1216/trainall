<!-- nav -->
<p align="center">
  <a href="06-rlvr-grpo.md">← RLVR+GRPO</a> ·
  <a href="README.md">索引</a> ·
  <a href="../GLOSSARY.md">术语表</a> ·
  <a href="en/07-agentic-rl.md">English</a> ·
  <a href="08-distillation-and-selfplay.md">蒸馏与自博弈 →</a>
</p>
<!-- /nav -->

# 智能体强化学习 (Agentic RL)

> **把"一次回答"换成"多步交互"：模型在一个环境里反复 观察 → 规划 → 调用工具 → 看结果，最后凭整段轨迹的成败拿奖励，再用 GRPO/PPO 学会更好的行动策略。**

![Agentic RL 的 observe→plan→act→result→reward 循环](../assets/agentic_rl.png)

## 直觉：它到底在做什么

单轮 RLVR（见 [RLVR / GRPO](06-rlvr-grpo.md)）里，模型把问题一次性写完答案，verifier 打一个分。但很多真实任务不是"写一段话"就能解决的：要解一道需要精确算术的数学题、要查一个数据库、要跑一段代码看报错再改、要在网页里点几下才能找到答案。这些任务的共同点是——**答案藏在与外部世界的交互里**，模型必须先"动手"拿到中间结果，才知道下一步该做什么。

Agentic RL 就是把语言模型放进一个**环境 (environment)**：

1. 环境给一个**观察 (observation)**（题面、上一步工具返回的结果）；
2. 模型作为**策略 (policy)** 输出一个**动作 (action)**——通常是一个工具调用字符串，比如 `calculator: 3 + 4`，或一个最终答案 `answer: 7`；
3. 环境执行动作：若是工具调用就 dispatch 给工具、把输出作为下一个观察喂回去；若是最终答案就判定成败、给奖励、结束 episode；
4. 重复，直到成功、提交答案或耗尽步数预算。

整个过程叫一个 **episode（回合）**，由若干 **transition（单步：观察→动作→奖励）** 串成。训练时，我们让模型把同一个任务跑很多遍，成功的回合奖励高、失败的低，再用策略梯度把概率质量推向"能走通"的动作序列。这就是 agentic RL：**学的不是"说什么"，而是"在一个会回话的世界里怎么一步步做"。**

## 原理与架构（深度讲解）

### data → objective → algorithm 的三段式

- **data（数据/环境）**：这里"数据"不是静态的 (prompt, answer) 对，而是一个**可复现的环境** + 一批任务样本 `Sample`。环境定义了动作空间（有哪些工具）、状态转移（动作如何改变观察）、以及奖励信号（什么算成功）。关键是**可复现性**：同一个 `Sample` + 同一个 policy 必须产生同一条轨迹，否则奖励是噪声，梯度无法收敛。trainall 的工具都是纯函数式、确定性的（`CalculatorTool` 走 AST 求值，`PythonTool` 在隔离子进程里跑），就是为了让环境可复现。
- **objective（目标）**：episode 被压平成 `Trajectory`（`response` = 拼接的所有动作，`reward` = 回合成败），然后交给和单轮 RLVR 完全相同的策略梯度目标——`GRPOObjective` / `PPOObjective` / `RLOOObjective`。也就是说，**agentic 只改了"轨迹怎么来"，没改"梯度怎么算"**。
- **algorithm（算法/参数高效）**：底层仍是 `full` / `lora` / `qlora` 微调，与目标解耦。

### 模型真正学到了什么

策略梯度告诉模型："在观察 $o$ 下，你采的动作 $a$ 导致的回合，如果最终比同组的平均好，就提高 $\pi(a\mid o)$；否则降低。" 注意奖励是**回合级**的（outcome reward），但它会被**摊派 (credit assignment)** 到回合里的每一个 token。于是模型逐渐学会：

- **什么时候该调工具 vs 直接回答**（不会算的算术先丢给 `calculator`）；
- **怎么读懂工具返回的观察**（看到 `7` 才提交 `answer: 7`）；
- **如何从错误中恢复**（工具报 `error: ...` 时换一种调用方式）。

这些都不是显式教的，而是从"哪条完整轨迹拿到了奖励"里反推出来的。

### 核心难点

**稀疏、长程奖励 (sparse long-horizon reward)。** 一个 10 步的回合，可能只有最后一步"提交答案"才给 1 分，中间 9 步全是 0。回合越长，能拿到非零奖励的轨迹越稀少，方差越大、学习越慢。两个缓解手段：

1. **过程奖励 (process reward)**：给中间步骤也打分（调对了工具 +0.1、调错了 −0.1），让梯度信号更密。trainall 里通过 `step_penalty`（每步小惩罚，鼓励短路径）和 `AgenticRunner(process_reward_weight=...)`（把每步奖励之和按权重加到 outcome 上）实现。最终标量奖励是 `reward = outcome + process_reward_weight * Σ step_reward`。
2. **课程 (curriculum)**：先给简单任务，逐步加难，保证总有一部分轨迹能成功、提供学习信号。

**误差传播 (error propagation)。** 这是 agentic 区别于单轮 RL 的本质问题：第 2 步走错，第 3、4、5 步的观察全部基于这个错误状态，整条后缀都被"污染"。单轮 RL 里一个 token 错了只影响局部；agentic 里一个动作错了会**沿时间轴放大**。这让 credit assignment 更难——是哪一步真正导致了失败？GRPO 用"同组对比"部分回避了精确归因（不需要知道哪一步错，只要知道这条轨迹整体比同组差就压低它），但代价是信号更粗。

**可复现环境 (reproducible env)。** 如果工具有随机性、有网络副作用、或带时间戳，同一动作两次返回不同观察，奖励就不可信。务必让环境确定化：固定随机种子、把外部调用 mock 成固定响应、给工具加超时与沙箱（`PythonTool` 用子进程 + timeout 正是为此）。

### episode 如何变成 GRPO 的 Trajectory

`AgenticRunner.run(sample)` 驱动一个完整 episode；`._to_trajectory` 把它压平：

- `response` = 所有动作用换行拼接（这就是要被打分、被求梯度的"生成"）；
- `reward` = outcome（终止 transition 的奖励，或外部 `Reward` 重新打分）+ `process_reward_weight × Σ step_reward`；
- `group_id` = 同一个 prompt 跑出来的若干轨迹共享一个 group。

把**同一个任务跑 N 遍**，得到 N 条共享 `group_id` 的 `Trajectory`，再 `compute_group_advantages` 求组内标准化优势，就得到了 `GRPOObjective` 需要的输入——这一步把"多步交互"无缝接回了标准 RLVR 管线。

## 目标函数（数学）

一个 episode 是观察-动作序列 $\tau = (o_0, a_0, o_1, a_1, \dots, o_{T-1}, a_{T-1})$，其中动作由策略采样 $a_t \sim \pi_\theta(\cdot \mid o_t)$。这条轨迹的标量奖励是终局结果加上加权过程奖励：

$$
R(\tau) \;=\; R_{\text{outcome}}(\tau) \;+\; \lambda_{\text{proc}} \sum_{t=0}^{T-1} r_t
$$

其中 $R_{\text{outcome}}$ 是成败奖励（成功 1、失败 0，或 verifier 给的连续分），$r_t$ 是第 $t$ 步的过程奖励（含 $-\text{step\_penalty}$），$\lambda_{\text{proc}}$ 是过程奖励权重。

对同一个任务采样一组 $G$ 条轨迹 $\{\tau_i\}_{i=1}^G$，GRPO 用**组内标准化**得到优势（不需要价值网络）：

$$
A_i \;=\; \frac{R(\tau_i) - \operatorname{mean}\big(\{R(\tau_j)\}_{j=1}^G\big)}{\operatorname{std}\big(\{R(\tau_j)\}_{j=1}^G\big) + \varepsilon}
$$

策略梯度目标（与单轮 GRPO 同形，把每条轨迹的全部动作 token 当作受 $A_i$ 监督的生成）：

$$
\mathcal{L}(\theta) \;=\; -\,\mathbb{E}_i\!\left[\min\!\Big(\rho_i\, A_i,\ \operatorname{clip}(\rho_i,\,1-\epsilon,\,1+\epsilon)\,A_i\Big)\right] \;+\; \beta\, \mathrm{KL}\!\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big)
$$

符号说明：

- $\rho_i = \dfrac{\pi_\theta(\tau_i)}{\pi_{\theta_{\text{old}}}(\tau_i)}$ 是新旧策略对该轨迹动作的重要性比；
- $A_i$ 是上面的组内优势，对回合内每个 token 共享；
- $\epsilon$（`clip_range`，默认 0.2）裁剪比值防止单步更新过大；
- $\beta$（`kl_coef`）把策略拉住、别偏离参考模型太远；
- $\varepsilon$（`eps`，默认 1e-6）防止零方差组除零——这正是为什么当一组轨迹奖励全相等时优势恒为 0（没有对比信息，不更新）。

关键直觉：奖励是回合级的 $R(\tau)$，但梯度通过 $\rho_i$ 摊派到轨迹里每一个动作 token——这就是 agentic 的 **credit assignment**。

## 数据长什么样

输入端是**任务样本** `Sample`（不是预先生成好的轨迹）：

```python
from trainall.types import Sample

Sample(prompt="reach 7", reference=7.0)   # prompt=任务描述, reference=判成败的真值
```

环境拿 `Sample.reference` 作为成败判据。`AgenticRunner` 跑出 episode 后压平成 `Trajectory`，这才是喂给 GRPO 的东西：

```python
from trainall.types import Trajectory

Trajectory(
    prompt="reach 7",
    response="calculator: 3 + 4\nanswer: 7",  # 所有动作拼接
    reward=1.0,                                # outcome + 加权 process
    group_id=0,                                # 同一任务的多条轨迹共享
    advantage=None,                            # 待 compute_group_advantages 填入
    meta={"success": True, "num_steps": 2,
          "outcome_reward": 1.0, "process_reward": 0.0},
)
```

`compute_group_advantages` 会就地填好每条的 `.advantage`。再之后，`GRPOObjective` 把这批 `Trajectory` collate 成 policy-gradient 的 `Batch`（`input_ids` / `attention_mask` / `response_mask` / `rewards` / `group_ids`），梯度只作用在 `response_mask=1` 的动作 token 上。

## 在 trainall 中怎么用

下面这段在 CPU 上、无 torch 即可运行：用 `ExpressionEnv`（一个调用计算器、达到目标数即成功的可复现环境）+ 两个脚本化的 callable policy（真实场景下换成采样的 LM），跑出 episode，压平成共享 group 的 `Trajectory`，再算组内优势——正好是喂给 `GRPOObjective` 的形态。

```python
from trainall.rl import (
    AgenticRunner, MultiStepEnv, ToolRegistry, CalculatorTool,
    compute_group_advantages,
)
from trainall.rl.environment import ExpressionEnv
from trainall.types import Sample, Trajectory

# 1) 可复现的工具型环境：用计算器达到目标数，再提交答案；成功 = 精确数值匹配。
env = ExpressionEnv()          # 动作空间 = {calculator}；可验证奖励

# 2) policy 就是 observation -> action（环境能懂的字符串）。
#    真实 policy 是采样的 LM；这里脚本化两个。
def good_policy():
    st = {"i": 0}
    def pol(_obs):
        st["i"] += 1
        return "calculator: 3 + 4" if st["i"] == 1 else "answer: 7"
    return pol

def bad_policy():
    def pol(_obs):
        return "calculator: 3 + 4" if not _obs.endswith("7") else "answer: 99"
    return pol

# 3) 驱动一个 episode，观察 observe -> act -> result -> reward 的轨迹。
ep = env.rollout(good_policy(), sample=Sample(prompt="reach 7", reference=7.0), max_steps=5)
print("episode  success:", ep.success, "total_reward:", ep.total_reward, "steps:", len(ep))

# 4) 构造一个 GRPO 组：同一个 prompt 跑 N 遍 -> N 条共享 group_id 的 Trajectory。
#    AgenticRunner.run 驱动整段 episode；._to_trajectory 把它压平
#    (response = 动作拼接, reward = 回合成败)。
sample = Sample(prompt="reach 7", reference=7.0)
trajs = []
for make in (good_policy, good_policy, bad_policy, bad_policy):
    runner = AgenticRunner(ExpressionEnv(), make(), max_steps=5)
    episode = runner.run(sample)
    trajs.append(runner._to_trajectory(episode, sample, group_id=0))

for t in trajs:
    print(f"  reward={t.reward:.1f} success={t.meta['success']} steps={t.meta['num_steps']}")

# 5) outcome 奖励 -> 组内优势，正好是 GRPOObjective 需要的输入。
compute_group_advantages(trajs)
print("advantages:", [round(t.advantage, 3) for t in trajs])
```

实际运行输出：

```
episode  success: True total_reward: 1.0 steps: 2
  reward=1.0 success=True steps=2
  reward=1.0 success=True steps=2
  reward=0.0 success=False steps=2
  reward=0.0 success=False steps=2
advantages: [1.0, 1.0, -1.0, -1.0]
```

成功的两条轨迹优势 $+1$、失败的两条 $-1$——把这批带优势的 `Trajectory` 交给 `GRPOObjective`，就能把策略推向"先算后答"的正确行为序列。要重打分（比如换 reward model 或 verifier），给 `AgenticRunner(reward=...)` 传一个 `Reward` 即可；要密化信号，调 `process_reward_weight` 与环境的 `step_penalty`。

## 何时用 / 何时不用

**适合：**

- 任务需要**与外部交互**才能解：工具调用（计算器/代码/检索）、多轮数据库或 API、网页操作、需要执行反馈的代码修复。
- 有**可复现、可验证**的环境与成败判据（数值匹配、单元测试通过、SQL 结果正确）。
- 你能负担**多回合采样**的算力（每个任务跑 N 遍、每遍多步）。

**不适合：**

- 单步就能答完、无需中间结果的任务——直接用单轮 [RLVR / GRPO](06-rlvr-grpo.md) 更省。
- 奖励无法自动判定（开放式写作、主观偏好）——那是 [偏好优化](04-preference-optimization.md) 或 [RLHF](05-rlhf.md) 的地盘。
- 环境不可复现（带网络/时间/随机副作用且无法 mock）——奖励是噪声，先把环境确定化再说。
- 还没有一个能稳定遵循指令、会调用工具格式的基座——先 [SFT](03-sft.md) 教会工具调用格式，再上 agentic RL。

## 常见陷阱与实践要点

- **先 SFT 再 RL**：策略得先会"输出合法的工具调用字符串"，否则一开始几乎没有轨迹能走通，奖励全 0、学不动。用少量轨迹做 SFT 冷启动（工具调用格式 + 提交格式）。
- **零方差组 = 零梯度**：如果一组 N 条轨迹全成功或全失败，组内 std≈0，优势全 0，这组白跑。监控成功率，理想区间是组内有成有败（约 0.2–0.8）；用 [课程 (curriculum)](08-distillation-and-selfplay.md) 把难度调到这个区间。
- **控制回合长度**：长程稀疏奖励是头号方差来源。设合理的 `max_steps`、加 `step_penalty` 鼓励短路径、必要时用过程奖励密化信号。
- **误差传播会放大**：一条轨迹早期走错会污染整条后缀，GRPO 只能粗粒度地压低整条。若发现模型反复在同一步翻车，考虑加针对那一步的过程奖励，或拆成更短的子任务。
- **沙箱与超时**：工具会执行模型生成的内容（代码、表达式），务必隔离（子进程 + timeout，像 `PythonTool`）、禁用危险调用（`CalculatorTool` 拒绝名字/属性/函数调用），否则 RL 会主动找到并利用副作用来"刷分"（reward hacking）。
- **奖励别只看终局**：纯 outcome 奖励对长任务太稀疏；纯 process 奖励又容易被刷（学会"看起来在做事"但不解决问题）。两者按 `process_reward_weight` 折中，并保证 process 奖励本身是可验证的。
- **可复现是底线**：固定种子、mock 外部依赖、确保同一 (sample, policy) 复现同一轨迹。不可复现的环境下，任何奖励曲线都不可信。

## 相关

- [RLVR / GRPO](06-rlvr-grpo.md)：本文复用的单轮策略梯度目标与组内优势。
- [RLHF](05-rlhf.md)：用奖励模型而非可验证 verifier 的 PPO 路线。
- [偏好优化](04-preference-optimization.md)：无 rollout 的离线对齐替代方案。
- [过程监督 (PRM)](09-process-supervision.md)：给中间步骤打分的过程奖励模型，与本文的 process reward 同源。
- [蒸馏与自博弈](08-distillation-and-selfplay.md)：用课程与自博弈生成 agentic 训练任务。
- [SFT](03-sft.md)：agentic RL 之前的工具调用冷启动。
- [LoRA / QLoRA](10-lora-qlora.md)：在 RL 阶段做参数高效微调。
- 术语表：[GRPO](../GLOSSARY.md#grpo)、[RLVR](../GLOSSARY.md#rlvr)、[Trajectory](../GLOSSARY.md#trajectory)、[reward hacking](../GLOSSARY.md#reward-hacking)。
- 返回 [方法索引](README.md)。
