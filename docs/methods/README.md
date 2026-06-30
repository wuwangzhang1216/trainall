# 方法索引 (Methods Index)

<sub>🌐 中文 · <a href="en/README.md">English</a></sub>

> `trainall` 的「方法地图」：现代 LLM 训练栈被拆成 11 篇自包含的深度文档，每篇都给出**直觉 → 原理与架构 → 目标函数（数学）→ 数据长什么样 → 在 trainall 中怎么用 → 何时用/何时不用 → 常见陷阱**。先用这页选对方法，再点进去看细节。配套的 [术语表 / 词典库 (GLOSSARY)](../GLOSSARY.md) 收录了 150+ 个概念的中文速查。

`trainall` 的核心心法是把每次训练拆成**三条正交的轴**——**数据 (data)** 决定*学什么*、**目标 (objective)** 决定*被奖励成为什么*、**算法 (algorithm)** 决定*更新哪些参数*。换一个字符串就能在 SFT → DPO → GRPO 之间切换，而数据与优化器原地不动。

![三条正交的轴：data / objective / algorithm](../assets/three_axes.png)

把这些方法**组合**成一条流水线，才是 2026 年最有价值的东西。典型的前沿管线是：领域语料 → CPT/DAPT → SFT → 拒绝采样/合成数据 → DPO → RLVR·GRPO → Agentic RL → 蒸馏出小部署模型。

![前沿训练流水线：从领域语料到小部署模型的流水线](../assets/frontier_pipeline.png)

---

## 11 篇方法文档

| # | 文档 | 一句话 | 何时用 |
|---|------|--------|--------|
| 01 | [预训练 (Pre-training)](01-pretraining.md) | 海量无标注文本上做 next-token 预测，把世界的统计规律压进权重 | 从零造基座、引入全新语言/模态，且手握十亿/万亿 token 通用语料 |
| 02 | [继续预训练 (CPT / DAPT)](02-continued-pretraining.md) | 同一个 next-token 目标在领域语料上「接着练」，靠 replay 防遗忘 | 模型不*知道*你的领域、且有大量无标注领域文本；通常排在 SFT 之前 |
| 03 | [监督微调 (SFT)](03-sft.md) | 在「提示→理想回答」上做交叉熵、只对回答计损，把能力塑形成行为 | 有高质量示范、想教听指令/按格式作答；几乎所有对齐流水线第一步 |
| 04 | [偏好优化 (DPO 等)](04-preference-optimization.md) | 把成对偏好当分类/回归目标，让策略自己变成隐式奖励模型 | 写不出*唯一*答案但能判「A 比 B 好」：语气、风格、安全拒答 |
| 05 | [RLHF / RLAIF](05-rlhf.md) | 学一个奖励模型把人类偏好压成标量，再用 PPO 优化、KL 拴住 | 目标主观/相对、写不出参考答案，且有能力调 PPO、做在线探索 |
| 06 | [RLVR + GRPO](06-rlvr-grpo.md) | 用确定性 verifier 当奖励、组内归一化估优势，扔掉价值网络 | 正确性*可程序判定*：数学、代码、SQL、结构化抽取、可校验格式 |
| 07 | [智能体强化学习 (Agentic RL)](07-agentic-rl.md) | 把单轮回答扩成多步「观察→调工具→看结果」，凭整段轨迹拿奖励 | 任务需与外部交互才能解（工具/数据库/代码修复）且环境可复现 |
| 08 | [蒸馏与数据飞轮](08-distillation-and-selfplay.md) | 用更强 teacher 或廉价 verifier 造监督信号，让模型反复自练 | 有强 teacher 但无标注团队，或有可验证任务想低成本造 SFT 数据 |
| 09 | [过程监督 (PRM)](09-process-supervision.md) | 不只判答案对错，而逐步判「这一步推理对不对」，信号更密更可定位 | 多步推理、想做 best-of-N 重排或给 RL 提供密集过程奖励 |
| 10 | [LoRA / QLoRA / 全参](10-lora-qlora.md) | 效率轴：冻结基座、旁路一个低秩可训练增量，不改训练目标 | 显存紧张做风格/领域适配；QLoRA 让单卡微调 30B+ 巨模型 |
| 11 | [模型架构 (Architectures)](11-architectures.md) | 所有方法运行其上的底座：RMSNorm/RoPE/GQA/MLA/SwiGLU/MoE | 选型与从零搭建 decoder-LM、调一份 ArchConfig 捏出任意变体 |

---

## 怎么读

- **第一次来**：按 01 → 11 顺序读，理解「数据/目标/算法」三轴如何贯穿全栈。
- **要选方法**：看上表「何时用」一列，或先读各文档的「何时用/何时不用」小节。
- **查术语**：任何不懂的词去 [术语表 / GLOSSARY](../GLOSSARY.md) 速查（含公式与交叉链接）。
- **要跑代码**：每篇的「在 trainall 中怎么用」都是 CPU 可跑、已实际运行过的最小示例。

返回项目主页：[../../README.md](../../README.md) · 设计文档：[../DESIGN.md](../DESIGN.md) · 概念速查：[../CONCEPTS.md](../CONCEPTS.md)
