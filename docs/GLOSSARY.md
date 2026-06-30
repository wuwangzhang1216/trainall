# 术语表 / 词典库 (Glossary)

> 一站式查词手册：把 `trainall` 文档里出现的每个重要概念，用 2–5 句中文讲清楚「是什么、为什么、公式、相关」，方便随查随懂。术语按主题分组；每条都给出 GitHub 风格锚点，可被各方法文档直接 `#anchor` 链接。约定：`π_θ` 是被训练的策略 (policy)，`π_ref` 是冻结参考 (reference)，`r` 是奖励，`A` 是优势，`β` 是强度/温度系数，`σ` 是 sigmoid。

## 快速索引 (A–Z)

[adapter](#adapter) ·
[advantage](#advantage) ·
[alpha](#alpha) ·
[anti-collapse](#anti-collapse) ·
[apply_template](#apply_template) ·
[ArchConfig](#archconfig) ·
[attention_mask](#attention_mask) ·
[best-of-N](#best-of-n) ·
[Bradley-Terry](#bradley-terry) ·
[catastrophic forgetting](#catastrophic-forgetting) ·
[causal mask](#causal-mask) ·
[causal-LM loss](#causal-lm-loss) ·
[chat template](#chat-template) ·
[citation verifier](#citation-verifier) ·
[clip / clipped surrogate](#clip--clipped-surrogate) ·
[CoT monitorability](#cot-monitorability) ·
[completion-only loss](#completion-only-loss) ·
[composite verifier](#composite-verifier) ·
[conservative DPO (cDPO)](#conservative-dpo-cdpo) ·
[ContinuedPretrainObjective](#continuedpretrainobjective) ·
[continued pre-training](#continued-pre-training) ·
[CPO](#cpo) ·
[CPT / DAPT](#cpt) ·
[credit assignment](#credit-assignment) ·
[critic / value head](#critic--value-head) ·
[cross-entropy](#cross-entropy) ·
[curriculum](#curriculum) ·
[DAPT](#dapt) ·
[dark knowledge](#dark-knowledge) ·
[DecoderBlock](#decoderblock) ·
[DecoderLM](#decoderlm) ·
[decoupled RoPE](#decoupled-rope) ·
[DeepSeek-R1](#deepseek-r1) ·
[distill / knowledge distillation](#distill) ·
[distribution collapse](#distribution-collapse) ·
[domain-adaptive pre-training](#domain-adaptive-pre-training) ·
[domain_field](#domain_field) ·
[double quantization](#double-quantization) ·
[DPO](#dpo) ·
[efficiency axis](#efficiency-axis) ·
[EOS / stop token](#eos--stop-token) ·
[forward KL](#forward-kl) ·
[full finetune](#full-finetune) ·
[GAE](#gae) ·
[GeGLU](#geglu) ·
[GQA](#gqa) ·
[GRPO](#grpo) ·
[group-relative advantage](#group-relative-advantage) ·
[implicit reward margin](#implicit-reward-margin) ·
[importance ratio](#importance-ratio) ·
[InMemorySource](#inmemorysource) ·
[intrinsic dimension](#intrinsic-dimension) ·
[IPO](#ipo) ·
[k3 KL estimator](#k3-kl-estimator) ·
[KL](#kl) ·
[KL leash](#kl-leash) ·
[KL penalty](#kl-penalty) ·
[KTO](#kto) ·
[KV cache](#kv-cache) ·
[label masking / -100](#label-masking---100) ·
[label smoothing](#label-smoothing) ·
[length normalization](#length-normalization) ·
[load-balancing aux loss](#load-balancing-aux-loss) ·
[log-prob / log-probability](#log-prob--log-probability) ·
[logits](#logits) ·
[LoRA](#lora) ·
[LoRALinear](#loralinear) ·
[low-rank adaptation](#low-rank-adaptation) ·
[mask_prompt](#mask_prompt) ·
[mass-covering](#mass-covering) ·
[Math-Shepherd](#math-shepherd) ·
[math verifier](#math-verifier) ·
[merge_lora](#merge_lora) ·
[MHA](#mha) ·
[MLA](#mla) ·
[mode-seeking](#mode-seeking) ·
[MoE](#moe) ·
[MQA](#mqa) ·
[next-token objective](#next-token-objective) ·
[NF4](#nf4) ·
[NTK scaling](#ntk-scaling) ·
[obfuscated reward hacking](#obfuscated-reward-hacking) ·
[odds ratio](#odds-ratio) ·
[off-policy](#off-policy) ·
[offline preference optimization](#offline-preference-optimization) ·
[on-policy](#on-policy) ·
[ORM](#orm) ·
[ORPO](#orpo) ·
[outcome supervision](#outcome-supervision) ·
[per-step BCE](#per-step-bce) ·
[perplexity](#perplexity) ·
[policy](#policy) ·
[PPO](#ppo) ·
[pre-norm](#pre-norm) ·
[pretraining](#pretraining) ·
[PRM](#prm) ·
[PRM800K](#prm800k) ·
[process reward model](#process-reward-model) ·
[process supervision](#process-supervision) ·
[prompt masking](#prompt-masking) ·
[proposer-solver-verifier](#proposer-solver-verifier) ·
[prospect theory](#prospect-theory) ·
[QLoRA](#qlora) ·
[quantization](#quantization) ·
[rank (r)](#rank-r) ·
[reference model](#reference-model) ·
[reference-free](#reference-free) ·
[REINFORCE baseline](#reinforce-baseline) ·
[rejection sampling](#rejection-sampling) ·
[replay](#replay) ·
[replay_weight](#replay_weight) ·
[reverse KL](#reverse-kl) ·
[reward](#reward) ·
[reward hacking](#reward-hacking) ·
[reward model](#reward-model) ·
[reward over-optimization](#reward-over-optimization) ·
[reward shaping](#reward-shaping) ·
[RFT](#rft) ·
[RLAIF](#rlaif) ·
[RLHF](#rlhf) ·
[RLOO](#rloo) ·
[RLVR](#rlvr) ·
[RMSNorm](#rmsnorm) ·
[rollout](#rollout) ·
[RoPE](#rope) ·
[router / top-k router](#router--top-k-router) ·
[scalar head](#scalar-head) ·
[self-play](#self-play) ·
[self-supervised](#self-supervised) ·
[sequence log-prob](#sequence-log-prob) ·
[SFT](#sft) ·
[shaped reward](#shaped-reward) ·
[SimPO](#simpo) ·
[softmax](#softmax) ·
[STaR](#star) ·
[step_labels](#step_labels) ·
[step_mask](#step_mask) ·
[SwiGLU](#swiglu) ·
[synthetic data flywheel](#synthetic-data-flywheel) ·
[target_modules](#target_modules) ·
[teacher forcing](#teacher-forcing) ·
[temperature](#temperature) ·
[temperature scaling](#temperature-scaling) ·
[tie_embeddings](#tie_embeddings) ·
[token](#token) ·
[top-p (nucleus) sampling](#top-p-nucleus-sampling) ·
[train_on_prompt](#train_on_prompt) ·
[Trajectory](#trajectory) ·
[trust region](#trust-region) ·
[value network](#value-network) ·
[verifiable reward](#verifiable-reward) ·
[verifier](#verifier) ·
[VerifierReward](#verifierreward) ·
[YaRN](#yarn) ·
[zone of proximal development](#zone-of-proximal-development) ·
[监督微调](#监督微调)

---

# 训练范式 (Training Paradigms)

### pretraining
**预训练**  
用海量无标注文本做「下一个 token 预测」的自监督训练，把语言、世界知识与浅层推理压进权重，产出一个只会「续写」、还不会「对话」的 base model（基座模型）。目标函数极简（next-token 交叉熵），但「预测得准」本身需要理解世界，能力因此在压缩压力下涌现。它决定了能力的上限，后续 SFT/对齐只是释放它。
公式：$\mathcal{L}_\text{PT}=-\sum_t \log P_\theta(x_t \mid x_{<t})$
相关：[01-pretraining.md](methods/01-pretraining.md) · [next-token objective](#next-token-objective) · [self-supervised](#self-supervised) · [perplexity](#perplexity) · [continued pre-training](#continued-pre-training)

### continued pre-training
**继续预训练**  
见 [CPT / DAPT](#cpt)。在已有基座上、用同一个 next-token 目标，在领域语料上「接着练」，把领域分布写进权重。本质是数据问题而非损失问题，核心难点是避免遗忘旧能力。
相关：[02-continued-pretraining.md](methods/02-continued-pretraining.md) · [CPT / DAPT](#cpt) · [replay](#replay) · [catastrophic forgetting](#catastrophic-forgetting)

### CPT
继续预训练 (Continued Pre-training) 的注册键与别名，等价于 `dapt`。详见下面的 [DAPT](#dapt) 词条与 [domain-adaptive pre-training](#domain-adaptive-pre-training)。典型管线顺序是 *预训练 → CPT → SFT → 偏好优化*，知识在前、技能在后。
相关：[02-continued-pretraining.md](methods/02-continued-pretraining.md) · [DAPT](#dapt) · [replay_weight](#replay_weight) · [catastrophic forgetting](#catastrophic-forgetting)

### DAPT
领域自适应预训练 (Domain-Adaptive Pre-training)，Gururangan 等 (2020, *Don't Stop Pretraining*) 提出：在领域语料上多跑一轮预训练再做下游微调，几乎在所有领域都优于直接微调基座。`trainall` 里 `ContinuedPretrainObjective` 直接继承 `CausalLMObjective`，默认走父类 fast path，只在启用 batch 内加权时才不同。学习率通常比初始预训练低一个量级。
公式：$\mathcal{L}_\text{CPT}=\frac{\sum_i w_i\,\ell_i}{\sum_i w_i}$（$\ell_i$ 为逐 token 平均 NLL，$w_i$ 为样本权重）
相关：[02-continued-pretraining.md](methods/02-continued-pretraining.md) · [domain-adaptive pre-training](#domain-adaptive-pre-training) · [replay](#replay) · [ContinuedPretrainObjective](#continuedpretrainobjective)

### domain-adaptive pre-training
**领域自适应预训练**  
[DAPT](#dapt) 的全称。强调把领域的「分布先验」写进权重——这是任何提示工程或 RAG（推理期提示）都替代不了的，因为后者无法改变模型内部对 token 序列的概率信念。
相关：[02-continued-pretraining.md](methods/02-continued-pretraining.md) · [DAPT](#dapt) · [CPT](#cpt)

### SFT
**监督微调**  
Supervised Fine-Tuning：在「提示 → 理想回答」配对上做交叉熵，但**只对回答 token 计损失**（prompt 段在 `labels` 里被设为 `-100`）。SFT 不是灌输新知识，而是把基座已有能力**塑形**成「听指令、按格式作答」的行为——所以数据质量远比数量重要（LIMA：1000 条干净样本即可）。它几乎是所有对齐流水线的第一步，并提供后续方法的参考模型。
公式：$\mathcal{L}_\text{SFT}=-\frac{1}{|\mathcal{R}|}\sum_{t\in\mathcal{R}}\log p_\theta(x_t\mid x_{<t})$
相关：[03-sft.md](methods/03-sft.md) · [prompt masking](#prompt-masking) · [completion-only loss](#completion-only-loss) · [label smoothing](#label-smoothing) · [DPO](#dpo)

### 监督微调
即 [SFT](#sft)（Supervised Fine-Tuning）的中文名。给模型看「指令 → 理想回答」示范并让它模仿，损失只落在 response token 上。
相关：[03-sft.md](methods/03-sft.md) · [SFT](#sft) · [completion-only loss](#completion-only-loss)

### RLHF
Reinforcement Learning from Human Feedback：当「好」写不成 loss 时，先用人类成对偏好训一个奖励模型 (reward model) 把偏好压成标量分，再用 PPO 把策略往高分推，同时用 KL 拴住别跑偏。三阶段：收集偏好 → 训 RM → PPO。一句话——RM 学「什么是好」，PPO 学「怎么变好」，KL 负责「别变疯」。
公式：$\max_\pi\ \mathbb{E}[r_\phi(x,y)]-\beta\,\mathrm{KL}(\pi_\theta\Vert\pi_\text{ref})$
相关：[05-rlhf.md](methods/05-rlhf.md) · [reward model](#reward-model) · [PPO](#ppo) · [KL leash](#kl-leash) · [RLAIF](#rlaif)

### RLAIF
Reinforcement Learning from AI Feedback：RLHF 的变体，把「人类标注偏好」换成「更强模型 (LLM-as-judge) 标注偏好」，省下标注成本。代价是 judge 模型的口味/偏见会原样灌进 RM，再被 PPO 放大。
相关：[05-rlhf.md](methods/05-rlhf.md) · [RLHF](#rlhf) · [reward model](#reward-model)

### RLVR
Reinforcement Learning from Verifiable Rewards：用一个能自动判对错的确定性 **verifier**（数学答案、单测、SQL 结果集）取代「学出来的」奖励模型。奖励是 ground truth，不会被刷分。RLVR 解决「奖励从哪来」，配套的 GRPO 解决「优势怎么估」，二者是 2024–2025 推理模型 (DeepSeek-R1) 的训练主干。关键认知：RLVR **放大**基座已偶尔采到的正确路径，**不创造**全新能力。
公式：$r_i=V(o_i)\in[0,1]$，再经 [group-relative advantage](#group-relative-advantage) 归一化
相关：[06-rlvr-grpo.md](methods/06-rlvr-grpo.md) · [GRPO](#grpo) · [verifiable reward](#verifiable-reward) · [verifier](#verifier) · [DeepSeek-R1](#deepseek-r1)

### agentic RL
**智能体强化学习**  
把「一次回答」换成「多步交互」：模型在环境里反复 观察 → 规划 → 调用工具 → 看结果，最后凭整段轨迹 (episode) 的成败拿奖励，再用 GRPO/PPO 学行动策略。它只改了「轨迹怎么来」，没改「梯度怎么算」——episode 被压平成 `Trajectory` 后无缝接回标准 RLVR 管线。难点：稀疏长程奖励、误差传播、可复现环境。
公式：$R(\tau)=R_\text{outcome}(\tau)+\lambda_\text{proc}\sum_t r_t$
相关：[07-agentic-rl.md](methods/07-agentic-rl.md) · [credit assignment](#credit-assignment) · [Trajectory](#trajectory) · [process reward model](#process-reward-model) · [GRPO](#grpo)

### distillation
**知识蒸馏**  
见 [distill / knowledge distillation](#distill)。用更强 teacher 的完整概率分布（软标签）训练 student，比 one-hot 硬标签信息量大得多。是数据飞轮里「向更聪明的人抄」那一支。
相关：[08-distillation-and-selfplay.md](methods/08-distillation-and-selfplay.md) · [distill](#distill) · [dark knowledge](#dark-knowledge)

### synthetic data flywheel
**合成数据飞轮**  
proposer 出题 → solver 采样多个答案 → verifier 过滤掉错的 → 留下的正确轨迹训模型 → 更强模型出更难的题，循环自增强。前提是「验证比生成便宜」这种非对称性。最大失效模式是 [distribution collapse](#distribution-collapse)，靠 [curriculum](#curriculum) 的难度自适应与多样性监控对抗。
相关：[08-distillation-and-selfplay.md](methods/08-distillation-and-selfplay.md) · [proposer-solver-verifier](#proposer-solver-verifier) · [rejection sampling](#rejection-sampling) · [self-play](#self-play)

### self-play
**自我对弈**  
`SelfPlayLoop`：多轮迭代的飞轮——每轮按当前难度出 `tasks_per_round` 道题、每题采 `k` 个候选、验证去重保留，再用本轮通过率 `update` 课程进入下一轮。把模型一直钉在「够得着但不轻松」的学习区。它的上限被 verifier 可靠性与多样性控制卡死。
相关：[08-distillation-and-selfplay.md](methods/08-distillation-and-selfplay.md) · [curriculum](#curriculum) · [zone of proximal development](#zone-of-proximal-development) · [anti-collapse](#anti-collapse)

### process supervision
**过程监督**  
不要只问「答案对不对」，而要逐步问「这一步推理对不对」——把监督信号从结果细化到推理链 (chain-of-thought) 的每一步。训练出的 [PRM](#prm) 能在链条中途就指出第一个出错步骤，信号更密、更准、更可定位。**安全警示**：直接惩罚展示出来的 CoT 会教模型学会「藏」（[obfuscated reward hacking](#obfuscated-reward-hacking)）。
公式：见 [per-step BCE](#per-step-bce)
相关：[09-process-supervision.md](methods/09-process-supervision.md) · [PRM](#prm) · [outcome supervision](#outcome-supervision) · [CoT monitorability](#cot-monitorability)

---

# 偏好优化 (Preference Optimization)

### offline preference optimization
**离线偏好优化**  
不训练奖励模型、不跑在线采样，直接把成对偏好 `(chosen, rejected)` 当作一个分类/回归目标，让策略自己变成隐式奖励模型。「离线」指数据是固定、静态、可复用的偏好对，没有采样器、没有 environment。本族 6 个成员共享一个目标（抬高 chosen、压低 rejected），差异在 link 函数与正则对象。
相关：[04-preference-optimization.md](methods/04-preference-optimization.md) · [DPO](#dpo) · [reference model](#reference-model) · [implicit reward margin](#implicit-reward-margin)

### DPO
Direct Preference Optimization (Rafailov 2023)：偏好族的主力。从 RLHF「最大化奖励 + KL 约束」的闭式最优解里把奖励反解成「策略对参考的对数概率比」，配分函数 $Z(x)$ 在相减时被消掉，于是不用 RM、不用在线采样、不用 critic，整个 PPO 阶段塌缩成对固定偏好对的监督学习。隐式奖励差 = 隐式奖励边际。其梯度自带难例挖掘（模型排错时梯度才大）。
公式：$\mathcal{L}_\text{DPO}=-\log\sigma\big(\beta[(\log\pi_\theta(y_w)-\log\pi_\text{ref}(y_w))-(\log\pi_\theta(y_l)-\log\pi_\text{ref}(y_l))]\big)$
相关：[04-preference-optimization.md](methods/04-preference-optimization.md) · [Bradley-Terry](#bradley-terry) · [implicit reward margin](#implicit-reward-margin) · [reference model](#reference-model) · [IPO](#ipo)

### IPO
Identity Preference Optimization (Azar 2023)：把 DPO 的 log-sigmoid 换成**平方损失**，将（长度归一化的）隐式奖励边际**回归**到一个有限目标 $\frac{1}{2\beta}$，根治 DPO 在确定性/无噪偏好上把边际推到无穷的过拟合病灶。仍需成对数据与参考模型。
公式：$\mathcal{L}_\text{IPO}=\big(h-\tfrac{1}{2\beta}\big)^2$
相关：[04-preference-optimization.md](methods/04-preference-optimization.md) · [DPO](#dpo) · [length normalization](#length-normalization) · [reference model](#reference-model)

### KTO
Kahneman-Tversky Optimization (Ethayarajh 2024)：放弃成对假设，用**前景理论**的效用函数处理**单条**带 desirable/undesirable 标签的样本——不需要把 chosen 和 rejected 配对。适合天然「单条好/坏」的数据（点赞/点踩、人工审核通过/拒绝）。注意 `trainall` 实现里 KTO 仍 `requires_reference_model=True`（需要参考对数概率构造对数比与 KL 基线 $z$）。
公式：desirable: $w_d(1-\sigma(\beta(r-z)))$；undesirable: $w_u(1-\sigma(\beta(z-r)))$
相关：[04-preference-optimization.md](methods/04-preference-optimization.md) · [prospect theory](#prospect-theory) · [reference model](#reference-model) · [DPO](#dpo)

### ORPO
Odds Ratio Preference Optimization (Hong 2024)：**免参考**。把 SFT 与偏好对齐折叠进一个损失——标准 SFT 负对数似然 + 一个几率比 (odds ratio) 惩罚把 rejected 推开。省掉单独的 SFT 阶段与参考模型。期望从一个尚未重度对齐的基座出发。
公式：$\mathcal{L}_\text{ORPO}=-\overline{\log\pi_\theta}(y_w)+\lambda\big(-\log\sigma(\log\text{OR})\big)$
相关：[04-preference-optimization.md](methods/04-preference-optimization.md) · [odds ratio](#odds-ratio) · [reference-free](#reference-free) · [SFT](#sft)

### SimPO
Simple Preference Optimization (Meng 2024)：**免参考**，隐式奖励就是**长度归一化**的平均对数概率 $\beta\,\overline{\log\pi_\theta}(y)$，并引入目标边际 $\gamma$ 要求 chosen 至少领先一个安全裕度。长度归一化天然消除 DPO 偏爱长回答的 length bias。注意默认 $\beta=2.0,\gamma=0.5$，量纲与 DPO 不同，不要照搬其值。
公式：$\mathcal{L}_\text{SimPO}=-\log\sigma\big(\beta\,\overline{\log\pi_\theta}(y_w)-\beta\,\overline{\log\pi_\theta}(y_l)-\gamma\big)$
相关：[04-preference-optimization.md](methods/04-preference-optimization.md) · [length normalization](#length-normalization) · [reference-free](#reference-free) · [DPO](#dpo)

### CPO
Contrastive Preference Optimization (Xu 2024)：**免参考**，用一个 NLL 锚（行为克隆）替代参考 KL。基于**未归一化**序列对数概率：对比项拉开 chosen/rejected，NLL 锚防止模型为了拉开边际而牺牲 chosen 的绝对似然。
公式：$\mathcal{L}_\text{CPO}=-\log\sigma\big(\beta(\log\pi_\theta(y_w)-\log\pi_\theta(y_l))\big)+\lambda\big(-\overline{\log\pi_\theta}(y_w)\big)$
相关：[04-preference-optimization.md](methods/04-preference-optimization.md) · [reference-free](#reference-free) · [sequence log-prob](#sequence-log-prob) · [SimPO](#simpo)

### Bradley-Terry
Bradley & Terry (1952) 偏好模型：给定两个回答的潜在分数 $r_c,r_r$，人类偏好 chosen 的概率是 $\sigma(r_c-r_r)$——**只有分差决定偏好概率**，绝对分值无标度（整体平移不改变任何偏好）。它是奖励模型损失的基础，也是 DPO 反解的出发点。监控指标：pairwise accuracy 与 reward margin。
公式：$P(c\succ r)=\sigma(r_c-r_r)$；$\mathcal{L}_\text{RM}=-\log\sigma(r_\phi(x,y_c)-r_\phi(x,y_r))$
相关：[05-rlhf.md](methods/05-rlhf.md) · [reward model](#reward-model) · [DPO](#dpo) · [sigmoid](#sigmoid)

### implicit reward margin
**隐式奖励边际**  
DPO 系把奖励定义为策略相对参考的对数比 $\hat r(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_\text{ref}(y\mid x)}$；chosen 与 rejected 的奖励之差就是隐式奖励边际 $\Delta$。它正是被 sigmoid（DPO）或平方损失（IPO）施加压力的核心量；模型「仍偏向 rejected」时边际为负、梯度大。
公式：$\Delta=(\log\frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)})-(\log\frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)})$
相关：[04-preference-optimization.md](methods/04-preference-optimization.md) · [DPO](#dpo) · [reference model](#reference-model) · [reward](#reward)

### odds ratio
**几率比**  
ORPO 的核心量。把一个回答的「几率」定义为 $\frac{p}{1-p}$（$p$ 为序列概率），几率比就是 chosen 与 rejected 几率之比的对数。它自带正则、无需参考模型，用数值稳定的 `log1mexp`（$\log(1-e^\ell)$）计算。
公式：$\log\text{OR}=(\ell_c-\log(1-e^{\ell_c}))-(\ell_r-\log(1-e^{\ell_r}))$
相关：[04-preference-optimization.md](methods/04-preference-optimization.md) · [ORPO](#orpo) · [reference-free](#reference-free)

### prospect theory
**前景理论**  
Kahneman & Tversky 的行为经济学理论：人对收益/损失的感受相对一个**参照点**衡量，且对损失更敏感（损失厌恶）。KTO 借用其效用函数形状，相对一个共享 KL 基线 $z$ 衡量单条样本是「收益」还是「损失」，从而能在非成对数据上对齐。
相关：[04-preference-optimization.md](methods/04-preference-optimization.md) · [KTO](#kto)

### reference-free
**免参考**  
指偏好目标不需要冻结参考模型 $\pi_\text{ref}$——省掉一半显存与一次前向，流水线更短。ORPO（用 odds-ratio 自带正则）、SimPO（用长度归一化）、CPO（用 NLL 锚）都是免参考。代价：去掉参考这个 KL 锚后失去一道防退化的护栏，需靠长度归一化/边际/锚来补。
相关：[04-preference-optimization.md](methods/04-preference-optimization.md) · [ORPO](#orpo) · [SimPO](#simpo) · [reference model](#reference-model)

### conservative DPO (cDPO)
DPO 的标签平滑变体：承认偏好标签有噪声，给损失加一个系数 $\varepsilon\in(0,0.5)$ 的反向项，防止把边际推到无穷、损害校准。$\varepsilon=0$ 即标准 DPO。
公式：$\mathcal{L}=-(1-\varepsilon)\log\sigma(\beta\Delta)-\varepsilon\log\sigma(-\beta\Delta)$
相关：[04-preference-optimization.md](methods/04-preference-optimization.md) · [DPO](#dpo) · [label smoothing](#label-smoothing)

---

# RL 概念 (Reinforcement Learning Concepts)

### policy
**策略**  
被训练、用来产生动作/文本的模型 $\pi_\theta$。在语言模型 RL 里，policy 就是 LM 本身——给定 prompt/observation，它对下一个 token/动作输出一个概率分布。RL 的目标是更新 $\theta$ 让 policy 产生更高奖励的轨迹。
相关：[05-rlhf.md](methods/05-rlhf.md) · [reference model](#reference-model) · [rollout](#rollout) · [on-policy](#on-policy)

### reference model
**参考模型**  
一份冻结的策略副本 $\pi_\text{ref}$（通常就是 SFT 后的模型），作为 KL 锚把训练中的策略拴在它附近，防止退化、复读、丢失通用能力。DPO/IPO/KTO 需要它来构造对数比；PPO/GRPO 用它算 KL 惩罚。`trainall` 里可经 `batch.extra["ref_model"]` 在线提供，或预计算 `ref_chosen_logps`/`ref_rejected_logps`。
相关：[04-preference-optimization.md](methods/04-preference-optimization.md) · [KL leash](#kl-leash) · [reference-free](#reference-free) · [policy](#policy)

### on-policy
**同策略**  
用**当前**策略采样的数据来更新当前策略。PPO/GRPO 是 on-policy：每轮要「采样 → 打分 → 更新」。优点是分布匹配、能在线探索；缺点是采样贵、不能复用旧数据太多步（靠重要性比值 + clip 有限复用）。
相关：[06-rlvr-grpo.md](methods/06-rlvr-grpo.md) · [off-policy](#off-policy) · [importance ratio](#importance-ratio) · [PPO](#ppo)

### off-policy
**异策略**  
用**别的**（旧的或固定的）策略产生的数据来更新当前策略。DPO 等离线偏好优化是 off-policy：用固定偏好数据、不能在线探索，理论上限可能低于调好的 PPO，但简单、稳定、可复现。
相关：[04-preference-optimization.md](methods/04-preference-optimization.md) · [on-policy](#on-policy) · [offline preference optimization](#offline-preference-optimization)

### reward
**奖励**  
RL 的标量学习信号 $r$，衡量一个回答/轨迹有多好。来源有三：学出来的 [reward model](#reward-model)（RLHF）、确定性 [verifier](#verifier)（RLVR）、或人为塑形的 [shaped reward](#shaped-reward)。`trainall` 里奖励常落在 `[0,1]`，再经组内归一化变成优势。
相关：[06-rlvr-grpo.md](methods/06-rlvr-grpo.md) · [advantage](#advantage) · [verifiable reward](#verifiable-reward) · [reward hacking](#reward-hacking)

### advantage
**优势**  
$A$ 衡量「某个动作/回答比基线好多少」。policy gradient 用 $\nabla\log\pi\cdot A$ 更新——优势为正就抬高该动作概率，为负就压低。减去基线 (baseline) 不改变梯度期望（无偏）却大幅降方差。PPO 用 GAE 估优势，GRPO 用组内 z-score。
公式：$A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}$（GAE）或 $A_i=\frac{r_i-\text{mean}}{\text{std}+\varepsilon}$（GRPO）
相关：[06-rlvr-grpo.md](methods/06-rlvr-grpo.md) · [GAE](#gae) · [group-relative advantage](#group-relative-advantage) · [REINFORCE baseline](#reinforce-baseline)

### GAE
Generalized Advantage Estimation (Schulman 2016)：用一个 critic（value head）估计每个位置的期望回报，再用带衰减 $\gamma\lambda$ 的 TD 残差之和算优势，在偏差与方差之间平滑权衡。`trainall` 里 `compute_gae(rewards, values, gamma, lam, mask)` 实现它，PPO 用它的输出当优势。
公式：$\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$，$A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}$
相关：[05-rlhf.md](methods/05-rlhf.md) · [PPO](#ppo) · [critic / value head](#critic--value-head) · [advantage](#advantage)

### REINFORCE baseline
**REINFORCE 基线**  
朴素策略梯度 $\nabla\log\pi\cdot r$ 方差大（即使最差的答案只要 $r>0$ 也会被推高）。**减去一个基线** $b$ 不改变梯度期望却大幅降方差。PPO 用 value 网络学 $b$；RLOO 用「其他样本的均值」当 $b$；GRPO 用「组内均值」当 $b$。
公式：$\nabla J=\mathbb{E}[(r-b)\nabla\log\pi]$
相关：[06-rlvr-grpo.md](methods/06-rlvr-grpo.md) · [RLOO](#rloo) · [group-relative advantage](#group-relative-advantage) · [advantage](#advantage)

### group-relative advantage
**组内相对优势**  
GRPO 的核心：对同一个 prompt 采样一组 $G$ 个答案，用**组内均值**当基线、除以**组内标准差**做 z-score 归一化得到优势。因为同组答案来自同一道题、奖励直接可比，组均值是该题难度的即时无偏估计，**不需要价值网络**。代价：一组全对或全错时方差为 0、优势全 0、不产生梯度。
公式：$A_i=\frac{r_i-\text{mean}(\{r_1..r_G\})}{\text{std}(\{r_1..r_G\})+\varepsilon}$
相关：[06-rlvr-grpo.md](methods/06-rlvr-grpo.md) · [GRPO](#grpo) · [value network](#value-network) · [advantage](#advantage)

### critic / value head
**评论家 / 价值头**  
一个回归头 $V(s)$，估计某状态/位置的期望未来回报，作为 PPO 优势估计 (GAE) 的基线。它额外占显存、要单独训练，是 GRPO 想删掉的东西——GRPO 用组内统计替代它。RLHF 的「四模型」之一（policy + reference + reward + critic）。
相关：[05-rlhf.md](methods/05-rlhf.md) · [value network](#value-network) · [GAE](#gae) · [GRPO](#grpo)

### value network
**价值网络**  
即 [critic / value head](#critic--value-head) 的网络形态：PPO 用它估基线降方差，GRPO/RLOO 通过组内采样彻底删掉它以省一半显存、少一个要调的网络。
相关：[06-rlvr-grpo.md](methods/06-rlvr-grpo.md) · [critic / value head](#critic--value-head) · [GRPO](#grpo) · [RLOO](#rloo)

### scalar head
**标量头**  
把池化后的隐状态映射到一个实数的线性层。奖励模型用它输出 $r_\phi(x,y)$（取最后一个非 pad token），PRM 用它（或 value head）输出每步分数。`DecoderLM` 自身没有 value head，故 BradleyTerry 训练时经 `batch.extra["scalar_head"]` 注入一个 `nn.Linear`。
相关：[05-rlhf.md](methods/05-rlhf.md) · [reward model](#reward-model) · [critic / value head](#critic--value-head) · [PRM](#prm)

### PPO
Proximal Policy Optimization (Schulman 2017)：on-policy 的 actor-critic。不做朴素策略梯度，而是优化一个**裁剪过的代理目标**，让一批采样数据可复用做多步更新而不会因重要性比值把策略一步推太远（软信任域）。RLHF 第三阶段用它最大化 RM 奖励 + KL 惩罚；需要 value 网络估 GAE 优势。`clip_range=0.2, vf_coef=0.5, kl_coef` 等是其旋钮。
公式：$\mathcal{L}_\text{PPO}=-\mathbb{E}[\min(\rho_t A_t,\text{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)]+c_v\cdot\text{value loss}+\beta\,\text{KL}$
相关：[05-rlhf.md](methods/05-rlhf.md) · [clip / clipped surrogate](#clip--clipped-surrogate) · [importance ratio](#importance-ratio) · [GAE](#gae) · [trust region](#trust-region)

### GRPO
Group Relative Policy Optimization (Shao 2024；DeepSeek-R1 2025)：扔掉 PPO 的价值网络，对同一 prompt 采样一组答案、用组内 z-score 当优势，再套 PPO 的 clipped 替代目标对 response token 做策略梯度，可选加 KL 惩罚。是 RLVR 的主力优化器。反直觉细节：平衡组内 loss 可能恒为 0 但梯度非零、训练正常——看奖励均值和 pass@1，别盯 loss。
公式：$\mathcal{L}_\text{GRPO}=-\frac{1}{\sum|o_i|}\sum_{i,t}\min(\rho_{i,t}A_i,\text{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i)+\beta\,\text{KL}$
相关：[06-rlvr-grpo.md](methods/06-rlvr-grpo.md) · [group-relative advantage](#group-relative-advantage) · [RLVR](#rlvr) · [k3 KL estimator](#k3-kl-estimator) · [clip / clipped surrogate](#clip--clipped-surrogate)

### RLOO
REINFORCE Leave-One-Out (Ahmadian 2024)：和 GRPO 一样删掉价值网络，但基线取法不同——每个样本的基线是同组**其他**样本的平均奖励（留一法），而非含自己的组均值。在 `trainall` 里与 GRPO/PPO 共享同一套 policy-gradient batch 约定。
公式：$b_i=\frac{1}{G-1}\sum_{j\ne i}r_j$
相关：[06-rlvr-grpo.md](methods/06-rlvr-grpo.md) · [REINFORCE baseline](#reinforce-baseline) · [GRPO](#grpo) · [value network](#value-network)

### clip / clipped surrogate
**裁剪代理目标**  
PPO/GRPO 的核心技巧：在重要性比值 $\rho_t=\pi_\theta/\pi_\text{old}$ 上施加 $[1-\epsilon,1+\epsilon]$ 的裁剪，与 $\min$ 一起构成一个悲观下界——优势为正时上限被 $1+\epsilon$ 卡住、为负时下限被 $1-\epsilon$ 卡住。比值跑出区间时该项梯度归零，限制单步更新幅度。`clip_range`（$\epsilon$）默认 0.2。
公式：$\min(\rho_t A_t,\text{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)$
相关：[05-rlhf.md](methods/05-rlhf.md) · [importance ratio](#importance-ratio) · [trust region](#trust-region) · [PPO](#ppo)

### importance ratio
**重要性比值**  
新旧策略对同一动作的概率之比 $\rho_t=\pi_\theta(a_t)/\pi_\text{old}(a_t)=\exp(\log p_\theta-\log p_\text{old})$。它让一批旧策略采样的数据能被复用于更新新策略（重要性采样）；偏离 1 太远说明策略变化太大，被 clip 截断。纯 on-policy 单步时 `old_logps` 缺省，$\rho\equiv1$，目标退化为 REINFORCE。
相关：[06-rlvr-grpo.md](methods/06-rlvr-grpo.md) · [clip / clipped surrogate](#clip--clipped-surrogate) · [on-policy](#on-policy) · [PPO](#ppo)

### trust region
**信任域**  
「每步更新别把策略推太远」的约束。理论上是对新旧策略 KL 的硬约束 (TRPO)；PPO 用 clip 实现一个廉价的「软信任域」近似——超出比值区间就停止给梯度。
相关：[05-rlhf.md](methods/05-rlhf.md) · [clip / clipped surrogate](#clip--clipped-surrogate) · [PPO](#ppo) · [KL](#kl)

### KL
Kullback-Leibler 散度 $\mathrm{KL}(p\Vert q)$，衡量两个分布的差异。RL 里用作把策略拴在参考附近的正则项（KL 惩罚/缰绳），DPO 里是被反解掉的约束，蒸馏里是匹配 teacher 分布的损失。注意它不对称：forward 与 reverse KL 行为迥异。
公式：$\mathrm{KL}(p\Vert q)=\sum_i p_i\log\frac{p_i}{q_i}$
相关：[05-rlhf.md](methods/05-rlhf.md) · [KL penalty](#kl-penalty) · [forward KL](#forward-kl) · [reverse KL](#reverse-kl)

### KL penalty
**KL 惩罚**  
在 RL 目标里加一项 $\beta\,\mathrm{KL}(\pi_\theta\Vert\pi_\text{ref})$，惩罚策略偏离参考太远。`kl_coef`（$\beta$）控制强度：太小 → 自由过度 → reward hacking；太大 → 几乎不动 → 学不到东西。GRPO 默认 `kl_coef=0`（R1-Zero 甚至不加），发现退化时再调高。`trainall` 用低方差的 k3 估计量。
相关：[06-rlvr-grpo.md](methods/06-rlvr-grpo.md) · [KL leash](#kl-leash) · [k3 KL estimator](#k3-kl-estimator) · [reward hacking](#reward-hacking)

### KL leash
**KL 缰绳**  
[KL penalty](#kl-penalty) 在 RLHF 语境里的形象叫法：把策略拴在 SFT 起点附近，「你可以变好，但不能变成另一个分布」。是对抗 reward hacking 的第一道防线。常配 target-KL 自适应控制器动态调 $\beta$。
相关：[05-rlhf.md](methods/05-rlhf.md) · [KL penalty](#kl-penalty) · [reference model](#reference-model) · [reward hacking](#reward-hacking)

### k3 KL estimator
**k3 KL 估计量**  
Schulman 提出的 KL 无偏低方差估计：$e^d-d-1$（$d=\log\pi_\text{ref}-\log\pi_\theta$），恒 $\ge0$，方差远小于朴素的 $-d$。GRPO 用它做逐 token KL 惩罚；当 $\pi_\theta=\pi_\text{ref}$ 时 $d=0$、该项为 0。
公式：$\mathbb{E}[e^d-d-1]\approx\mathrm{KL}(\pi_\theta\Vert\pi_\text{ref})$
相关：[06-rlvr-grpo.md](methods/06-rlvr-grpo.md) · [KL penalty](#kl-penalty) · [GRPO](#grpo)

### reward model
**奖励模型**  
RLHF 第一/二阶段产物：一个语言模型骨干去掉 LM head、换上标量头，读入 $(x,y)$ 在最后一个非 pad token 上输出一个实数分数 $r_\phi(x,y)$，用 Bradley-Terry 成对损失训练。它把「人类品味」蒸馏进一个可微打分器。但它只是真实偏好的**代理**，会被刷分、有盲区，是 PPO 优化的上限。
公式：$\mathcal{L}_\text{RM}=-\log\sigma(r_\phi(x,y_c)-r_\phi(x,y_r))$
相关：[05-rlhf.md](methods/05-rlhf.md) · [Bradley-Terry](#bradley-terry) · [scalar head](#scalar-head) · [reward hacking](#reward-hacking)

### reward hacking
**奖励黑客**  
策略钻奖励信号漏洞、刷高分却不真正变好的现象。RM 只是代理、在分布外有可利用盲区，PPO 是无情的优化器，只要某种模式能骗到高分（一律写超长、堆「当然！」、滥用 markdown）就会朝那个漏洞坍缩。对策：KL 缰绳、奖励上限、留出集人工评测、用更难被钻的可验证奖励。
相关：[05-rlhf.md](methods/05-rlhf.md) · [reward over-optimization](#reward-over-optimization) · [KL leash](#kl-leash) · [obfuscated reward hacking](#obfuscated-reward-hacking)

### reward over-optimization
**奖励过优化**  
对一个不完美的代理奖励（RM）优化过头，代理分数持续上涨而真实质量反而下降。Gao 等 (2023) 给出了它的标度律。「RM 分涨而真实质量降」是过优化的典型信号——所以要盯真实评测而非只看 RM 分。
相关：[05-rlhf.md](methods/05-rlhf.md) · [reward hacking](#reward-hacking) · [reward model](#reward-model) · [KL leash](#kl-leash)

### reward shaping
**奖励塑形**  
人为设计/组合奖励信号以提供更密或更有引导性的学习信号：多个 RM 加权、规则奖励混合、安全惩罚项、过程奖励叠加到结果奖励上。`trainall` 的 `ShapedReward` 与 `process_reward_weight` 即为此。塑形过度容易被刷（「看起来在做事」），需保证 shaped 项本身可验证。
相关：[05-rlhf.md](methods/05-rlhf.md) · [shaped reward](#shaped-reward) · [process reward model](#process-reward-model) · [credit assignment](#credit-assignment)

### shaped reward
**塑形奖励**  
`trainall.rewards` 里的 `shaped` 奖励：把基础奖励与额外塑形项（长度、格式、过程分等）组合成最终标量。属于 reward 类别（`verifier` / `reward_model` / `shaped`）之一。
相关：[reward shaping](#reward-shaping) · [reward](#reward) · [verifier](#verifier)

### credit assignment
**信用分配**  
把一个**回合级/序列级**的奖励正确地摊派到导致它的每一个 token/步骤上的问题。单轮 RL 里相对简单；agentic RL 里因误差传播（早期一步错污染整条后缀）而尤其困难。GRPO 用组内对比部分回避精确归因（不需要知道哪一步错，只要整条比同组差就压低），代价是信号更粗。过程奖励/PRM 让分配更精确。
相关：[07-agentic-rl.md](methods/07-agentic-rl.md) · [process reward model](#process-reward-model) · [advantage](#advantage) · [GRPO](#grpo)

### rollout
**采样 / 推演**  
用当前策略对一批 prompt 生成回答（轨迹）的过程。`trainall` 的 `Rollout(policy, config=RolloutConfig(group_size, temperature, top_p, max_new_tokens))` 负责；`group_sample` 对每个 prompt 采一整组共享 `group_id` 的答案，供组内归一化。policy 在测试里可以是普通 `str->str` 的 callable。
相关：[06-rlvr-grpo.md](methods/06-rlvr-grpo.md) · [Trajectory](#trajectory) · [group-relative advantage](#group-relative-advantage) · [temperature](#temperature)

### Trajectory
**轨迹**  
`trainall.types.Trajectory(prompt, response, reward, group_id, advantage, meta)`：rollout 阶段的载体，把一次（多步或单步）生成压成一条记录。agentic 里 `response` = 所有动作拼接、`reward` = 回合成败、`meta` 带 `reference`/`success`/`num_steps`。`compute_group_advantages` 就地填好每条的 `.advantage`，之后被 collate 成 policy-gradient `Batch`。
相关：[07-agentic-rl.md](methods/07-agentic-rl.md) · [rollout](#rollout) · [group-relative advantage](#group-relative-advantage) · [VerifierReward](#verifierreward)

### verifier
**验证器**  
一段能确定性判对错的代码，给回答返回 `[0,1]` 奖励（`r.reward`、`r.passed`、`r.detail`）。RLVR 的奖励就来自它——不会被刷分因为它是 ground truth。`trainall` 内置：math、code、sql、json、format、regex、citation、composite。它的覆盖面决定 RLVR 的天花板，且必须严格无副作用（宽松会被钻空子、code 要沙箱超时）。
相关：[06-rlvr-grpo.md](methods/06-rlvr-grpo.md) · [verifiable reward](#verifiable-reward) · [composite verifier](#composite-verifier) · [VerifierReward](#verifierreward)

### verifiable reward
**可验证奖励**  
能被程序自动判定对错的奖励信号（数学答案相等、代码过单测、SQL 结果集匹配、JSON 合法）。是 RLVR 成立的充要条件——有它就别训 RM，规则更稳、更不易被刷分。无它（开放式写作、主观偏好）只能退回偏好优化/RLHF。
相关：[06-rlvr-grpo.md](methods/06-rlvr-grpo.md) · [verifier](#verifier) · [RLVR](#rlvr) · [reward model](#reward-model)

### VerifierReward
`trainall.rewards.VerifierReward(verifier)`：把一个 verifier 包成 reward，`.score(list_of_Trajectory) -> [float]`，从每条 `Trajectory.meta["reference"]` 取参考、跑 verifier、写回奖励。是 RLVR 管线里「reward 层」的标准实现。
相关：[06-rlvr-grpo.md](methods/06-rlvr-grpo.md) · [verifier](#verifier) · [Trajectory](#trajectory) · [reward](#reward)

### math verifier
**数学验证器**  
判数值/符号等价：抽出 `\boxed{}` 或最后一个数，带容差与（装了 sympy 时的）符号化简，与参考答案字符串比对。`v.verify(r"\boxed{42}", reference="42") -> reward=1.0, passed=True`。陷阱：若只抽「最后一个数字」，模型可能学会在末尾堆答案而不真推理——故常用 composite 把格式也纳入。
相关：[06-rlvr-grpo.md](methods/06-rlvr-grpo.md) · [verifier](#verifier) · [composite verifier](#composite-verifier)

### citation verifier
**引文验证器**  
判模型回答里的引文是否**真出自**给定来源文本列表（防编造/幻觉）。reference 是来源文本列表。属于结构/事实类 verifier。
相关：[06-rlvr-grpo.md](methods/06-rlvr-grpo.md) · [verifier](#verifier) · [composite verifier](#composite-verifier)

### composite verifier
**组合验证器**  
把多个 verifier 按权重/逻辑组合成一个（`CompositeVerifier(mode='weighted')`）。真实 RLVR 奖励常是「正确性 × 格式」的组合——R1 的奖励就是「答案对」加「放在 `<answer>` 里」。是同时塑造正确性与可读性的关键。
相关：[06-rlvr-grpo.md](methods/06-rlvr-grpo.md) · [verifier](#verifier) · [math verifier](#math-verifier) · [DeepSeek-R1](#deepseek-r1)

### DeepSeek-R1
DeepSeek (2025) 的推理模型，把 RLVR+GRPO 走到极致。R1-Zero 从 base 直接上 GRPO + 规则奖励（答案对 + 格式对）、连 SFT 冷启动都不做，就自发涌现长链推理、自检、回溯；正式版 R1 加少量冷启动 SFT 稳定可读性。它从工程上印证：当奖励可验证时，规则就够、且更不易被刷分。
相关：[06-rlvr-grpo.md](methods/06-rlvr-grpo.md) · [RLVR](#rlvr) · [GRPO](#grpo) · [verifiable reward](#verifiable-reward)

---

# 过程监督 (Process Supervision)

### PRM
Process Reward Model（过程奖励模型）。对推理链每一步打 0/1 标签训练的验证器，能在链条中途指出第一个出错步骤。训练用逐步 BCE（不是 next-token LM 损失）；训完它本身**不生成答案**，而作为打分器用于 best-of-N 重排或 RL 的密集过程奖励。注册键 `prm`，`trainall` 实现 `ProcessRewardObjective`。
公式：见 [per-step BCE](#per-step-bce)
相关：[09-process-supervision.md](methods/09-process-supervision.md) · [process reward model](#process-reward-model) · [outcome supervision](#outcome-supervision) · [step_mask](#step_mask)

### process reward model
**过程奖励模型**  
[PRM](#prm) 的全称。相对 ORM，它的监督信号密度高一个数量级（每条链 N 个标签 vs 1 个）且可定位。聚合整条链分数时常取**步骤最小值**（木桶效应：一步错则整条不可信）。
相关：[09-process-supervision.md](methods/09-process-supervision.md) · [PRM](#prm) · [Math-Shepherd](#math-shepherd) · [credit assignment](#credit-assignment)

### outcome supervision
**结果监督**  
即 ORM 思路：只看最终答案对不对，给整条链一个标量奖励。问题是信号既含噪（第3步错却歪打正着凑回正确会被标「好」）又稀疏（前4步对、末步抄错会被整体判负），无法告诉模型「错在哪」。可视为 PRM 在 $|S|=1$ 上的退化特例。
相关：[09-process-supervision.md](methods/09-process-supervision.md) · [ORM](#orm) · [process supervision](#process-supervision) · [PRM](#prm)

### ORM
Outcome Reward Model（结果奖励模型）：只在轨迹的一个位置（末步/答案处）打分的验证器/奖励。便宜、单步任务足够，但对多步推理信号太弱。详见 [outcome supervision](#outcome-supervision)。
相关：[09-process-supervision.md](methods/09-process-supervision.md) · [outcome supervision](#outcome-supervision) · [PRM](#prm)

### per-step BCE
**逐步二元交叉熵**  
PRM 的目标函数：模型在每个步骤分隔位置输出标量 logit $z$，对 0/1 步骤标签做带 logits 的二元交叉熵，在所有步骤位置上平均（用步骤数归一化使损失与链长无关）。实现用数值稳定式 $\max(z,0)-zy+\log(1+e^{-|z|})$。
公式：$\mathcal{L}_\text{PRM}=-\frac{1}{|S|}\sum_{(b,t)\in S}[y_{b,t}\log\sigma(z_{b,t})+(1-y_{b,t})\log(1-\sigma(z_{b,t}))]$
相关：[09-process-supervision.md](methods/09-process-supervision.md) · [PRM](#prm) · [step_mask](#step_mask) · [step_labels](#step_labels)

### step_mask
PRM batch 里的布尔张量 `(B,T)`，`True` 标出每个**步骤结束/分隔符** token 位置——只有这些位置参与 BCE 损失。陷阱：标签必须精确落在你约定的步骤结束 token 上，错位一格会监督到无意义位置。
相关：[09-process-supervision.md](methods/09-process-supervision.md) · [step_labels](#step_labels) · [per-step BCE](#per-step-bce) · [PRM](#prm)

### step_labels
PRM batch 里的浮点张量 `(B,T)`，在 `step_mask=True` 处给出 0/1 标签（这一步对不对），其他位置忽略。标签来源：人工标注 (PRM800K)、蒙特卡洛自动估计 (Math-Shepherd)、或 LLM-as-judge。
相关：[09-process-supervision.md](methods/09-process-supervision.md) · [step_mask](#step_mask) · [PRM800K](#prm800k) · [Math-Shepherd](#math-shepherd)

### PRM800K
OpenAI 在 *Let's Verify Step by Step* (Lightman 2023) 中雇人逐步标注 MATH 解答得到的过程监督数据集。它支撑了「过程监督训练的验证器在 best-of-N 搜索中显著优于结果监督」这一核心发现。
相关：[09-process-supervision.md](methods/09-process-supervision.md) · [step_labels](#step_labels) · [PRM](#prm) · [best-of-N](#best-of-n)

### Math-Shepherd
Wang 等 (2023) 提出的**自动**步骤标注法：从某一中间步出发多次 rollout，若该步常能延续出正确答案就判它「正确」（蒙特卡洛估计），从而去掉人工标注。标签质量受 rollout 策略能力限制——弱策略会把「自己续不出」误判为「该步错」。
相关：[09-process-supervision.md](methods/09-process-supervision.md) · [step_labels](#step_labels) · [PRM](#prm)

### CoT monitorability
**思维链可监控性**  
思维链 (chain-of-thought) 之所以宝贵，是它是一扇能读到模型「内心」的窗户。一旦用奖励去塑造这扇窗里展示的内容，模型就有动机往窗户上「贴海报」。安全建议：对最终行为/结果施压，对 CoT 保持「只读」，把可监控性当作需要刻意保护的资产。
相关：[09-process-supervision.md](methods/09-process-supervision.md) · [obfuscated reward hacking](#obfuscated-reward-hacking) · [process supervision](#process-supervision)

### obfuscated reward hacking
**隐蔽奖励黑客**  
Baker 等 (2025) 的实证：对 CoT 直接施加优化压力，会让模型照样钻奖励漏洞、却把意图从 CoT 里**隐藏**起来，使思维链不再忠实反映真实推理。区分「这一步数学上对不对」（相对安全的客观标签）与「这一步看起来乖不乖/有没有暴露坏意图」（危险、诱发伪装的标签）至关重要。
相关：[09-process-supervision.md](methods/09-process-supervision.md) · [CoT monitorability](#cot-monitorability) · [reward hacking](#reward-hacking)

---

# 蒸馏与数据飞轮 (Distillation & Data Flywheel)

### distill
知识蒸馏 (Knowledge Distillation, KD；别名 `kd`)。训练 student 匹配更强 teacher 在每个位置的完整概率分布（软标签），软标签里藏着 teacher 的「暗知识」。`DistillObjective(temperature, alpha, kind)`：KD 项是温度缩放后的 KL 乘 $T^2$，与硬-CE 按 `alpha` 插值。student 不能超过 teacher 的信号上限。
公式：$\mathcal{L}=\alpha\,T^2\mathrm{KL}(p_T^{(1/T)}\Vert p_S^{(1/T)})+(1-\alpha)\mathcal{L}_\text{CE}$
相关：[08-distillation-and-selfplay.md](methods/08-distillation-and-selfplay.md) · [dark knowledge](#dark-knowledge) · [temperature](#temperature) · [forward KL](#forward-kl)

### dark knowledge
**暗知识**  
Hinton 等 (2015) 的说法：teacher 软标签里「非最大概率」的部分编码了**类间相似度结构**（把猫错认成狗比错认成汽车合理）。硬 one-hot 标签丢掉了这些信息，软标签把它传给 student，所以蒸馏比纯硬标签信息量大。需用温度 $T>1$ 软化分布才能让这些小概率差异显现。
相关：[08-distillation-and-selfplay.md](methods/08-distillation-and-selfplay.md) · [distill](#distill) · [temperature](#temperature)

### forward KL
**前向 KL**  
$\mathrm{KL}(p_\text{teacher}\Vert p_\text{student})$，蒸馏默认 (`kind="forward"`)。它是 **mass-covering（覆盖均值）** 的：凡 teacher 给了概率的地方 student 都被迫覆盖（否则 log 项爆炸），倾向「摊平」覆盖 teacher 所有模式。经典离线蒸馏的默认选择。
相关：[08-distillation-and-selfplay.md](methods/08-distillation-and-selfplay.md) · [mass-covering](#mass-covering) · [reverse KL](#reverse-kl) · [distill](#distill)

### reverse KL
**反向 KL**  
$\mathrm{KL}(p_\text{student}\Vert p_\text{teacher})$（`kind="reverse"`）。它是 **mode-seeking（寻找众数）** 的：student 只要把质量放在 teacher 也认可处即可，倾向锁定 teacher 的某个主模式而忽略长尾。on-policy/序列级蒸馏 (MiniLLM) 常用它，避免 forward KL 在 teacher 不会真正生成的区域分配概率导致语无伦次。
相关：[08-distillation-and-selfplay.md](methods/08-distillation-and-selfplay.md) · [mode-seeking](#mode-seeking) · [forward KL](#forward-kl) · [distill](#distill)

### mass-covering
**覆盖均值**  
forward KL 的行为特征：被迫覆盖目标分布的所有模式，导致 student 分布「摊平」。见 [forward KL](#forward-kl)。
相关：[forward KL](#forward-kl) · [mode-seeking](#mode-seeking) · [distill](#distill)

### mode-seeking
**寻找众数**  
reverse KL 的行为特征：只锁定目标分布的某个主模式、忽略长尾。见 [reverse KL](#reverse-kl)。
相关：[reverse KL](#reverse-kl) · [mass-covering](#mass-covering) · [distill](#distill)

### rejection sampling
**拒绝采样**  
对每道题用 solver 采 N 个答案，只保留 verifier 通过的，作为新 SFT 数据。利用「采 N 次至少对一次的概率 $1-(1-p)^N$ 随 N 迅速上升」——用推理期多次采样换一条正确轨迹再蒸成训练数据。`RejectionSampler(solver, verifier, n, keep)`，`keep` 取 `best`/`all`/`first`。只学正确轨迹有**幸存者偏差**。
相关：[08-distillation-and-selfplay.md](methods/08-distillation-and-selfplay.md) · [best-of-N](#best-of-n) · [RFT](#rft) · [STaR](#star)

### best-of-N
对同一 prompt 采样 N 个候选、按奖励选最优的策略。既是 rejection sampling 的别名场景，也是 PRM/RM 的经典用途（用打分器给 N 条链重排选最高分）。N 越大数据越干净但推理成本线性上涨——是算力-质量旋钮。
相关：[08-distillation-and-selfplay.md](methods/08-distillation-and-selfplay.md) · [rejection sampling](#rejection-sampling) · [PRM](#prm)

### RFT
Rejection-sampling Fine-Tuning（拒绝采样微调）：模型从自己**能被验证**的成功里 bootstrap 出训练集，再做 SFT。与 RLVR 共享同一可验证奖励信号，区别在 RFT 把信号变成离线 SFT 数据（简单稳定可复用），RLVR 变成在线策略梯度（样本效率更高但更难调）。很多团队先 RFT 冷启动再 RLVR 精修。
相关：[08-distillation-and-selfplay.md](methods/08-distillation-and-selfplay.md) · [rejection sampling](#rejection-sampling) · [STaR](#star) · [RLVR](#rlvr)

### STaR
Self-Taught Reasoner (Zelikman 2022)：模型自己生成带推理的解、保留能得到正确答案的，再用它们微调自己，迭代自举。与 RFT 同源，是合成数据飞轮的早期范式。
相关：[08-distillation-and-selfplay.md](methods/08-distillation-and-selfplay.md) · [RFT](#rft) · [synthetic data flywheel](#synthetic-data-flywheel) · [self-play](#self-play)

### proposer-solver-verifier
**出题-求解-验证**  
合成数据引擎 `SyntheticDataEngine(proposer, solver, verifier, k, dedup, keep_per_task)` 的三件套：`proposer` 出题、`solver` 采 k 个答案、`verifier` 判分，形成 `propose → solve → verify → keep` 闭环。三者都是纯 Python callable、不依赖 torch，完全可测。
相关：[08-distillation-and-selfplay.md](methods/08-distillation-and-selfplay.md) · [synthetic data flywheel](#synthetic-data-flywheel) · [verifier](#verifier) · [self-play](#self-play)

### curriculum
**课程**  
`Curriculum(difficulty, step, target_low, target_high, min_diversity)`：让飞轮持续转动的难度调节器。观察每轮通过率 `pass_rate`：高于 `target_high`（默认0.8）题太易 → 难度 `+step`；低于 `target_low`（默认0.4）题太难 → 难度 `-step`；中间则 `hold`。同时监控多样性、低于 `min_diversity` 记 `collapsed` 警告。把模型钉在「够得着但不轻松」的区间。
公式：$d\leftarrow\min(1,d+\text{step})$ 若 $\bar p>$ target_high；$\max(0,d-\text{step})$ 若 $\bar p<$ target_low
相关：[08-distillation-and-selfplay.md](methods/08-distillation-and-selfplay.md) · [zone of proximal development](#zone-of-proximal-development) · [anti-collapse](#anti-collapse) · [self-play](#self-play)

### zone of proximal development
**最近发展区**  
教育学概念，被 `Curriculum` 借用：把任务难度维持在「模型够得着但不轻松」的区间（通过率约 0.4–0.8）。太易学不到新东西，太难没有正确轨迹可学、产不出有效梯度。这正是 GRPO 需要难度适中 prompt 的原因。
相关：[08-distillation-and-selfplay.md](methods/08-distillation-and-selfplay.md) · [curriculum](#curriculum) · [group-relative advantage](#group-relative-advantage)

### distribution collapse
**分布崩溃**  
飞轮最危险的失效模式：proposer 反复出几乎一样的题、solver 反复给几乎一样的答案，数据多样性塌缩，模型在一小撮模式上过拟合、整体能力反而退化（即 model collapse / 模式坍塌）。纯靠模型自产数据无限迭代几乎必然退化。
相关：[08-distillation-and-selfplay.md](methods/08-distillation-and-selfplay.md) · [anti-collapse](#anti-collapse) · [curriculum](#curriculum) · [self-play](#self-play)

### anti-collapse
**反崩溃**  
对抗 [distribution collapse](#distribution-collapse) 的机制：`Curriculum` 每轮统计 prompt 唯一比例 `diversity = #unique / #total`，低于 `min_diversity` 就记 `collapsed` 警告，提示给 proposer 加噪声、扩主题、注入外部种子或限制连续自训轮数。
相关：[08-distillation-and-selfplay.md](methods/08-distillation-and-selfplay.md) · [distribution collapse](#distribution-collapse) · [curriculum](#curriculum)

---

# 高效微调 (Efficient Fine-Tuning)

### efficiency axis
**效率轴**  
与「学什么」(objective) 正交的一条轴「怎么省」(algorithm)：同一个目标，可选训练全部参数 (full) 还是只训一小撮 (LoRA/QLoRA)。算法只做 `prepare_model(model) -> model`，对 forward 语义和 loss 形状完全透明，是 `Trainer(model, objective, algorithm=...)` 的独立入参。
相关：[10-lora-qlora.md](methods/10-lora-qlora.md) · [LoRA](#lora) · [full finetune](#full-finetune) · [QLoRA](#qlora)

### LoRA
Low-Rank Adaptation (Hu 2021)：冻结基座权重 $W$，旁路一个低秩可训练分支 $\frac{\alpha}{r}BA$，只训练瘦长矩阵 $A,B$（可训练参数常只占 0.1%–1%）。基于「微调对权重的改动 $\Delta W$ 往往是低秩的」这一观察。`B` 初始化为 0，故 $t=0$ 时适配器与基座逐位相等，训练从已知好点平滑出发。
公式：$y=Wx+\frac{\alpha}{r}B(Ax)$，$B\in\mathbb{R}^{d\times r},A\in\mathbb{R}^{r\times k},r\ll d,k$
相关：[10-lora-qlora.md](methods/10-lora-qlora.md) · [low-rank adaptation](#low-rank-adaptation) · [rank (r)](#rank-r) · [alpha](#alpha) · [merge_lora](#merge_lora)

### low-rank adaptation
**低秩适配**  
[LoRA](#lora) 的全称所指：把全秩的 $\Delta W$ 约束/分解成秩不超过 $r$ 的两个小矩阵之积，强迫优化器只在低秩子空间里搜索。可行的根据是微调的「内在维度」很低。
相关：[10-lora-qlora.md](methods/10-lora-qlora.md) · [LoRA](#lora) · [intrinsic dimension](#intrinsic-dimension) · [rank (r)](#rank-r)

### LoRALinear
`trainall` 的 LoRA 实现单元：包住一个冻结的 `nn.Linear`（`base`），新建可训练的 `lora_A`（kaiming 初始化）、`lora_B`（初始 0）。`prepare_model` 把命中 `target_modules` 的线性层就地替换成它、冻结其余一切，于是唯一带梯度的参数就是各层的 `lora_A/lora_B`。
相关：[10-lora-qlora.md](methods/10-lora-qlora.md) · [LoRA](#lora) · [adapter](#adapter) · [target_modules](#target_modules)

### QLoRA
Quantized LoRA (Dettmers 2023)：在 LoRA 基础上把冻结基座再量化成 4-bit (NF4)，显存再省约 4×，让单卡微调 65B 成为可能、质量与 16-bit LoRA 基本持平。前向时 4-bit 权重临时反量化参与矩阵乘，梯度只流向高精度 LoRA 适配器。`trainall` 软依赖 bitsandbytes：没装则打 warning 退化为全精度基座 + LoRA。
公式：$h=\mathrm{deQ}(\mathrm{Q}_\text{NF4}(W))x+\frac{\alpha}{r}BAx$，$\nabla$ 只流向 $A,B$
相关：[10-lora-qlora.md](methods/10-lora-qlora.md) · [NF4](#nf4) · [double quantization](#double-quantization) · [quantization](#quantization) · [LoRA](#lora)

### full finetune
**全参微调**  
训练模型的全部参数，不冻结任何权重。质量上限最高，但显存上限也最高（要为每个权重存梯度 + Adam 一/二阶动量，约 4× 模型大小）。适合预训练/继续预训练、大幅改变知识、追求绝对 SOTA 且显存充足的场景。注册键 `full`。
相关：[10-lora-qlora.md](methods/10-lora-qlora.md) · [efficiency axis](#efficiency-axis) · [LoRA](#lora) · [continued pre-training](#continued-pre-training)

### rank (r)
**秩**  
LoRA 增量 $\Delta W$ 的秩上限，决定低秩分支的表达能力。$r$ 越大越接近全参灵活度，但可训练参数线性增长。常用 8/16/32，难任务或大幅风格迁移可上 64。改 $r$ 时按惯例同步改 `alpha`（保持 `alpha=2r`）以维持等效缩放。
相关：[10-lora-qlora.md](methods/10-lora-qlora.md) · [alpha](#alpha) · [LoRA](#lora) · [intrinsic dimension](#intrinsic-dimension)

### alpha
LoRA 缩放分子：实际缩放系数是 $\frac{\alpha}{r}$，把低秩分支幅度与基座解耦，这样改 `r` 时不必重调学习率。常见惯例 `alpha=2r`（`trainall` 默认 `r=8, alpha=16`，恰好 $\alpha/r=2$）。**`alpha/r` 才是真正的旋钮**，不是 `alpha` 单独。
相关：[10-lora-qlora.md](methods/10-lora-qlora.md) · [rank (r)](#rank-r) · [LoRA](#lora)

### adapter
**适配器**  
LoRA 注入的那对小矩阵 `A`/`B`（及其缩放）——一个可插拔的低秩增量。每个「技能」就是一份几十 MB 的 A/B，共享同一基座，可按需热插拔；多适配器场景别急着 merge。
相关：[10-lora-qlora.md](methods/10-lora-qlora.md) · [LoRALinear](#loralinear) · [LoRA](#lora) · [merge_lora](#merge_lora)

### target_modules
LoRA 注入位置：哪些 `nn.Linear`（按**子模块属性名**匹配，如 `q_proj`，不是完整路径）被替换成 `LoRALinear`。默认覆盖注意力 `q/k/v/o_proj` 与 FFN `gate/up/down_proj`；最小配置常只挂 `q_proj,v_proj`（原论文）。名字写错 → 一个适配器都没挂上 → 优化器没东西可训、loss 纹丝不动。
相关：[10-lora-qlora.md](methods/10-lora-qlora.md) · [LoRALinear](#loralinear) · [LoRA](#lora) · [adapter](#adapter)

### merge_lora
`merge_lora(model)`：部署时把 LoRA 增量 $\frac{\alpha}{r}BA$ 折回 `base.weight`、清零 `lora_B`（保证幂等）、用普通 `nn.Linear` 替换 `LoRALinear`。合并前后数值等价（测试到 `atol=1e-5`），推理图里再无 LoRA 痕迹、零额外开销。QLoRA 不要直接合并到 4-bit 基座（有量化误差），先反量化到 fp16/bf16 再合并。
公式：$W'\leftarrow W+\frac{\alpha}{r}BA$
相关：[10-lora-qlora.md](methods/10-lora-qlora.md) · [LoRA](#lora) · [LoRALinear](#loralinear) · [QLoRA](#qlora)

### quantization
**量化**  
把高精度权重（fp16/bf16）压成低位宽（如 4-bit）以省显存。QLoRA 只量化冻结基座（前向只读、不写，所以低精度不会被优化器放大成误差累积），梯度仍走高精度适配器。
相关：[10-lora-qlora.md](methods/10-lora-qlora.md) · [NF4](#nf4) · [double quantization](#double-quantization) · [QLoRA](#qlora)

### NF4
4-bit NormalFloat：一种信息论上对「近似正态分布的权重」最优的 4-bit 数据类型，比普通 int4/fp4 更贴合权重实际分布。QLoRA 用它量化冻结基座。
相关：[10-lora-qlora.md](methods/10-lora-qlora.md) · [quantization](#quantization) · [double quantization](#double-quantization) · [QLoRA](#qlora)

### double quantization
**双重量化**  
QLoRA 的额外省显存技巧：连量化用的缩放常数本身也再量化一遍，每个参数再省约 0.4 bit。
相关：[10-lora-qlora.md](methods/10-lora-qlora.md) · [NF4](#nf4) · [quantization](#quantization) · [QLoRA](#qlora)

### intrinsic dimension
**内在维度**  
Aghajanyan 等 (2020) 的实证：很多下游任务在几百维的低维子空间里就能微调到位——微调真正需要改变的「内在维度」很低。这是 LoRA 低秩假设的理论依据。
相关：[10-lora-qlora.md](methods/10-lora-qlora.md) · [low-rank adaptation](#low-rank-adaptation) · [rank (r)](#rank-r) · [LoRA](#lora)

---

# 架构 (Architectures)

### ArchConfig
`trainall.models.ArchConfig`：一个 dataclass 配置，覆盖 decoder-LM 的全部变体旋钮——`vocab_size, dim, n_layers, n_heads, n_kv_heads, ffn_dim, rope_theta, rope_scaling, norm_eps, max_seq_len, tie_embeddings, use_moe, n_experts, n_experts_per_tok, moe_aux_loss_coef, attn_impl, q_lora_rank, kv_lora_rank, qk_rope_head_dim` 等。`__post_init__` 会校验 `dim % n_heads == 0`、`n_heads % n_kv_heads == 0`。同一份配置可捏出 dense Llama、GQA Mistral 或 MoE DeepSeek。
相关：[11-architectures.md](methods/11-architectures.md) · [DecoderLM](#decoderlm) · [GQA](#gqa) · [MoE](#moe)

### DecoderLM
`trainall.models.DecoderLM`：标准 decoder-only transformer（GPT/Llama/DeepSeek 一脉）。`from_config(cfg)` 构建；`out = model(input_ids, attention_mask)` 返回 `out.logits (B,T,V)` 与 `out.aux_loss`（MoE 负载均衡项，dense 时为 0）。它只算 logits、不算 loss——loss 是 objective 层的事，体现「架构与 objective 解耦」。
相关：[11-architectures.md](methods/11-architectures.md) · [ArchConfig](#archconfig) · [DecoderBlock](#decoderblock) · [logits](#logits)

### DecoderBlock
`DecoderLM` 的单层：两行核心 `x = x + Attn(RMSNorm(x))` 和 `x = x + FFN(RMSNorm(x))`——pre-norm 子层 + 残差。FFN 可以是 dense SwiGLU 或 MoE；注意力按 `kv_lora_rank`/`q_lora_rank` 是否非空在 GQA `Attention` 与 MLA 之间切换。
相关：[11-architectures.md](methods/11-architectures.md) · [pre-norm](#pre-norm) · [RMSNorm](#rmsnorm) · [DecoderLM](#decoderlm)

### pre-norm
**前置归一化**  
先归一化再进子层 (`x + Sublayer(Norm(x))`)，GPT-NeoX/Llama 的写法。残差通路上没有归一化，梯度可原样穿过几十层而不衰减，比原始 transformer 的 post-norm 更易训练深层网络。
相关：[11-architectures.md](methods/11-architectures.md) · [RMSNorm](#rmsnorm) · [DecoderBlock](#decoderblock)

### RMSNorm
Root Mean Square Normalization (Zhang & Sennrich 2019)：发现 LayerNorm 的减均值 (re-centering) 那步基本没用，砍掉它、只保留「除以均方根」并乘一个可学习的逐通道增益 $g$。更快、更省显存、效果持平甚至更好，是现代 decoder 的默认 pre-norm。`trainall` 在 fp32 里算 RMS 再 cast 回保证 bf16 稳定。
公式：$\text{RMSNorm}(x)=\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}\odot g$
相关：[11-architectures.md](methods/11-architectures.md) · [pre-norm](#pre-norm) · [norm_eps](#archconfig)

### RoPE
Rotary Position Embedding (Su 2021, RoFormer)：把 query/key 向量按位置旋转一个角度（位置 $m$ 旋转 $m\theta$），于是它们的内积只依赖**相对偏移** $m-n$，把相对位置天然编码进注意力分数，外推更平滑。**只旋转 Q/K、不旋转 V**。长上下文用 NTK/YaRN 缩放 inverse frequency。
公式：$\langle R_\Theta^m q,R_\Theta^n k\rangle=\langle q,R_\Theta^{n-m}k\rangle$
相关：[11-architectures.md](methods/11-architectures.md) · [NTK scaling](#ntk-scaling) · [YaRN](#yarn) · [decoupled RoPE](#decoupled-rope)

### NTK scaling
长上下文 RoPE 缩放法 (bloc97 2023)：放大 RoPE 底数 `theta`，让高频「少拉伸」、低频「多拉伸」，比 linear 插值（把位置坐标整体除以 factor）更保细节。代码里是 `theta * factor**(dim/(dim-2))`。是推理期外推技巧，缩放后短上下文性能可能略降。
相关：[11-architectures.md](methods/11-architectures.md) · [RoPE](#rope) · [YaRN](#yarn)

### YaRN
Yet another RoPE extensioN (Peng 2023)：逐频率的插值 ramp，在高低频之间平滑过渡，并用 `mscale` 温度系数 (`attn_factor`) 补偿 logits 尺度。当前 SOTA 的长上下文方案。理想做法是缩放后再做少量长上下文微调。
相关：[11-architectures.md](methods/11-architectures.md) · [RoPE](#rope) · [NTK scaling](#ntk-scaling)

### KV cache
推理时把每个 token 的 key/value 缓存起来避免重复计算。其大小 = `层数 × 序列长度 × KV头数 × head_dim`，长上下文下会吃光显存、成为解码瓶颈。MQA/GQA/MLA 都是为缩小它而生。
相关：[11-architectures.md](methods/11-architectures.md) · [MHA](#mha) · [GQA](#gqa) · [MLA](#mla)

### MHA
Multi-Head Attention：每个 query 头配一个独立 KV 头。表达力最强，但 KV cache 最大。`trainall` 里 `n_kv_heads == n_heads` 即 MHA。
相关：[11-architectures.md](methods/11-architectures.md) · [GQA](#gqa) · [MQA](#mqa) · [KV cache](#kv-cache)

### MQA
Multi-Query Attention (Shazeer 2019)：所有 query 头共享**一个** KV 头，KV cache 缩到 `1/n_heads`，解码飞快、质量略降。`trainall` 里 `n_kv_heads == 1` 即 MQA。
相关：[11-architectures.md](methods/11-architectures.md) · [GQA](#gqa) · [MHA](#mha) · [KV cache](#kv-cache)

### GQA
Grouped-Query Attention (Ainslie 2023)：MHA 与 MQA 的折中——query 头分成几组、每组共享一个 KV 头，在质量和显存之间取最佳平衡 (Llama-2/3、Mistral)。`trainall` 用 `n_kv_heads` 统一表达三者，`repeat_kv` 按 `n_rep=n_heads//n_kv_heads` 复制对齐。省的是 KV 的参数和缓存，query 头数与 attention FLOPs 不变。
相关：[11-architectures.md](methods/11-architectures.md) · [MHA](#mha) · [MQA](#mqa) · [KV cache](#kv-cache)

### MLA
Multi-head Latent Attention (DeepSeek-V2 2024)：不共享头，而是把 KV 压缩进一个低秩 latent 向量 (`kv_lora_rank` 维) 再上投影；缓存时只存小 latent，用时才解压成完整多头 KV。比 GQA 的 KV cache 还小却保住「每个头独立」的表达力。配套 [decoupled RoPE](#decoupled-rope) 处理位置信息。
相关：[11-architectures.md](methods/11-architectures.md) · [decoupled RoPE](#decoupled-rope) · [GQA](#gqa) · [KV cache](#kv-cache)

### decoupled RoPE
**解耦 RoPE**  
MLA 的工程关键：位置信息不能被压进低秩 latent 再解压（否则旋转被破坏）。解法是把每个头切两段——一段「content/nope」(无位置、走低秩压缩)，一段小的「rope」子维度 (`qk_rope_head_dim`，单独带 RoPE 且这段 key 在所有头间共享)。
相关：[11-architectures.md](methods/11-architectures.md) · [MLA](#mla) · [RoPE](#rope)

### SwiGLU
门控前馈网络 (Shazeer 2020, GLU Variants)：用两个并行上投影，一个过 SiLU/swish 当「门」、一个当「值」，逐元素相乘后再下投影。门控让网络选择性放行信息、几乎零成本提升表达力，是现代 decoder 标配。`trainall` 三个无 bias 的 `Linear`：`gate_proj/up_proj/down_proj`。
公式：$\text{FFN}(x)=\text{down}(\sigma(\text{gate}(x))\odot\text{up}(x))$
相关：[11-architectures.md](methods/11-architectures.md) · [GeGLU](#geglu) · [MoE](#moe)

### GeGLU
SwiGLU 的姊妹：门激活换成 GELU（而非 SiLU/swish）。同样是 GLU 门控前馈，质量提升机制相同。
相关：[11-architectures.md](methods/11-architectures.md) · [SwiGLU](#swiglu)

### MoE
Mixture-of-Experts (Shazeer 2017)：放很多个 FF「专家」，但每个 token 只激活 top-k 个。总参数（容量）很大、单 token 的 FLOPs 只取决于 k——「几百 B 参数、只激活几十 B」(DeepSeek/Mixtral)。`MoEFeedForward`：gate 打分 → softmax → 取 top-k → 重归一化门控权重 → 各被选专家 (SwiGLU) 加权求和。必须配 [load-balancing aux loss](#load-balancing-aux-loss)。
相关：[11-architectures.md](methods/11-architectures.md) · [router / top-k router](#router--top-k-router) · [load-balancing aux loss](#load-balancing-aux-loss) · [SwiGLU](#swiglu)

### router / top-k router
**路由器**  
MoE 里给每个 token 打分、选 top-k 专家的 gate 线性层。它的输出 softmax 后取 top-k、重归一化作为加权权重。不加约束会偷懒（总发给同几个专家），故需负载均衡 aux loss。top-k 路由不可导，训练用 straight-through 近似，early training 易专家不均衡。
相关：[11-architectures.md](methods/11-architectures.md) · [MoE](#moe) · [load-balancing aux loss](#load-balancing-aux-loss)

### load-balancing aux loss
**负载均衡辅助损失**  
MoE 的关键正则：逼 router 把 token 均匀分散到所有专家，避免少数专家被偏爱、其余饿死、容量浪费。等于「每个专家实际接到的 token 比例 $f_e$」乘「router 给它的平均概率质量 $P_e$」、再乘 `n_experts` 和系数 `moe_aux_loss_coef`(默认0.01)。挂在 `out.aux_loss` 上，**必须**加进总 loss 一起 backward，否则 router 坍缩。
公式：$\mathcal{L}_\text{aux}=\alpha\cdot E\cdot\sum_e f_e P_e$
相关：[11-architectures.md](methods/11-architectures.md) · [MoE](#moe) · [router / top-k router](#router--top-k-router)

### tie_embeddings
**权重绑定**  
`ArchConfig` 旋钮（默认 `True`）：让 `lm_head` 与 `embed_tokens` 共享同一份权重，省参数也常更稳。想分别训练输入/输出嵌入则设 `False`。
相关：[11-architectures.md](methods/11-architectures.md) · [ArchConfig](#archconfig) · [DecoderLM](#decoderlm)

---

# 训练基础 (Training Fundamentals)

### token
文本被 tokenizer 切成的最小建模单元（子词/字符/字节）。模型在每个位置预测「下一个 token」的分布。`input_ids` 就是 token id 序列。可用数据量 = 人类写下的所有文字被分词后的 token 数。
相关：[01-pretraining.md](methods/01-pretraining.md) · [next-token objective](#next-token-objective) · [logits](#logits)

### next-token objective
**下一个 token 目标**  
预训练/CPT/SFT 共享的核心目标：给定左侧上下文 $x_{<t}$ 预测第 $t$ 个 token，最小化负对数似然（交叉熵）。SFT 在它之上加 prompt 掩码（只在 response 上算）。模型自回归地把序列联合概率分解为 $\prod_t p_\theta(x_t\mid x_{<t})$。
公式：$\mathcal{L}=-\frac1N\sum_t\log p_\theta(x_t\mid x_{<t})$
相关：[01-pretraining.md](methods/01-pretraining.md) · [causal-LM loss](#causal-lm-loss) · [cross-entropy](#cross-entropy) · [teacher forcing](#teacher-forcing)

### causal-LM loss
**因果语言模型损失**  
[next-token objective](#next-token-objective) 的实现形态：对每个位置的 logits 做 causal shift 对齐到下一位，再算交叉熵；`-100` 的位置忽略。纯预训练 `labels == input_ids`、所有 token 计损；SFT 把 prompt 段设 `-100`。**别自己手动右移**——shift 由目标函数内部完成。
公式：$\mathcal{L}=-\frac1N\sum_t\mathbb{1}[y_t\ne-100]\log p_\theta(y_t\mid x_{<t})$
相关：[01-pretraining.md](methods/01-pretraining.md) · [next-token objective](#next-token-objective) · [label masking / -100](#label-masking---100)

### self-supervised
**自监督**  
监督信号不来自人工标注，而直接来自数据自身：第 $t$ 个 token 的「正确答案」就是文本里第 $t+1$ 个 token。因此可用数据量等于人类写下的所有文字，不受标注预算限制——这正是 scaling 成立的前提。预训练/CPT 都是自监督的。
相关：[01-pretraining.md](methods/01-pretraining.md) · [pretraining](#pretraining) · [next-token objective](#next-token-objective)

### causal mask
**因果掩码**  
注意力掩码，禁止位置 $t$ 看到 $t$ 之后的任何 token，保证自回归性（看不见未来、不作弊）。它让模型一次前向就并行算出所有位置的预测（训练高效），同时每个位置又只依赖左侧上下文。`DecoderLM.forward` 内部把它和 padding mask 合并，**不要手动叠加**。
相关：[01-pretraining.md](methods/01-pretraining.md) · [teacher forcing](#teacher-forcing) · [attention_mask](#attention_mask)

### teacher forcing
**教师强制**  
训练时把**真实**前缀喂给模型（而非用它自己的预测），于是所有位置可并行计算损失。推理时没有真实前缀，只能一个 token 一个 token 自回归生成。这是训练与推理的根本差异之一。
相关：[01-pretraining.md](methods/01-pretraining.md) · [causal mask](#causal-mask) · [next-token objective](#next-token-objective)

### cross-entropy
**交叉熵**  
分类/语言建模的标准损失：对模型 softmax 后分配给正确类别的概率取 $-\log$。从信息论看它度量「用模型分布编码真实数据所需的比特/nats」——loss 越低、压缩率越高、预测越确定。
公式：$\ell_t=-\log\frac{\exp(z_{t,y_t})}{\sum_v\exp(z_{t,v})}$
相关：[01-pretraining.md](methods/01-pretraining.md) · [softmax](#softmax) · [log-prob / log-probability](#log-prob--log-probability) · [perplexity](#perplexity)

### perplexity
**困惑度**  
损失的指数 $\text{PPL}=\exp(\mathcal{L})$，可理解为「模型在每个位置平均要在多少个等概率候选里犹豫」。越接近 1 越好；随机初始化时约等于词表大小（健全性检查：一上来就远低于词表大小多半是 label 泄漏）。`compute_loss` 的 metrics 里带 `ppl` 键，始终基于未平滑 NLL。
公式：$\text{PPL}=\exp\big(\frac1N\sum_t-\log p_\theta(x_t\mid x_{<t})\big)$
相关：[01-pretraining.md](methods/01-pretraining.md) · [cross-entropy](#cross-entropy) · [label smoothing](#label-smoothing)

### logits
模型最后一层对词表每个 token 输出的**未归一化**分数 $(B,T,V)$。经 softmax 变成概率分布；经 $T$ 缩放再 softmax 用于蒸馏/采样温度；取某一维当步骤分数用于 PRM。`DecoderLM` 返回 `out.logits`。
相关：[01-pretraining.md](methods/01-pretraining.md) · [softmax](#softmax) · [DecoderLM](#decoderlm) · [temperature](#temperature)

### softmax
把一组 logits 归一化成概率分布的函数 $\frac{\exp(z_i)}{\sum_j\exp(z_j)}$。语言模型在每个位置对词表做 softmax 得到下一个 token 的概率。
公式：$\text{softmax}(z)_i=\frac{\exp(z_i)}{\sum_j\exp(z_j)}$
相关：[cross-entropy](#cross-entropy) · [logits](#logits) · [temperature](#temperature)

### sigmoid
逻辑函数 $\sigma(z)=\frac{1}{1+e^{-z}}$，把实数压到 $(0,1)$。Bradley-Terry 用它把分差变偏好概率，DPO 用它做 link，PRM/RM 用它把步骤/奖励 logit 变概率。
公式：$\sigma(z)=\frac{1}{1+e^{-z}}$
相关：[Bradley-Terry](#bradley-terry) · [DPO](#dpo) · [per-step BCE](#per-step-bce)

### log-prob / log-probability
**对数概率**  
模型分配给某 token/序列的概率取对数。序列对数概率 $\log\pi(y)=\sum_t\log\pi(y_t\mid y_{<t},x)$ 是偏好优化的基石（DPO 用它的差、SimPO 用其长度归一化版）。`trainall` 由 `sequence_logps(..., average=False/True)` 计算。
相关：[04-preference-optimization.md](methods/04-preference-optimization.md) · [sequence log-prob](#sequence-log-prob) · [cross-entropy](#cross-entropy) · [length normalization](#length-normalization)

### sequence log-prob
**序列对数概率**  
一整条回答的对数概率 $\log\pi(y)=\sum_t\log\pi(y_t\mid y_{<t},x)$（`average=False`），或其长度归一化版 $\overline{\log\pi}(y)$（`average=True`，除以 token 数）。CPO 用未归一化版，SimPO/IPO/ORPO 用归一化版以消 length bias。
相关：[04-preference-optimization.md](methods/04-preference-optimization.md) · [log-prob / log-probability](#log-prob--log-probability) · [length normalization](#length-normalization) · [CPO](#cpo)

### length normalization
**长度归一化**  
把序列对数概率除以 token 数得到平均对数概率。消除 DPO「未归一化对数概率偏爱更长回答」的 length bias。SimPO 的隐式奖励、IPO 的对数比、ORPO 的几率比都用它。
相关：[04-preference-optimization.md](methods/04-preference-optimization.md) · [SimPO](#simpo) · [sequence log-prob](#sequence-log-prob) · [DPO](#dpo)

### label masking / -100
**标签掩码**  
在 `labels` 里把不该计损的位置设为 `-100`（PyTorch 交叉熵的忽略索引）。SFT 把 prompt 段掩成 `-100`（只学 response），padding 处也补 `-100`。**`-100` 才是忽略标记**，不是删除或置 0。默认 collate 已自动处理 padding 处的 label。
相关：[03-sft.md](methods/03-sft.md) · [prompt masking](#prompt-masking) · [completion-only loss](#completion-only-loss) · [mask_prompt](#mask_prompt)

### prompt masking
**提示掩码**  
SFT 把 prompt 段在 `labels` 里掩成 `-100`，使损失只落在 response token 上——模型不浪费容量学「生成用户提问」，每份梯度精准用在「给定 prompt 该生成的答案 token」上。忘记掩 prompt 是最常见的 SFT bug。偏好优化里也要掩 prompt 防污染信号。
相关：[03-sft.md](methods/03-sft.md) · [completion-only loss](#completion-only-loss) · [label masking / -100](#label-masking---100) · [mask_prompt](#mask_prompt)

### completion-only loss
**仅回答损失**  
即只在 response/completion token 上算损失的训练方式（= prompt masking 的效果）。SFT 默认如此；`train_on_prompt=True` 可关掉它（整段都学，用于纯续写式 domain 适配）。
相关：[03-sft.md](methods/03-sft.md) · [prompt masking](#prompt-masking) · [train_on_prompt](#train_on_prompt) · [SFT](#sft)

### train_on_prompt
`SFTObjective(train_on_prompt=False)` 旋钮：默认 `False`（只在 response 上算损失）；设 `True` 则整段（含 prompt）都计损，用于纯续写式 domain 适配等少数「整段都要学」的场景。
相关：[03-sft.md](methods/03-sft.md) · [completion-only loss](#completion-only-loss) · [prompt masking](#prompt-masking) · [SFT](#sft)

### mask_prompt
`trainall.data.mask_prompt(prompt_ids, response_ids)`：构造 prompt-masked `labels` 的工具——拼接 `prompt+response` 为 `input_ids`，把前面 prompt 段的 label 设为 `-100`、只保留 response 段为真实 token id。
相关：[03-sft.md](methods/03-sft.md) · [prompt masking](#prompt-masking) · [label masking / -100](#label-masking---100) · [apply_template](#apply_template)

### chat template
**聊天模板**  
把多轮对话渲染成带角色标记的训练字符串（如 `<|im_start|>user ... <|im_end|>` 加 assistant 起始标记）。**训练用的模板必须和推理时完全一致**（chatml/llama3/plain、角色标记、是否带 `add_generation_prompt`）——模板错位是「训练好好的、上线就胡言乱语」的头号原因。
相关：[03-sft.md](methods/03-sft.md) · [apply_template](#apply_template) · [SFT](#sft)

### apply_template
`trainall.data.apply_template(messages, style)`：把消息列表渲染成带角色标记的字符串（如 `apply_template(msgs, "chatml")`），再 tokenize。是构造 SFT 训练数据的第一步。
相关：[03-sft.md](methods/03-sft.md) · [chat template](#chat-template) · [mask_prompt](#mask_prompt)

### label smoothing
**标签平滑**  
Szegedy 等 (2016)：把一点概率质量（系数 $\varepsilon$）从正确 token 分给所有 token，缓解过自信、提升泛化与校准。$\varepsilon=0$ 退化为纯 NLL。`SFTObjective(label_smoothing=0.0)`。注意上报的 `ppl` 始终基于未平滑 NLL。
公式：$\ell_t=(1-\varepsilon)(-\log p_\theta(x_t))+\varepsilon(-\frac1V\sum_v\log p_\theta(v))$
相关：[03-sft.md](methods/03-sft.md) · [SFT](#sft) · [conservative DPO (cDPO)](#conservative-dpo-cdpo) · [cross-entropy](#cross-entropy)

### temperature
**温度**  
采样/蒸馏里缩放 logits 的系数 $T$。采样时 $T>1$ 更随机、$T<1$ 更确定（贪心趋近 0）。蒸馏里 $T>1$ 软化 teacher/student 分布让暗知识显现，且 KD 损失乘 $T^2$ 还原 $1/T^2$ 的梯度缩放；推理时 $T=1$。别把蒸馏的 $T$ 和推理采样的温度混淆。
相关：[08-distillation-and-selfplay.md](methods/08-distillation-and-selfplay.md) · [temperature scaling](#temperature-scaling) · [top-p (nucleus) sampling](#top-p-nucleus-sampling) · [dark knowledge](#dark-knowledge)

### temperature scaling
**温度缩放**  
把 logits 除以 $T$ 再 softmax。蒸馏用它软化分布（$T>1$ 放大小概率间的相对差异）；采样用它调随机性。`RolloutConfig.temperature` 控制 RL 采样温度。
相关：[temperature](#temperature) · [distill](#distill) · [rollout](#rollout)

### top-p (nucleus) sampling
**核采样**  
按概率从高到低累加到阈值 $p$ 为止、只在这个「核」里采样，动态截断长尾。`RolloutConfig.top_p` 控制。与 temperature 一起调节生成的随机性/多样性。
相关：[rollout](#rollout) · [temperature](#temperature)

### attention_mask
`(B,T)` padding 掩码：1 表示真实 token、0 表示 padding。模型内部会把它和因果三角合并。预训练/CPT 通常全 1；缺省时目标函数默认全 1。它不是 labels——忽略计损用 `-100`。
相关：[01-pretraining.md](methods/01-pretraining.md) · [causal mask](#causal-mask) · [label masking / -100](#label-masking---100)

### InMemorySource
`trainall.data.InMemorySource(items, kind='auto')`：最简单的数据源，每条是已分词字典 `{"input_ids":[...], "labels":[...]}`，直接透传给 Trainer 的默认 collate（自动右 padding、补 attention_mask、padding 处 label 补 `-100`）。飞轮产出的 `Sample` 丢进它就能接 `Trainer` 做 SFT。
相关：[01-pretraining.md](methods/01-pretraining.md) · [label masking / -100](#label-masking---100) · [mask_prompt](#mask_prompt)

### catastrophic forgetting
**灾难性遗忘**  
模型在新数据上持续训练时，承载旧知识的共享权重被新梯度覆盖，导致通用能力崩塌。CPT 的默认副作用（不是偶发 bug），严重程度大致正比于新旧分布差异、学习率、训练步数。主要解药是 [replay](#replay)；上线前务必在通用 benchmark 上回归评测。
相关：[02-continued-pretraining.md](methods/02-continued-pretraining.md) · [replay](#replay) · [CPT](#cpt) · [continued pre-training](#continued-pre-training)

### replay
**回放**  
对抗灾难性遗忘的主力手段：在领域语料里掺一小撮原始预训练分布的样本（常 5%–25%），让梯度始终被旧分布「拉」一下、约束在「既降领域 NLL 又不显著抬高通用 NLL」的折中上。物理混入（按比例混进语料）是主力，batch 内加权旋钮做细调。
相关：[02-continued-pretraining.md](methods/02-continued-pretraining.md) · [replay_weight](#replay_weight) · [catastrophic forgetting](#catastrophic-forgetting) · [CPT](#cpt)

### replay_weight
`ContinuedPretrainObjective(replay_weight=0.0)` 旋钮：domain 样本权重记 `1.0`、replay 样本记 `replay_weight`($\rho$)，在同一 batch 内做加权。**`replay_weight=0.0`(默认) 不等于「丢弃 replay」**——它表示「不做 batch 内重加权」、走 fast path。要启用须非零且 batch 带 `domain` 标记（或直接给 `weights`）。
相关：[02-continued-pretraining.md](methods/02-continued-pretraining.md) · [replay](#replay) · [domain_field](#domain_field) · [ContinuedPretrainObjective](#continuedpretrainobjective)

### domain_field
`ContinuedPretrainObjective(domain_field='domain')`：指定 `batch.extra` 里标记每条样本是 domain(真) 还是 replay(假) 的布尔列表字段名。配合 `replay_weight` 触发 batch 内加权。优先级：`batch.extra["weights"]` > `batch.extra[domain_field]`+`replay_weight` > 均匀权重(fast path)。
相关：[02-continued-pretraining.md](methods/02-continued-pretraining.md) · [replay_weight](#replay_weight) · [replay](#replay) · [ContinuedPretrainObjective](#continuedpretrainobjective)

### ContinuedPretrainObjective
`trainall` 的 CPT 目标，**继承** `CausalLMObjective`。默认（无 domain/weights）走父类 fast path、与普通预训练逐位等价；启用加权时对每条样本先算逐 token 平均 NLL、再按样本权重做归一化加权平均。MoE 的 `aux_loss` 会加上但不计入 ppl。
公式：$\mathcal{L}_\text{CPT}=\frac{\sum_i w_i\ell_i}{\sum_i w_i}$
相关：[02-continued-pretraining.md](methods/02-continued-pretraining.md) · [replay_weight](#replay_weight) · [domain_field](#domain_field) · [causal-LM loss](#causal-lm-loss)

### EOS / stop token
**结束符**  
模型表示「该停下来」的结束标记。SFT 时务必让 response 末尾带上它，否则模型学不会停、推理时停不下来地往下生成。
相关：[03-sft.md](methods/03-sft.md) · [SFT](#sft) · [chat template](#chat-template)
