<!-- nav -->
<p align="center">
  <a href="09-process-supervision.md">← 过程监督</a> ·
  <a href="README.md">索引</a> ·
  <a href="../GLOSSARY.md">术语表</a> ·
  <a href="en/10-lora-qlora.md">English</a> ·
  <a href="11-architectures.md">模型架构 →</a>
</p>
<!-- /nav -->

# LoRA / QLoRA / 全参微调 (LoRA / QLoRA / Full Finetune)

> **效率轴 (efficiency axis)：在不改变训练目标的前提下，决定"训练哪些参数、用多少显存"——把冻结的预训练权重 $W$ 旁路成一个低秩可训练的 $\frac{\alpha}{r}BA$ 增量。**

![LoRA 与 QLoRA：冻结基座 + 低秩适配器](../assets/lora_qlora.png)

## 直觉：它到底在做什么

前面所有方法文档（[SFT](03-sft.md)、[偏好优化](04-preference-optimization.md)、[RLVR](06-rlvr-grpo.md)……）讲的都是 **目标函数 (objective)**：训练信号长什么样、loss 怎么算。本文讲的是一条 **正交 (orthogonal)** 的轴：**算法 (algorithm)**——同一个目标，你可以选择"训练全部参数"还是"只训练一小撮参数"。

直觉很简单。一个 7B 模型有 70 亿个权重。全参微调 (full finetune) 要为每个权重都存梯度、存优化器状态（Adam 还要存一阶/二阶动量，约 2 倍权重大小），显存动辄是模型本身的 4 倍以上。但经验和理论都表明：**微调对权重的改动 $\Delta W$ 往往是"低秩 (low-rank)"的**——它落在一个维度远小于 $W$ 的子空间里。既然如此，与其直接学一个满秩的 $\Delta W$，不如把它分解成两个瘦长矩阵 $B$（$d\times r$）和 $A$（$r\times k$）的乘积，只学这两个小矩阵，$r$ 取 8 或 16 就够。

- **LoRA**：冻结 $W$，在旁边并联一个 $\frac{\alpha}{r}BA$ 的低秩分支，只训练 $A,B$。可训练参数常常只占全模型的 0.1%–1%。
- **QLoRA**：在 LoRA 基础上，把冻结的基座再压成 4-bit（NF4），显存进一步砍掉约 4 倍——让单卡微调 65B 成为可能。
- **全参微调**：什么都不省，全部参数都训练，是质量上限，也是显存上限。

关键认知：**这三者都不改变 loss**。你可以拿任意一个目标（SFT、DPO、GRPO）配任意一个算法（full、lora、qlora）。在 trainall 里这正是 `Trainer(model, objective, algorithm=...)` 的两个独立入参。

## 原理与架构（深度讲解）

### 数据 → 目标 → 算法 的三层框架

trainall 把一次训练拆成三层，彼此正交：

1. **datasource / data**：数据从哪来、长什么样。
2. **objective**：把数据变成一个标量 loss（这是"学什么"）。
3. **algorithm**：决定"这个 loss 对哪些参数求梯度、参数以什么形式存在"（这是"怎么省"）。

`algorithm` 只做一件事：`prepare_model(model) -> model`，把模型改造成"只有该训的参数 `requires_grad=True`"的形态，并暴露 `trainable_parameters(model)` 供优化器使用。它对 forward 的语义、对 loss 的形状完全透明。

### 为什么 $\Delta W$ 可以是低秩的

预训练已经把通用能力压进了 $W$。下游适配（学一种风格、一个领域、一个偏好）本质上是在这个高维表示上做"小幅旋转/平移"，需要改变的"内在维度 (intrinsic dimension)"很低（Aghajanyan et al., 2020 的实证：很多任务在几百维子空间里就能微调到位）。LoRA（Hu et al., 2021）把这个观察工程化：直接 **约束** $\Delta W$ 的秩不超过 $r$，强迫优化器只在这个低秩子空间里搜索。

一个被冻结的线性层 $y=Wx$，加上 LoRA 后变成：

$$y = Wx + \frac{\alpha}{r}\,B(Ax)$$

其中 $W\in\mathbb{R}^{d\times k}$ 冻结，$A\in\mathbb{R}^{r\times k}$、$B\in\mathbb{R}^{d\times r}$ 可训练，$r \ll \min(d,k)$。

### 三个超参各自的作用

- **`r`（秩）**：增量的表达能力上限。$r$ 越大越接近全参的灵活度，但可训练参数线性增长。常用 8/16/32；难任务或大幅度风格迁移可上 64。
- **`alpha`（缩放分子）**：实际缩放系数是 $\frac{\alpha}{r}$。它把低秩分支的幅度与基座解耦——这样改 `r` 时不必重调学习率。常见惯例 `alpha = 2r`（trainall 默认 `r=8, alpha=16`，恰好 $\alpha/r=2$）。
- **`target_modules`（注入位置）**：哪些 `nn.Linear` 被替换成 `LoRALinear`。默认覆盖注意力的 `q/k/v/o_proj` 与 FFN 的 `gate/up/down_proj`。最小配置常只挂 `q_proj,v_proj`（原论文做法）；想榨更多质量就把 FFN 也挂上。

### 初始化的精妙之处：$B=0$ 让训练从基座精确出发

trainall 的 `LoRALinear` 把 $A$ 做 kaiming 初始化、$B$ **初始化为零**。于是 $t=0$ 时 $\Delta W = \frac{\alpha}{r}BA = 0$，适配后的层与原始基座 **逐位相等**——训练是从预训练模型这个"已知好点"出发的平滑微调，而不是从一个随机扰动开始。这就是测试 `test_lora_starts_as_noop` 验证的性质。

### 冻结与"只训适配器"是怎么落地的

`prepare_model` 的流程（见 `src/trainall/algorithms/lora.py`）：

1. 先把模型所有参数 `requires_grad_(False)`——一刀切冻住。
2. 遍历模块树，凡是属性名命中 `target_modules` 的 `nn.Linear`，就地替换成 `LoRALinear(child, r, alpha, dropout)`。`LoRALinear` 内部持有冻结的 `base`（原权重），并新建 `lora_A`、`lora_B` 两个 **可训练** 参数。
3. 结果：唯一带梯度的参数就是各层的 `lora_A/lora_B`。优化器只为它们分配状态，显存随之骤降。

### QLoRA：把冻结的基座压成 4-bit NF4

QLoRA（Dettmers et al., 2023）的洞见：**既然基座是冻结的，前向时它只读不写，那它的精度就可以很低**。具体做法（`qlora.py`）：在挂 LoRA 之前，先把目标线性层的权重用 bitsandbytes 的 `Linear4bit` 量化成 4-bit：

- **NF4 (4-bit NormalFloat)**：一种信息论上对"近似正态分布的权重"最优的 4-bit 数据类型，比普通 int4/fp4 更贴合权重的实际分布。
- **double quantization（双重量化）**：连量化用的缩放常数本身也再量化一遍，每个参数再省约 0.4 bit。
- **计算时反量化**：前向时 4-bit 权重临时反量化成 bf16 参与矩阵乘，**梯度只流向高精度的 LoRA 适配器**——基座永远不更新，所以它的低精度不会被优化器放大成误差累积。

这样一台 48GB 显存的单卡就能微调 65B 模型，且质量与 16-bit LoRA 基本持平。trainall 的实现是"软依赖"：装了 `bitsandbytes` 才真正 4-bit 量化；没装则打印一条 warning 并退化为"全精度基座 + LoRA"，逻辑上仍然可跑（方便 CPU 测试）。

### 合并 (merge)：部署时把适配器折回基座，零额外开销

训练时 LoRA 是"基座 + 旁路"两条路，多一次小矩阵乘。部署时不想要这点开销，就把增量 **折叠 (fold)** 回权重：

$$W' \leftarrow W + \frac{\alpha}{r}BA$$

`merge_lora(model)` 遍历所有 `LoRALinear`，调用 `.merge()`：把 $\frac{\alpha}{r}(BA)$ 加进 `base.weight`，再把 `lora_B` 清零（保证幂等），最后用这个普通 `nn.Linear` 替换掉 `LoRALinear`。合并后前向与合并前 **数值等价**（测试 `test_lora_merge_equivalence` 验证到 `atol=1e-5`），但模型回到标准 `nn.Linear`，推理图里再无任何 LoRA 痕迹。注意：4-bit QLoRA 的合并需要先反量化基座，否则会有量化误差，通常合并到 fp16/bf16 副本上再部署。

## 目标函数（数学）

LoRA 本身 **不是一个目标函数**——它不改 loss，只改"参数化方式"。把它写清楚需要两层。

**适配层的前向**（$h=Wx$ 被替换为）：

$$h = Wx + \Delta W x = Wx + \frac{\alpha}{r}\,B A\,x,\qquad B\in\mathbb{R}^{d\times r},\; A\in\mathbb{R}^{r\times k}$$

- $W\in\mathbb{R}^{d\times k}$：冻结的预训练权重，$\nabla_W=0$（不参与优化）。
- $A,B$：唯一可训练参数，$A$ kaiming 初始化、$B$ 初始化为 $0$，故初始 $\Delta W=0$。
- $r$：秩，约束 $\mathrm{rank}(\Delta W)\le r$。
- $\frac{\alpha}{r}$：缩放系数，把低秩分支幅度与 $r$ 解耦。

**优化目标** 则是 **任意你选的目标 $\mathcal{L}$**（SFT 的交叉熵、DPO 的偏好 logistic loss 等），只是梯度仅对 $\theta_{\text{LoRA}}=\{A,B\}$ 求：

$$\min_{\{A,B\}}\ \mathcal{L}\big(f_{W,\,A,B}(\cdot)\big),\qquad \theta_{\text{base}}=W\ \text{冻结}$$

**合并恒等式**（部署时）：

$$W' = W + \frac{\alpha}{r}BA \quad\Longrightarrow\quad W'x \equiv Wx + \frac{\alpha}{r}B A x$$

即合并前后对任意输入 $x$ 输出严格相等（浮点误差内）。

**QLoRA** 在此之上把 $W$ 换成其 4-bit 量化版 $\mathrm{Q}_{\text{NF4}}(W)$，前向时反量化 $\mathrm{deQ}$：

$$h = \mathrm{deQ}\big(\mathrm{Q}_{\text{NF4}}(W)\big)x + \frac{\alpha}{r}BAx,\qquad \nabla \text{ 只流向 } A,B$$

## 数据长什么样

这是本文的特殊之处：**算法层不消费任何特定的数据格式**。`algorithm` 只接触 `model`（通过 `prepare_model`），不接触 `Batch`。Batch 的形状完全由你配的 **objective** 决定：

- 配 [SFT](03-sft.md) → Batch 是 `input_ids / attention_mask / labels`。
- 配 [DPO](04-preference-optimization.md) → Batch 是 `chosen_* / rejected_*`。
- 配 [GRPO](06-rlvr-grpo.md) → Batch 是 `input_ids / response_mask / rewards / group_ids`。

换句话说，LoRA/QLoRA/full 对数据是 **完全透明** 的。它真正"改造"的是 **模型本身**：

```text
原始:  parent.q_proj : nn.Linear(weight 可训练)
       │
       ▼  algo.prepare_model(model)
LoRA:  parent.q_proj : LoRALinear
                       ├─ base   : nn.Linear(weight 冻结)
                       ├─ lora_A : Parameter(r, in)   ← 可训练
                       └─ lora_B : Parameter(out, r)  ← 可训练（初始为 0）
       │
       ▼  merge_lora(model)
合并:  parent.q_proj : nn.Linear(weight = W + (α/r)BA, 可训练性恢复)
```

## 在 trainall 中怎么用

下面这段代码 **已实际运行通过**：构造一个迷你模块，用 `build("lora")` 注入适配器，打印出"只有 A/B 可训练"，扰动 $B$ 让适配器生效，再 `merge_lora` 折叠并验证数值等价。

```python
import torch
import torch.nn as nn
import trainall
from trainall.algorithms import merge_lora, LoRALinear

# 一个迷你模块，带一个 LoRA 可以 target 的注意力风格投影 q_proj。
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.out    = nn.Linear(8, 8, bias=False)
    def forward(self, x):
        return self.out(self.q_proj(x))

model = Tiny()

# build('lora') 解析到 LoRA 算法；prepare_model 把命中的线性层换成
# LoRALinear，并冻结其余一切。
algo  = trainall.build("lora", category="algorithm",
                       r=8, alpha=16, target_modules=["q_proj"])
model = algo.prepare_model(model)

# 只有适配器 A/B 可训练；base + 未命中的层保持冻结。
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
total     = sum(p.numel() for p in model.parameters())
tunable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("trainable params:", trainable)
print(f"tunable / total = {tunable}/{total} = {100*tunable/total:.1f}%")
print("q_proj is LoRALinear:", isinstance(model.q_proj, LoRALinear))

# 初始 B=0 -> 适配器是 no-op (W' == W)。扰动 B 让它真正生效。
x = torch.randn(3, 8)
with torch.no_grad():
    model.q_proj.lora_B.add_(0.1 * torch.randn_like(model.q_proj.lora_B))
adapted = model(x)

# merge 把 W' = W + (alpha/r) B A 折回成普通 nn.Linear，便于部署。
merge_lora(model)
print("after merge q_proj is plain Linear:", type(model.q_proj).__name__)
merged = model(x)
print("merge equivalence (max abs diff):", (adapted - merged).abs().max().item())
```

实际输出：

```text
trainable params: ['q_proj.lora_A', 'q_proj.lora_B']
tunable / total = 128/256 = 50.0%
q_proj is LoRALinear: True
after merge q_proj is plain Linear: Linear
merge equivalence (max abs diff): 8.940696716308594e-08
```

（这里 8×8 的小模块里适配器占比偏高；真实 7B 模型上同样的 `r=8` 通常只占 0.1%–1%。）

放进真正的训练循环时，算法只是 `Trainer` 的一个入参，与目标完全解耦：

```python
import trainall
from trainall.data import InMemorySource, mask_prompt
from trainall.models import ArchConfig, DecoderLM
from trainall.training import Trainer, TrainerConfig

# 一个迷你 decoder-LM + 一份极小数据，让"换算法"可以真正跑起来。
V = 64
decoder_lm = DecoderLM.from_config(
    ArchConfig(vocab_size=V, dim=32, n_layers=2, n_heads=4, n_kv_heads=2,
               ffn_dim=64, max_seq_len=64))

def _make(prompt_ids, response_ids):
    input_ids, labels = mask_prompt(prompt_ids, response_ids)
    return {"input_ids": input_ids, "labels": labels}

data_source = InMemorySource([_make([3, 4, 5], [10, 11, 12]),
                              _make([6, 7], [20, 21, 22, 23])])

# 同一个 objective，换 algorithm 就切换效率档位
trainer = Trainer(
    model=decoder_lm,
    objective=trainall.build("sft", category="objective"),
    algorithm=trainall.build("lora", category="algorithm", r=16, alpha=32),
    data=data_source,
    config=TrainerConfig(device="cpu", batch_size=2, max_steps=5, bf16=False),
)
trainer.train()
# 把 "lora" 换成 "qlora" 或 "full" 即切换效率档位，loss 不变。
```

## 何时用 / 何时不用

| 场景 | 推荐档位 | 原因 |
|------|----------|------|
| 单卡 / 显存紧张，做风格或领域适配 | **LoRA** | 0.1%–1% 可训练参数，质量接近全参 |
| 想在消费级单卡微调 30B+ 巨模型 | **QLoRA** | 4-bit 基座再省 4× 显存 |
| 要同时维护很多"技能"，按需热插拔 | **LoRA** | 每个技能就是一份几十 MB 的 A/B，共享同一基座 |
| 预训练 / 继续预训练，要大幅改变模型知识 | **全参** | 低秩约束会限制能学到的内容量（见下） |
| 数据量巨大、追求绝对 SOTA 质量、显存充足 | **全参** | 没有秩约束，质量上限最高 |
| 需要修改非线性层（embedding、norm）的行为 | **全参 / 部分解冻** | LoRA 默认只挂 `nn.Linear` |

经验法则：**适配 (adaptation) 用 LoRA，注入大量新知识 (knowledge injection) 用全参**。LoRA 擅长"改风格、对齐偏好、学格式"，不擅长"灌入海量新事实"——后者需要的有效秩太高，低秩约束反而成为瓶颈。继续预训练（[CPT](02-continued-pretraining.md)）通常走全参。

## 常见陷阱与实践要点

- **`target_modules` 名字必须精确匹配属性名**。trainall 按 **子模块属性名**（如 `q_proj`）匹配，不是按完整路径。名字写错 → 一个适配器都没挂上 → `trainable_parameters` 为空 → 优化器没东西可训，loss 纹丝不动。挂载后务必断言 `any(p.requires_grad for p in model.parameters())`。
- **`alpha/r` 才是真正的旋钮，不是 `alpha` 单独**。调大 `r` 时若想保持等效学习率，按惯例同步调 `alpha`（保持 `alpha=2r`）。只改 `r` 不改 `alpha` 会顺带改变有效缩放，容易把"秩不够"误判成"学习率不对"。
- **LoRA 的学习率通常比全参高 1–2 个数量级**（常见 1e-4 ~ 3e-4）。因为只训练少量参数、且 $B$ 从零起步，需要更大步长才能动起来。
- **只挂 `q,v` 还是挂全部？** 原论文 `q_proj,v_proj` 已能拿到大部分收益；把 FFN（`gate/up/down_proj`）也挂上能再提质量，代价是参数翻倍。资源够就全挂。
- **QLoRA 没装 `bitsandbytes` 时会静默退化**为全精度基座 + LoRA（只打 warning），功能正常但 **没有省显存**。线上要省显存务必确认 `bitsandbytes` 已安装。
- **合并的幂等与精度**：`merge_lora` 合并后会把 `lora_B` 清零，可安全重复调用。但 **QLoRA 不要直接合并到 4-bit 基座**——会引入量化误差；正确做法是先把基座反量化到 fp16/bf16 再合并。
- **多适配器场景别急着 merge**。merge 的价值在"单一最终模型零开销部署"。如果你要在运行时切换多个技能适配器，应保留 `LoRALinear` 形态、只换 A/B，而不是合并。
- **合并后参数恢复可训练**。`merge_lora` 返回的是普通 `nn.Linear`，其 `weight.requires_grad` 不再被 LoRA 强制冻结——若要继续冻结，需自己重新设置。

## 相关

- [SFT](03-sft.md)、[偏好优化](04-preference-optimization.md)、[RLVR / GRPO](06-rlvr-grpo.md)：可与本文任意效率档位组合的 **目标函数**。
- [继续预训练 (CPT)](02-continued-pretraining.md)：通常走全参的典型场景。
- [架构](11-architectures.md)：LoRA 注入的 `q/k/v/o_proj`、`gate/up/down_proj` 究竟在 Transformer 的哪些位置。
- [蒸馏与自博弈](08-distillation-and-selfplay.md)：另一条降本路线（压小模型而非省训练显存）。
- 总览：[README](README.md) · 术语表：[LoRA](../GLOSSARY.md#lora) · [QLoRA](../GLOSSARY.md#qlora) · [full finetune](../GLOSSARY.md#full-finetune)

---

参考文献：
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, 2023.
- Aghajanyan et al., *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*, 2020.
