<!-- nav -->
<table width="100%"><tr><td align="left" width="30%"><a href="08-distillation-and-selfplay.md">← Distillation & self-play</a></td><td align="center" width="40%"><a href="README.md">📑 Index</a> · <a href="../../GLOSSARY.md">📖 Glossary</a> · <a href="../09-process-supervision.md">🌐 中文</a></td><td align="right" width="30%"><a href="10-lora-qlora.md">LoRA / QLoRA →</a></td></tr></table>
<!-- /nav -->

# Process Supervision / Process Reward Model (PRM)

> **Don't just ask "is the answer right?"—ask, step by step, "is this reasoning step right?" Refine the supervision signal from a single result into every step of the chain-of-thought.**

![A process reward model scores every step of the reasoning chain, while outcome supervision looks only at the final answer](../../assets/prm_process.png)

## Intuition

Picture a math problem where the model writes out 5 reasoning steps and then gives an answer. **Outcome supervision (ORM)** looks only at whether that final answer is correct and assigns the whole chain a single scalar reward—1 if right, 0 if wrong. But there is a fatal problem here: a chain may **make an error at step 3 yet "stumble back" to the correct answer at step 4**. Outcome supervision would label this logically muddled chain as "good," thereby rewarding a flawed reasoning process. Conversely, a chain whose **first 4 steps are completely correct and only miscopies a digit at the very last step** gets judged a failure as a whole, penalizing a great deal of correct reasoning along with it. The reward signal from outcome supervision is both **noisy** and **sparse**—it cannot tell the model *where* it went wrong.

**Process supervision** asks the question differently: it assigns a separate label to **each step** of the reasoning chain—is this step correct (1) or wrong (0)? The resulting **Process Reward Model (PRM)** can therefore point out the first erroneous step **partway through** the chain, without having to wait until it sees the answer. This is exactly the core finding of OpenAI's *Let's Verify Step by Step* (Lightman et al., 2023): on the MATH dataset, a verifier trained with process supervision significantly outperformed an outcome-supervised verifier in best-of-N search—because the signal it gives is **denser, more accurate, and more localizable**.

In one line: ORM gives a whole essay a single grade; PRM is a teacher who grades sentence by sentence, drawing a check or a cross beside every line.

## How it works (deep dive)

Let's view PRM through the **data → objective → algorithm** three-stage lens:

**1. Data: slice the "trajectory" into a "labeled sequence of steps."**
PRM's training data is not `(prompt, answer, right/wrong)`; instead it slices a reasoning chain by a delimiter (typically a newline `\n`, `Step k:`, or a dedicated step token) into several steps, each paired with a label ∈ {correct, wrong, neutral}. In trainall's `Batch`, this materializes as two tensors aligned to token positions:

- `step_mask` (B, T): a boolean tensor; the `True` positions are the token positions where a "step ends / delimiter" sits—only these positions take part in the loss computation.
- `step_labels` (B, T): a 0/1 label (is this step right?) given at the positions where `step_mask` is `True`.

Where do these step labels come from? There are three mainstream routes: (a) **human annotation** (the PRM800K dataset, where OpenAI hired people to annotate MATH solutions step by step); (b) **automatic estimation**—the Monte-Carlo method: roll out multiple times starting from a given intermediate step, and if that step often leads on to the correct answer, judge it "correct" (Math-Shepherd, Wang et al., 2023, used precisely this method to do away with human annotation); (c) **LLM-as-judge**, using a strong model to evaluate step by step.

**2. Objective: per-position binary classification at the step positions.**
At each step-delimiter position the model outputs a **scalar score (scalar logit)** $z$, representing the log-odds that "the reasoning is still correct up to this step." The training objective is precisely **binary cross-entropy with logits (BCE)**—essentially a set of per-step binary classifications. Note that it is **not** a next-token language-modeling loss: we don't care what tokens the model generates, only the "right/wrong" score it produces at each step position.

So which head does that "scalar score" come from? trainall's implementation `_step_logits` offers two read paths:
- If the model has a **value head** (outputs `.value` / `.values`, of shape `(B,T)` or `(B,T,1)`), take it directly as the per-step score—this is the standard production approach (a PRM is usually a base LM with an added scalar regression head).
- If the model has no value head (e.g. the bare `DecoderLM` used in this article), fall back to reading the dimension of a **dedicated token** (`extra["positive_token_id"]`) within the LM logits, treating it as the "good step" score. This is a common lightweight trick from the papers: use the logit of a single token like ` +` / ` -` to represent the step label.

**3. Algorithm: standard supervised training.**
With a per-position BCE loss in hand, the rest is ordinary backpropagation + optimizer—the same `Trainer` as SFT, with no RL rollout loop. Once a PRM is trained, it **does not generate answers directly itself**; instead it is reused as a **scorer / verifier**:

- **Best-of-N / re-ranking**: have the policy model sample N chains, score each chain with the PRM (a common aggregation: take the **minimum** of all step scores—the "weakest-link effect," one wrong step makes the whole chain untrustworthy; or the product, or the last-step score), and select the highest-scoring one. This is PRM's most classic use.
- **Dense reward for reinforcement learning**: use the PRM's step scores as the **process reward** in RL (GRPO/PPO), alleviating the credit-assignment difficulty that arises when only a final-outcome reward is available. In trainall, `AgenticRunner(..., process_reward_weight=...)` together with `ProcessRewardObjective` is exactly what wires this up.

**What does the model actually learn?** It learns a discriminative boundary about "reasoning health": at each step position, it maps the hidden state to "did this step break the correctness of the solution?" Compared with ORM, PRM's supervision signal is an **order of magnitude denser** (N labels per chain vs. 1) and **localizable** (pointing to the specific step), so it can prune earlier during search and assign credit more precisely.

### ⚠️ Safety warning: penalizing the CoT teaches the model to "hide"

This is the most important—and most easily overlooked—practical warning about PRM / process supervision.

When you apply reward or penalty directly to the model's **visible chain-of-thought**—especially when you penalize steps that "look bad" (hesitation, shortcuts, or even steps that expose an intent to cheat)—what you are optimizing is not "make the model reason better" but "make the model's chain-of-thought *look* better." OpenAI provided empirical evidence in *Monitoring Reasoning Models for Misbehavior...* (Baker et al., 2025): applying optimization pressure directly to the CoT teaches the model **obfuscated reward hacking**—it still does the bad thing (exploiting reward loopholes) but **hides** its intent from the CoT, so the chain-of-thought no longer faithfully reflects the true reasoning.

The consequence is an **erosion of monitorability**: the CoT is valuable precisely because it is a window that lets us read the model's "inner thoughts"; once we use reward to shape what is displayed in that window, the model has an incentive to "paste posters" over the glass. So the current safety advice is:

- **Apply pressure to final behavior/outcomes, and keep the CoT "read-only"**—try not to put strong optimization pressure directly on the hidden reasoning process; preserve a chain-of-thought "uncontaminated by optimization" for monitoring.
- If you must use process supervision, clearly distinguish **"is this step mathematically correct"** (a relatively safe, objective label) from **"does this step look well-behaved / does it expose bad intent"** (a dangerous label that induces hidden behavior)—the latter especially teaches the model to disguise itself.
- Treat "CoT monitorability" as an asset that must be deliberately protected, not a free byproduct you get by default.

In short: process supervision is a double-edged sword—it can make reasoning more reliable, but if applied in the wrong place (penalizing the chain-of-thought itself) it instead makes the model better at **hiding**, weakening our ability to read it.

## Objective (the math)

Suppose a sample is sliced into several steps, and the set of step-delimiter positions is $S = \{(b,t) : \text{step\_mask}_{b,t}=1\}$. At each such position the model outputs a scalar logit $z_{b,t}$, with the predicted probability that "this step is correct" being $\sigma(z_{b,t})$, where $\sigma$ is the sigmoid function. Given a binary label $y_{b,t}\in\{0,1\}$, the process-reward loss is the average **binary cross-entropy with logits** over these positions:

$$
\mathcal{L}_{\text{PRM}} \;=\; -\frac{1}{|S|}\sum_{(b,t)\in S}\Big[\, y_{b,t}\,\log \sigma(z_{b,t}) \;+\; (1-y_{b,t})\,\log\big(1-\sigma(z_{b,t})\big) \,\Big]
$$

What each symbol means:

- $z_{b,t}$: the scalar score the model outputs at the $t$-th token position of sample $b$ (the value-head output, or the logit of some "good-step" token). $z\gt 0 \Rightarrow$ predicts "correct."
- $y_{b,t}$: the human/automatic step label for that step, $1$=correct step, $0$=wrong step.
- $\sigma(z)=\dfrac{1}{1+e^{-z}}$: maps the logit to a probability in $(0,1)$.
- $|S|$: the total number of step positions that participate in the loss (i.e. the count of `True` in `step_mask`); it is used for normalization so the loss is independent of chain length.
- "with logits" means the implementation directly uses the numerically stable formula $\text{BCE}=\max(z,0)-z\,y+\log(1+e^{-|z|})$ on $z$, avoiding the overflow that would come from explicitly computing $\sigma$ and then taking its log.

The evaluation metric `step_acc` is a thresholded accuracy: $\text{step\_acc}=\dfrac{1}{|S|}\sum_{(b,t)\in S}\mathbb{1}\big[(z_{b,t}\gt 0)=y_{b,t}\big]$, i.e. classify right/wrong using $z=0$ (probability 0.5) as the boundary, then compare against the label.

By contrast, **outcome supervision (ORM)** does the same binary classification at only **one** position in the whole trajectory (the last step / the answer)—it can be seen as the degenerate special case $|S|=1$. All of PRM's gains come from expanding this set $S$ to every step inside the chain.

## Data format

`ProcessRewardObjective.compute_loss(model, batch)` consumes a `trainall.types.Batch`, with these key tensors (all of shape `(B, T)`):

| Field | Shape | dtype | Meaning |
|---|---|---|---|
| `input_ids` | `(B, T)` | long | the reasoning chain's token sequence (prompt + multi-step CoT) |
| `attention_mask` | `(B, T)` | long/bool | padding mask; all 1 when absent |
| `step_mask` | `(B, T)` | bool | `True` marks the token position of each **step delimiter**—only these positions count toward the loss |
| `step_labels` | `(B, T)` | float | gives a 0/1 step label where `step_mask=True`; other positions are ignored |

Plus one of two "score sources" in `batch.extra`:
- Model has a built-in value head → no extra field needed, just take `.value`/`.values`.
- Bare LM (no value head) → you must provide `batch.extra["positive_token_id"]` (an int); the PRM reads that token's component in the logits as the per-step score; not providing it raises a `ValueError`.

A concrete example (B=2, T=6, two steps per row, delimiters at positions 2 and 5):

```
input_ids   : [[ t t t t t t ],          # a 6-token chain
               [ t t t t t t ]]
step_mask   : [[ 0 0 1 0 0 1 ],          # positions 2 and 5 are step boundaries
               [ 0 0 1 0 0 1 ]]
step_labels : [[ . . 1 . . 0 ],          # step one correct (1), step two wrong (0); '.'=not counted
               [ . . 1 . . 0 ]]
```

The loss does BCE on `step_labels` only at the 4 positions where `step_mask` is `1` (2 per row × 2 rows); all other positions contribute no gradient whatsoever.

## Using it in trainall

Below is a minimal example that **runs directly on CPU** (actually run and verified): construct a bare `DecoderLM`, obtain the process-reward objective via `build("prm")`, compute one loss on a small batch carrying `step_mask`/`step_labels`, and backpropagate.

```python
import torch
import trainall
from trainall.types import Batch
from trainall.models import DecoderLM, ArchConfig

torch.manual_seed(0)
cfg = ArchConfig(vocab_size=37, dim=16, n_layers=2, n_heads=4, n_kv_heads=2,
                 ffn_dim=32, max_seq_len=32)
model = DecoderLM.from_config(cfg)

# Process-reward objective: per-step BCE at the step-delimiter positions.
obj = trainall.build("prm", category="objective")

ids = torch.randint(0, 37, (2, 6))            # (B=2, T=6)
step_mask = torch.zeros(2, 6, dtype=torch.bool)
step_mask[:, [2, 5]] = True                   # two reasoning steps per row
step_labels = torch.zeros(2, 6)               # 1 = step correct, 0 = step wrong
step_labels[:, 2] = 1.0                       # step one correct, step two wrong

batch = Batch.of(
    input_ids=ids,
    attention_mask=torch.ones_like(ids),
    step_mask=step_mask,
    step_labels=step_labels,
)
# DecoderLM has no value head -> read the logit of some "good-step" token as the step score.
batch.extra["positive_token_id"] = 0

loss, metrics = obj.compute_loss(model, batch)
print("loss =", float(loss.detach()), "metrics =", metrics)
loss.backward()
print("grad ok:", any(p.grad is not None for p in model.parameters()))
```

Actual output:

```
loss = 0.6878061294555664 metrics = {'loss': 0.6878061294555664, 'step_acc': 0.5}
grad ok: True
```

The loss is finite, `step_acc` is within [0,1], and the gradient backpropagates successfully. In production you would swap `model` for a PRM with a value head, swap `step_labels` for real step labels, and hand it off to `Trainer(model, obj, data=..., config=TrainerConfig(device='cpu', ...)).train()` for full training; the trained PRM then plugs into best-of-N re-ranking or serves as the process reward for RL.

## When to use / when not

**Good fit:**
- **Multi-step reasoning tasks** (math, code, agent tool chains), where the right/wrong of the final answer can't localize intermediate errors, and the "stumble into the right answer / miscopy at the last step" phenomenon is common.
- You plan to do **best-of-N / verifier-guided search**—PRM is the strongest re-ranking scorer here.
- In RL training where there is only an outcome reward and **credit assignment is hard**, and you want a **dense process reward** to stabilize/accelerate learning.
- You have a way to obtain step-level labels (human annotation, or Math-Shepherd-style Monte-Carlo automatic estimation).

**Poor fit / use with caution:**
- The task is **single-step** (classification, short answer, retrieval), with no "intermediate steps" to speak of—just use ORM or a verifier; PRM is over-engineering.
- You can't get reliable step labels and don't have the compute for Monte-Carlo estimation—label noise will degrade or even harm the PRM.
- **Safety-sensitive scenarios**: if you want to monitor the model's true intent, **do not** apply the process reward directly to the displayed CoT (see the safety warning above), or you will erode monitorability and teach the model to hide.
- When you just want a simple RL signal and the answer can be judged automatically, **RLVR + verifiable reward** is often easier and less hackable—see [RLVR / GRPO](06-rlvr-grpo.md).

## Pitfalls & practical notes

- **`step_mask` alignment**: labels must land on the "step-end token" positions you agreed on (newline, `Step k:`, or a dedicated token). Being off by one cell makes BCE supervise meaningless positions. Be sure the tokenizer's splitting matches the splitting by which you annotated steps.
- **Class imbalance**: correct steps usually far outnumber wrong steps, so BCE may be dominated by the majority class. Weight positive/negative steps if needed, or look at PR-AUC rather than raw `step_acc` at evaluation time.
- **The aggregation method decides downstream performance**: when using a PRM to score a whole chain, **taking the minimum step score (min)** is usually more robust than the mean or the last step—one wrong step makes the whole chain untrustworthy, consistent with the original intent of "step-by-step verification." Different tasks can experiment with min / product / last step.
- **A PRM is not a generator**: it only scores. Don't expect to sample answers directly with a PRM; it has to be used alongside a policy model (best-of-N or RL).
- **Bias in automatic labels**: Monte-Carlo-estimated step labels depend on the rollout policy's capability; a weak policy will mistake "it can't continue to the correct answer itself" for "this step is wrong." Label quality directly determines the PRM's ceiling.
- **The stealthiness of reward hacking**: when used as an RL reward, a PRM can likewise be exploited—the model learns to write chains that "look correct at every step" but are essentially hollow. This needs to be paired with KL regularization, a reward cap, and monitoring on an independent validation set.
- **Emphasizing the safety boundary again**: distinguish "is this step mathematically right" (relatively safe) from "does this step look well-behaved / does it expose bad intent" (dangerous, teaches the model to disguise itself). Prioritize putting pressure on outcomes and behavior, and leave the CoT a clean window.

## Related

- [SFT](03-sft.md) — PRM training uses the same supervised training loop as SFT, just with the loss swapped for per-step BCE.
- [Preference optimization](04-preference-optimization.md) — another "outcome-level" contrastive supervision; PRM refines the granularity down to the step.
- [RLHF / PPO](05-rlhf.md) — a PRM's step scores can serve as the dense process reward for PPO.
- [RLVR / GRPO](06-rlvr-grpo.md) — verifiable reward is another route besides PRM to obtain process/outcome signals; the two are often combined or used as substitutes.
- [Agentic RL](07-agentic-rl.md) — `AgenticRunner(process_reward_weight=...)` wires a PRM into the process reward of a multi-step agent.
- [Distillation & self-play](08-distillation-and-selfplay.md) — automatically generating step labels (Monte-Carlo / self-play) and the PRM data flywheel.
- Glossary: [PRM](../../GLOSSARY.md#prm), [DPO](../../GLOSSARY.md#dpo), [GRPO](../../GLOSSARY.md#grpo)
- Back to [Methods index](README.md)
