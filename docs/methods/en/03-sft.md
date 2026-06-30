<!-- nav -->
<table width="100%"><tr><td align="left" width="30%"><a href="02-continued-pretraining.md">← Continued pre-training</a></td><td align="center" width="40%"><a href="README.md">📑 Index</a> · <a href="../../GLOSSARY.md">📖 Glossary</a> · <a href="../03-sft.md">🌐 中文</a></td><td align="right" width="30%"><a href="04-preference-optimization.md">Preference optimization →</a></td></tr></table>
<!-- /nav -->

# Supervised Fine-Tuning (SFT)

> **Cross-entropy on (prompt → answer) pairs, but the loss is computed only on the answer tokens — SFT is not about pumping in new knowledge, it is about shaping the pretrained model's knowledge into the behaviour of "follow instructions, answer in the right format."**

![SFT: the prompt segment is masked to -100, and gradients flow back only on the response tokens](../../assets/sft.png)

## Intuition

A base model fresh out of pretraining is essentially a "continuation engine": give it a piece of text and it continues with the most likely next text. It has read a vast amount of corpus and **knows** a great deal, but it does not know that the input you hand it is an "instruction that needs to be answered" — it is just as likely to continue your question into another question, or into some off-topic web text.

What SFT does is very plain: take a batch of high-quality "instruction → ideal answer" pairs `(prompt, response)`, and have the model **generate the `response`** after the `prompt`, fitting it with the standard language-model cross-entropy loss. The one crucial detail is this — **the loss is computed only on the `response` tokens; the `prompt` tokens are masked out (set to `-100`)**. The model does not need to learn to "generate the question"; it only needs to learn to "generate a good answer given the question."

After a few thousand to a few hundred thousand such examples, the model turns from a "continuation engine" into a "chat/instruction assistant": it learns the format of a conversation (turns, role markers), learns "when to stop" (emitting the end-of-sequence token), and learns the tone and structure of an answer. This is the first step of the InstructGPT pipeline (Ouyang et al. 2022), and the starting point of almost every chat model.

## How it works (deep dive)

Look at SFT through the three-part lens of **data → objective → algorithm**:

- **Data**: one example is a `(prompt, response)`. The `prompt` is usually already wrapped by a **chat template** — adding role markers such as `<|im_start|>user ... <|im_end|>`, and appending the assistant's start marker at the end (see `apply_template` below). The whole `prompt + response` is then sliced into token ids by the tokenizer, and every position outside the response segment is filled with `-100` in `labels`.
- **Objective**: it is the causal-LM negative log-likelihood (NLL), but **with a mask**. The next section gives the formula. In `trainall` this is `SFTObjective`.
- **Algorithm**: full-parameter fine-tuning `full`, or the parameter-efficient `lora` / `qlora` (see [LoRA / QLoRA](10-lora-qlora.md)). The algorithm decides "which weights get updated," and is orthogonal to the objective.

**What is it actually adjusting?** A point worth stressing repeatedly: **SFT mostly does "shaping behaviour," not "injecting knowledge."** The reason is that the amount of SFT data (typically on the order of 10³–10⁵) is negligible relative to pretraining (10¹²+ tokens), and a few epochs of gradients are simply not enough to press a large amount of new facts into the model. What it really changes is the **"entry point" of the model's conditional distribution**: after seeing the instruction format, the model redistributes probability mass onto the "answer directly" trajectory, while suppressing the probability of trajectories like "continue into another question," "go off-topic," or "keep talking endlessly." In other words, SFT is **activating and aligning capabilities the pretrained model already learned**, so that they are reliably triggered in the role of an "assistant."

This also explains a repeatedly validated empirical law: **for SFT, data quality matters far more than quantity**. LIMA (Zhou et al. 2023, "Less Is More for Alignment") trained strong instruction-following ability with only 1000 carefully chosen examples, precisely because what SFT learns is "a mapping of style and format," and style only needs a small number of high-quality demonstrations to be stably established. Conversely, mixing in low-quality, self-contradictory, or format-messy examples will directly teach the model bad habits — the model faithfully imitates whatever you show it, errors included.

**Why compute the loss only on the response?** If you do not mask the prompt, the model spends part of its capacity learning "how to generate the user's question," which is both wasteful and dilutes the loss with the verbose prompt, weakening the fitting signal for the response. After masking, every bit of gradient is spent precisely on "given this prompt, what is the next answer token to generate." (`SFTObjective` provides `train_on_prompt=True` to turn off this behaviour, for the few scenarios where "the whole sequence should be learned," such as pure continuation-style domain adaptation.)

**How it connects to downstream methods**: SFT yields "a decent policy," but it only learns to **imitate a single demonstration** and cannot express relative preferences like "how much better A is than B." To align further with human preferences, you need to do [preference optimization](04-preference-optimization.md) (DPO etc.) or [RLHF](05-rlhf.md) on top of the SFT model. SFT is almost always the **initialization and reference point (reference model)** for these methods.

## Objective (the math)

Let the concatenated token sequence of one example be $x = (x_1, \dots, x_T)$, where the index set of the response segment is $\mathcal{R}$ (the prompt segment is masked and not in $\mathcal{R}$). The model $p_\theta$ is autoregressive, and the logits at token $t$ predict token $t{+}1$. The SFT loss is the average negative log-likelihood over the response tokens:

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{|\mathcal{R}|} \sum_{t \in \mathcal{R}} \log p_\theta\!\left(x_t \mid x_{<t}\right)
$$

- $x_t$: the $t$-th gold token (the supervision target).
- $x_{<t}$: all tokens before it (including the prompt), used as conditioning.
- $\mathcal{R}$: the index set of response tokens; $|\mathcal{R}|$ is its size, used to normalize by the number of effective tokens.
- $p_\theta(x_t \mid x_{<t})$: the model's predicted probability for the gold token at position $t$, obtained by gathering after `log_softmax(logits)`.

Adding **label smoothing** (coefficient $\varepsilon$, Szegedy et al. 2016), the loss for each token becomes:

$$
\ell_t = (1-\varepsilon)\,\big(\!-\log p_\theta(x_t \mid x_{<t})\big) \;+\; \varepsilon \cdot \Big(\!-\tfrac{1}{V}\textstyle\sum_{v=1}^{V} \log p_\theta(v \mid x_{<t})\Big)
$$

where $V$ is the vocabulary size. $\varepsilon=0$ reduces to pure NLL. The smoothing term hands a little probability mass to all tokens, mitigating over-confidence and improving generalization and calibration. The `ppl` reported in `trainall` is always based on the unsmoothed NLL: $\text{ppl} = \exp\big(\tfrac{1}{|\mathcal{R}|}\sum_{t\in\mathcal{R}} -\log p_\theta(x_t\mid x_{<t})\big)$.

## Data format

`SFTObjective.compute_loss(model, batch)` consumes a `trainall.types.Batch` containing the following tensors (shape `(B, T)`):

- `input_ids`: the concatenated `prompt + response` token ids.
- `attention_mask`: 1 means a real token, 0 means padding.
- `labels`: the supervision target. **The prompt segment and the padding segment are filled with `-100` (ignored); only the response segment holds real token ids.**

The most direct way to construct this is to write each example as a pre-tokenized dict `{'input_ids': [...], 'labels': [...]}` and hand it to `InMemorySource` — the `Trainer`'s default collate will automatically right-pad, fill in the `attention_mask`, and set the labels at padding positions to `-100`. `mask_prompt(prompt_ids, response_ids)` is the tool for constructing this kind of `labels`:

```python
from trainall.data import mask_prompt

input_ids, labels = mask_prompt([1, 2, 3], [4, 5])
assert input_ids == [1, 2, 3, 4, 5]
assert labels    == [-100, -100, -100, 4, 5]   # the first 3 prompt tokens are masked
```

And the `prompt` text itself is generally first rendered into a string with role markers by `apply_template(messages, "chatml")`, then tokenized.

## Using it in trainall

The example below runs end-to-end (CPU, tiny model): it renders one conversation with a chat template, uses `mask_prompt` to construct prompt-masked `labels`, runs a few steps with `InMemorySource` + `SFTObjective` + a CPU `Trainer`, and separately calls `compute_loss` on a single `Batch` to verify the loss is a finite scalar.

```python
import torch
from trainall.data import InMemorySource, mask_prompt, apply_template
from trainall.models import ArchConfig, DecoderLM
from trainall.training import Trainer, TrainerConfig
import trainall

# 1) Render one conversation into a training string with the chat template (demo only; tokens are placeholder integers)
msgs = [{"role": "user", "content": "2+2 等于几?"},
        {"role": "assistant", "content": "等于 4。"}]
print(apply_template(msgs, "chatml"))

# 2) Build a prompt-only-masked example: prompt segment labels=-100, loss only on the response
def make_sample(prompt_ids, response_ids):
    input_ids, labels = mask_prompt(prompt_ids, response_ids)
    return {"input_ids": input_ids, "labels": labels}

V = 64
samples = [make_sample([3, 4, 5], [10, 11, 12]),
           make_sample([6, 7], [20, 21, 22, 23])]
print("labels[0] =", samples[0]["labels"])  # the first 3 should be -100
data = InMemorySource(samples)

# 3) Tiny model + SFTObjective + CPU Trainer
cfg = ArchConfig(vocab_size=V, dim=32, n_layers=2, n_heads=4, n_kv_heads=2,
                 ffn_dim=64, max_seq_len=64)
model = DecoderLM.from_config(cfg)
sft = trainall.build("sft", category="objective")  # SFTObjective(label_smoothing=0.0)

trainer = Trainer(
    model, sft, data=data,
    config=TrainerConfig(device="cpu", batch_size=2, max_steps=3,
                         lr=1e-3, log_every=1, bf16=False),
)
trainer.train()

# 4) You can also call compute_loss directly on a Batch to confirm it is a finite scalar
from trainall.types import Batch
ids = torch.randint(0, V, (2, 8))
labels = ids.clone(); labels[:, :4] = -100  # mask the prompt segment
batch = Batch.of(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
loss, metrics = sft.compute_loss(model, batch)
print("loss =", float(loss.detach()), "ppl =", metrics["ppl"])
assert torch.isfinite(loss)
```

Run output (excerpt): over three training steps the loss slowly decreases from ~4.11, `compute_loss` prints a finite `loss` and `ppl`, and the first 3 values of `labels[0]` are indeed `-100`, confirming the prompt is masked correctly.

To turn on label smoothing, use `trainall.build("sft", category="objective", label_smoothing=0.1)`; to learn the whole sequence (without masking the prompt), pass `train_on_prompt=True`.

## When to use / when not

**Use SFT when:**
- You have a base model (or just finished [continued pre-training (CPT)](02-continued-pretraining.md)) and want to turn it into an assistant that follows instructions and answers in a specified format.
- You can obtain (or can create) a batch of high-quality "instruction → ideal answer" demonstration data.
- As the **initialization policy and reference model** for DPO / RLHF / RLVR — the first step of almost every alignment pipeline.

**Do not (only) use SFT when:**
- You want to **pump a large amount of new facts / new domain knowledge** into the model — that is the job of [pretraining](01-pretraining.md) / [CPT](02-continued-pretraining.md); SFT's data volume cannot shape knowledge.
- You only have **relative preference** signals like "A is better than B," without a single "ideal answer" — use [preference optimization](04-preference-optimization.md) (the DPO family).
- The task has a **verifier that can automatically judge right from wrong** (math, code, SQL) and you want to directly optimize "accuracy" — use [RLVR / GRPO](06-rlvr-grpo.md), which can explore solutions that surpass the demonstrations, whereas SFT can only imitate up to the ceiling of the demonstrations.

## Pitfalls & practical notes

- **Forgetting to mask the prompt**: the most common bug. If the prompt segment of `labels` is not set to `-100`, the model will learn to generate user questions, wasting capacity and diluting the signal. Always confirm that the prefix of `labels` is `-100` (this article's example deliberately prints to verify).
- **The template must match inference**: the `apply_template` style used in training (chatml / llama3 / plain), the role markers, and whether `add_generation_prompt` is included must be perfectly aligned with how you deploy for inference. Template mismatch is the number-one cause of "trained fine, talks gibberish in production."
- **Data quality > quantity**: see LIMA. Better 1k clean, consistent, uniformly-styled examples than 100k examples carrying errors and contradictions — the model faithfully imitates everything you give it, bad examples included.
- **Don't over-train**: SFT usually runs only 1–3 epochs. Too many epochs overfit to the surface wording of the demonstrations, lose diversity, and may even start forgetting pretrained knowledge (catastrophic forgetting). Watch the validation set, not the training loss.
- **Learning rate and algorithm**: the lr for full SFT is far lower than for pretraining (typically 1e-5 ~ 2e-5). If you are tight on memory, switch to [LoRA / QLoRA](10-lora-qlora.md) — the results are usually close to full fine-tuning at far lower cost.
- **EOS / stop token**: make sure the end-of-sequence marker is appended at the end of the response, otherwise the model will not learn "when to stop" and will keep generating endlessly at inference time.
- **Padding does not participate in the loss**: the default collate already sets the labels at padding positions to `-100`, so no manual handling is needed; but if you customize the collate, do not forget this step.

## Related

- [Pretraining](01-pretraining.md) — the starting point of SFT: you need continuation ability before you can talk about shaping it.
- [Continued Pretraining / DAPT](02-continued-pretraining.md) — injects domain knowledge, often placed before SFT.
- [Preference Optimization / DPO](04-preference-optimization.md) — the next step after SFT, learning relative preferences.
- [RLHF (PPO)](05-rlhf.md) — reinforcement alignment with a reward model; the SFT model is its initialization and reference.
- [RLVR / GRPO](06-rlvr-grpo.md) — surpasses the demonstration ceiling when a verifiable reward is available.
- [LoRA / QLoRA](10-lora-qlora.md) — the parameter-efficient algorithm choice for SFT.
- Glossary: [SFT](../../GLOSSARY.md#sft) · [DPO](../../GLOSSARY.md#dpo) · [LoRA](../../GLOSSARY.md#lora)
- Back to [Methods index](README.md)
