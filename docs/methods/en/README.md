# Methods Index

> `trainall`'s "map of methods": the modern LLM training stack is split into 11 self-contained deep-dive documents. Each one walks through **intuition → principles & architecture → objective function (the math) → what the data looks like → how to use it in trainall → when to use / when not to → common pitfalls**. Use this page first to pick the right method, then click in for the details. The companion [Glossary / Lexicon (GLOSSARY)](../../GLOSSARY.md) is a quick reference for 90+ concepts.

The core philosophy of `trainall` is to decompose every training run into **three orthogonal axes** — **data** decides *what is learned*, **objective** decides *what behavior is rewarded*, and **algorithm** decides *which parameters get updated*. Swap one string and you switch between SFT → DPO → GRPO, while the data and the optimizer stay put.

![Three orthogonal axes: data / objective / algorithm](../../assets/three_axes.png)

The most valuable thing in 2026 is **composing** these methods into a single pipeline. A typical frontier pipeline is: domain corpus → CPT/DAPT → SFT → rejection sampling / synthetic data → DPO → RLVR·GRPO → Agentic RL → distill down to a small deployable model.

![Frontier training pipeline: from domain corpus to small deployable model](../../assets/frontier_pipeline.png)

---

## The 11 method documents

| # | Doc | One-liner | When to use |
|---|------|--------|--------|
| 01 | [Pre-training](01-pretraining.md) | Next-token prediction over massive unlabeled text, compressing the statistical regularities of the world into the weights | Building a base from scratch, introducing a brand-new language/modality, with billions/trillions of tokens of general corpus in hand |
| 02 | [Continued pre-training (CPT / DAPT)](02-continued-pretraining.md) | The same next-token objective "kept training" on domain corpus, with replay to prevent forgetting | The model does not *know* your domain and you have lots of unlabeled domain text; usually runs before SFT |
| 03 | [Supervised fine-tuning (SFT)](03-sft.md) | Cross-entropy over "prompt → ideal answer," computing loss only on the answer, shaping capability into behavior | You have high-quality demonstrations and want to teach instruction-following / formatted answers; the first step of almost every alignment pipeline |
| 04 | [Preference optimization (DPO etc.)](04-preference-optimization.md) | Treat pairwise preferences as a classification/regression objective, letting the policy become its own implicit reward model | You can't write the *unique* answer but you can judge "A is better than B": tone, style, safe refusals |
| 05 | [RLHF / RLAIF](05-rlhf.md) | Learn a reward model that compresses human preferences into a scalar, then optimize with PPO, tethered by KL | The goal is subjective/relative with no reference answer, and you can tune PPO and do online exploration |
| 06 | [RLVR + GRPO](06-rlvr-grpo.md) | Use a deterministic verifier as the reward, estimate advantage by normalizing within a group, and throw away the value network | Correctness is *programmatically decidable*: math, code, SQL, structured extraction, verifiable formats |
| 07 | [Agentic RL](07-agentic-rl.md) | Expand a single-turn answer into multi-step "observe → call tool → see result," earning reward from the whole trajectory | The task needs interaction with the outside world to solve (tools/database/code repair) and the environment is reproducible |
| 08 | [Distillation & data flywheel](08-distillation-and-selfplay.md) | Use a stronger teacher or a cheap verifier to manufacture supervision signal, letting the model train on itself repeatedly | You have a strong teacher but no annotation team, or a verifiable task and want low-cost SFT data |
| 09 | [Process supervision (PRM)](09-process-supervision.md) | Don't just judge whether the answer is right — judge step by step whether "this reasoning step is correct," for a denser and more localizable signal | Multi-step reasoning; you want best-of-N reranking or dense process rewards for RL |
| 10 | [LoRA / QLoRA / full](10-lora-qlora.md) | The efficiency axis: freeze the base, bypass it with a low-rank trainable delta, without changing the training objective | Tight on VRAM and doing style/domain adaptation; QLoRA lets a single GPU fine-tune 30B+ giants |
| 11 | [Architectures](11-architectures.md) | The substrate every method runs on: RMSNorm/RoPE/GQA/MLA/SwiGLU/MoE | Model selection and building a decoder-LM from scratch; tune one ArchConfig to shape any variant |

---

## How to read this

- **First time here**: read 01 → 11 in order, to understand how the three axes "data / objective / algorithm" run through the whole stack.
- **Picking a method**: look at the "When to use" column above, or read the "when to use / when not to" section of each document first.
- **Looking up a term**: for any word you don't know, check the [Glossary / GLOSSARY](../../GLOSSARY.md) (with formulas and cross-links).
- **Want to run code**: the "how to use it in trainall" section of each document is a minimal example that runs on CPU and has actually been executed.

Languages: [中文](../README.md) · English

Back to project home: [../../../README.md](../../../README.md) · Design doc: [../../DESIGN.md](../../DESIGN.md) · Concept cheatsheet: [../../CONCEPTS.md](../../CONCEPTS.md)
