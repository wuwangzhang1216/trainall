# Concepts

A field guide to the method families `trainall` implements. For each: the **idea**, the **key formula**, and — most importantly — **when not to use it**. The unifying lesson is that *the feedback you can collect picks the method*.

Conventions below: `π_θ` is the policy being trained, `π_ref` a frozen reference, `r` a reward, `A` an advantage, `β` a strength/temperature coefficient.

---

## CPT / DAPT — continued & domain-adaptive pretraining

Registry keys: `cpt`, `dapt`, `pretrain`, `clm`.

**Idea.** Keep doing next-token prediction, but on a new corpus, to absorb fresh knowledge or shift the model into a domain before any alignment.

**Formula.** Standard causal language-model loss over the corpus:

```
L = − Σ_t  log π_θ(x_t | x_<t)
```

**When not to use it.** It teaches *distribution*, not *behavior* — it will not make a base model follow instructions or chat. It is data-hungry and can cause catastrophic forgetting of general ability. If you want format or instruction-following, you want SFT, not more pretraining.

---

## SFT — supervised fine-tuning

Registry key: `sft`.

**Idea.** Show the model prompt→response demonstrations and have it imitate them. The prompt tokens are masked (`labels = -100`); loss falls only on the response.

**Formula.** Causal LM loss on response tokens only:

```
L = − Σ_{t ∈ response}  log π_θ(y_t | prompt, y_<t)
```

**When not to use it.** SFT can only imitate behaviors you can *demonstrate*. It cannot learn from comparative ("A is better than B") or sparse ("this was correct") feedback, and pure imitation tends to plateau and copy demonstrator mistakes. For "better/worse" signal, move to a preference method; for "right/wrong" on checkable tasks, move to RLVR.

---

## The preference family — DPO, IPO, KTO, ORPO, SimPO, CPO

These all align a model from comparison data **without** training a separate reward model or running online RL. They differ in what signal they consume and how they regularize.

### DPO — Direct Preference Optimization

Registry key: `dpo`. (Rafailov et al., 2023.)

**Idea.** A preference pair `(chosen y_w, rejected y_l)` directly defines a loss; the optimal RLHF policy has a closed form, so you can skip the reward model and PPO entirely.

**Formula.**

```
L_DPO = − log σ( β · [ (log π_θ(y_w) − log π_ref(y_w)) − (log π_θ(y_l) − log π_ref(y_l)) ] )
```

**When not to use it.** You need genuine **pairs** and a **reference model**. With noisy or weak preferences DPO over-optimizes and can push *both* log-probs down. If you only have unpaired thumbs-up/down, use KTO; if you want to avoid the reference model, use ORPO/SimPO/CPO.

### IPO — Identity Preference Optimization

Registry key: `ipo`.

**Idea.** A regularized variant that replaces DPO's log-sigmoid with a squared-loss toward a target margin, fixing DPO's tendency to overfit to deterministic preferences.

**When not to use it.** Same pair + reference requirement as DPO. Choose it specifically when DPO is overfitting; otherwise it offers no benefit.

### KTO — Kahneman–Tversky Optimization

Registry key: `kto`.

**Idea.** Learn from **unpaired** binary labels (a single response tagged desirable/undesirable) using a prospect-theory utility, so you never need to match a chosen to a rejected.

**When not to use it.** If you *do* have well-matched pairs, DPO is usually a stronger signal. KTO's win is purely logistical — it tolerates the data you can actually collect (independent good/bad judgments).

### ORPO — Odds-Ratio Preference Optimization

Registry key: `orpo`. **Reference-free.**

**Idea.** Fold SFT and preference alignment into one loss: standard SFT plus an odds-ratio penalty that disfavors the rejected response — no separate SFT stage and no reference model.

**Formula (schematically).**

```
L_ORPO = L_SFT(y_w)  +  λ · L_OR(y_w, y_l)
```

**When not to use it.** Because it bakes in SFT, it expects to start from a base (not already heavily aligned) model and needs both the chosen response and a rejected one. If you have already done SFT, the SFT term is redundant.

### SimPO / CPO — reference-free preference optimization

Registry keys: `simpo`, `cpo`. **Reference-free.**

**Idea.** Drop the reference model from the preference objective: **SimPO** uses the length-normalized average log-probability as an implicit reward with a target margin; **CPO** adds a behavior-cloning regularizer.

**When not to use it.** Removing the reference removes a guardrail — without length normalization / margins these can exploit response length or drift. Use them when memory for a reference model is the binding constraint; otherwise DPO's reference anchor is safer.

---

## Reward modeling (RLHF, part 1)

Registry keys: `reward_model`, `rm`, `bt`.

**Idea.** Train a scalar reward model on preference pairs under the Bradley–Terry model, so it can later score arbitrary responses for online RL.

**Formula.**

```
L_RM = − log σ( r_φ(x, y_w) − r_φ(x, y_l) )
```

**When not to use it.** A separate reward model is only worth it if you will *run RL against it*. For one-shot offline alignment, DPO and friends skip this stage. Reward models are also hackable — the policy will find their blind spots — so don't deploy one as a final judge without KL control.

---

## RLHF with PPO (RLHF, part 2)

Registry key: `ppo`.

**Idea.** Use the learned reward model as the environment and optimize the policy online with PPO, keeping it close to the reference via a KL penalty.

**Formula.** Clipped surrogate with KL regularization:

```
maximize  E[ r_φ(x, y) ]  −  β · KL(π_θ ‖ π_ref)
```

**When not to use it.** PPO needs a value network, careful tuning, and is sensitive to reward-model exploitation. If your reward is a *verifier* (objectively checkable), prefer RLVR with GRPO/RLOO, which drop the learned reward model and (for GRPO) the value net.

---

## RLVR — RL from verifiable rewards (GRPO, PPO, RLOO)

Registry keys: `grpo`, `rloo`, `ppo`.

**Idea.** Replace the learned reward model with a deterministic **verifier** (math answer correct? code passes tests? SQL returns the right rows?). The reward is unhackable because it is the ground truth.

- **GRPO** (Group Relative Policy Optimization): sample a *group* of `G` responses per prompt and use the group's own statistics as the baseline — no value network.

  ```
  A_i = (r_i − mean(r_1..r_G)) / (std(r_1..r_G) + ε)
  ```

- **RLOO** (REINFORCE Leave-One-Out): baseline for each sample is the mean reward of the *other* samples in the group.
- **PPO** here is the same clipped objective, used when you prefer a learned value baseline.

**When not to use it.** **RLVR requires a verifier** — a programmatic way to check correctness. If your task has no objective right/wrong signal (open-ended writing, helpfulness, style), there is nothing to verify and you must fall back to preference methods or a reward model. GRPO also needs *several* samples per prompt (a group); a group size of one degenerates.

In `trainall`: verifiers live in `trainall.verifiers`, are wrapped into rewards by `VerifierReward`, and group advantages are computed by `compute_group_advantages` (`trainall.rl`).

---

## Agentic RL

Registry key: `grpo` (over `Episode`s, via `trainall.rl.AgenticRunner`).

**Idea.** Extend RLVR to multi-step, tool-using trajectories. The policy acts in an `Environment` (calling tools like a calculator or Python sandbox), and the whole `Episode` is scored — by outcome and optionally by per-step process rewards — then turned into trajectories for GRPO/PPO.

**Formula.** Same group-relative advantage as GRPO, but `r` is the (possibly shaped) reward of an entire multi-turn episode rather than a single response.

**When not to use it.** Only worth the complexity when the task genuinely needs *interaction* (tools, environment state, multiple turns). For single-shot answers, plain RLVR is simpler and more stable. It also inherits RLVR's hard requirement: you need a verifiable success signal at the end of the episode.

---

## Distillation / synthetic data / self-play

Registry keys: `distill`, `kd` (objective); `SyntheticDataEngine`, `RejectionSampler`, `SelfPlayLoop` (data paths in `trainall.data`).

**Idea.** Generate training signal from models instead of humans.

- **Distillation** trains a student to match a stronger teacher (e.g. KL to teacher logits, or SFT on teacher outputs).

  ```
  L_KD = KL( π_teacher(· | x)  ‖  π_student(· | x) )
  ```

- **Rejection sampling / best-of-N** (`RejectionSampler`): sample many candidates, keep only the verifier-passing ones as SFT data — the data-side of "distilling" a model's own best behavior.
- **Synthetic / self-play** (`SyntheticDataEngine`, `SelfPlayLoop`): a proposer invents tasks, a solver attempts them, a verifier filters, and a `Curriculum` raises difficulty as pass-rate climbs.

**When not to use it.** A student cannot exceed the signal it is given — distillation is capped by the teacher, and self-play is capped by the verifier's reliability and by mode collapse if the curriculum has no diversity / anti-collapse control. If a strong teacher or a trustworthy verifier is unavailable, these paths quietly amplify their own errors.

---

## PRM — process reward models / process supervision

Registry key: `prm`.

**Idea.** Instead of rewarding only the final answer, score **each reasoning step**, giving dense supervision for long chains of thought.

**Formula.** Per-step correctness loss over steps `s` of the reasoning trace:

```
L_PRM = − Σ_s [ y_s log p_φ(s)  +  (1 − y_s) log(1 − p_φ(s)) ]
```

**When not to use it.** PRM needs **step-level labels**, which are expensive to collect or to synthesize reliably. For short answers, or when only the outcome matters and is checkable, an outcome reward (RLVR) is far cheaper and avoids the labeling burden.

---

## LoRA / QLoRA — the efficiency layer (not a method)

Registry keys: `lora`, `qlora` (algorithm axis).

**Idea.** These are *not* training methods — they are how parameters update, and compose under *any* objective above.

- **LoRA** freezes the base weights `W` and learns a low-rank update `BA`, scaled by `α/r`:

  ```
  W' = W + (α / r) · B A,    B ∈ R^{d×r},  A ∈ R^{r×k},  r ≪ d
  ```

  Only `A`, `B` train; merge with `merge_lora` to fold the adapter back into `W`.

- **QLoRA** quantizes the frozen base to 4-bit (via bitsandbytes; falls back to fp + warning if unavailable) and trains LoRA adapters on top, slashing memory.

**When not to use it.** If you have the hardware, full fine-tuning (`full`) reaches the best quality. Adapters add a small quality gap and, with too-small rank, can underfit large behavioral changes. QLoRA's quantization adds further noise — for the most demanding objectives or smaller models where memory isn't the bottleneck, prefer full fine-tuning.
