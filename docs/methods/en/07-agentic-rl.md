<!-- nav -->
<table width="100%"><tr><td align="left" width="30%"><a href="06-rlvr-grpo.md">← RLVR + GRPO</a></td><td align="center" width="40%"><a href="README.md">📑 Index</a> · <a href="../../GLOSSARY.md">📖 Glossary</a> · <a href="../07-agentic-rl.md">🌐 中文</a></td><td align="right" width="30%"><a href="08-distillation-and-selfplay.md">Distillation & self-play →</a></td></tr></table>
<!-- /nav -->

# Agentic Reinforcement Learning (Agentic RL)

> **Replace "one answer" with "multi-step interaction": inside an environment the model repeatedly observes → plans → calls tools → looks at the result, then earns a reward based on whether the whole trajectory succeeded, and uses GRPO/PPO to learn a better action policy.**

![The observe→plan→act→result→reward loop of Agentic RL](../../assets/agentic_rl.png)

## Intuition: what is it actually doing

In single-turn RLVR (see [RLVR / GRPO](06-rlvr-grpo.md)), the model writes out the whole answer in one shot and a verifier assigns a score. But many real tasks cannot be solved by just "writing a paragraph": solving a math problem that needs exact arithmetic, querying a database, running a snippet of code to see the error and then fixing it, clicking around a web page a few times to find the answer. What these tasks have in common is that **the answer is hidden inside the interaction with the outside world** — the model has to "act" first to obtain intermediate results before it knows what to do next.

Agentic RL puts a language model into an **environment**:

1. The environment gives an **observation** (the problem statement, or the result returned by the previous tool step);
2. The model, acting as the **policy**, emits an **action** — usually a tool-call string such as `calculator: 3 + 4`, or a final answer like `answer: 7`;
3. The environment executes the action: if it is a tool call, it dispatches to the tool and feeds the output back as the next observation; if it is a final answer, it judges success/failure, gives a reward, and ends the episode;
4. Repeat until success, until an answer is submitted, or until the step budget is exhausted.

The whole process is called an **episode**, which is strung together from several **transitions** (a single step: observation → action → reward). During training, we have the model run the same task many times — successful episodes get high reward, failed ones low — and then use the policy gradient to push probability mass toward the action sequences that "get through." That is agentic RL: **what it learns is not "what to say," but "how to act step by step in a world that talks back."**

## How it works (deep dive)

### The data → objective → algorithm three-stage pipeline

- **data (data/environment)**: here "data" is not a static (prompt, answer) pair, but a **reproducible environment** plus a batch of task samples `Sample`. The environment defines the action space (which tools exist), the state transitions (how an action changes the observation), and the reward signal (what counts as success). The key is **reproducibility**: the same `Sample` + the same policy must produce the same trajectory, otherwise the reward is noise and the gradient cannot converge. trainall's tools are purely functional and deterministic (`CalculatorTool` evaluates via AST, `PythonTool` runs in an isolated subprocess), precisely so that the environment is reproducible.
- **objective**: the episode is flattened into a `Trajectory` (`response` = the concatenation of all actions, `reward` = whether the episode succeeded), and then handed to exactly the same policy-gradient objective as single-turn RLVR — `GRPOObjective` / `PPOObjective` / `RLOOObjective`. In other words, **agentic only changes "where the trajectory comes from," not "how the gradient is computed."**
- **algorithm (algorithm / parameter-efficient)**: the bottom layer is still `full` / `lora` / `qlora` fine-tuning, decoupled from the objective.

### What the model actually learns

The policy gradient tells the model: "given observation $o$, if the episode resulting from the action $a$ you took ends up better than the group average, raise $\pi(a\mid o)$; otherwise lower it." Note that the reward is **episode-level** (outcome reward), but it gets **distributed (credit assignment)** to every token in the episode. So the model gradually learns:

- **when to call a tool vs. answer directly** (arithmetic it cannot do gets handed off to `calculator` first);
- **how to read the observation a tool returns** (only after seeing `7` does it submit `answer: 7`);
- **how to recover from errors** (when a tool reports `error: ...`, switch to a different way of calling it).

None of this is taught explicitly; it is all inferred backward from "which complete trajectory earned the reward."

### The core difficulty

**Sparse, long-horizon reward.** A 10-step episode may only give 1 point on the very last step ("submit answer"), while the other 9 steps are all 0. The longer the episode, the rarer the trajectories that earn any nonzero reward, the higher the variance, and the slower the learning. Two ways to mitigate this:

1. **Process reward**: also score the intermediate steps (calling a tool correctly +0.1, incorrectly −0.1), so the gradient signal is denser. In trainall this is implemented via `step_penalty` (a small per-step penalty that encourages short paths) and `AgenticRunner(process_reward_weight=...)` (which adds the weighted sum of per-step rewards onto the outcome). The final scalar reward is `reward = outcome + process_reward_weight * Σ step_reward`.
2. **Curriculum**: start with easy tasks and gradually increase difficulty, ensuring that some fraction of trajectories always succeed and provide a learning signal.

**Error propagation.** This is the essential problem that distinguishes agentic from single-turn RL: if step 2 goes wrong, the observations of steps 3, 4, 5 are all based on that erroneous state, and the entire suffix is "contaminated." In single-turn RL, one wrong token only affects locally; in agentic RL, one wrong action gets **amplified along the time axis**. This makes credit assignment harder — which step actually caused the failure? GRPO partly sidesteps precise attribution with "in-group comparison" (you don't need to know which step was wrong, only that this trajectory is overall worse than its group, so you push it down), but the cost is a coarser signal.

**Reproducible environment.** If a tool has randomness, network side effects, or timestamps, the same action returns different observations on two runs, and the reward becomes untrustworthy. You must make the environment deterministic: fix random seeds, mock external calls into fixed responses, add timeouts and sandboxing to tools (`PythonTool` uses subprocess + timeout exactly for this).

### How an episode becomes a GRPO Trajectory

`AgenticRunner.run(sample)` drives a complete episode; `._to_trajectory` flattens it:

- `response` = all actions joined with newlines (this is the "generation" to be scored and to have gradients computed on);
- `reward` = outcome (the reward of the terminating transition, or a re-scoring by an external `Reward`) + `process_reward_weight × Σ step_reward`;
- `group_id` = the several trajectories produced from the same prompt share a single group.

**Run the same task N times** to get N `Trajectory` objects sharing a `group_id`, then `compute_group_advantages` computes the in-group standardized advantage, and you have the input `GRPOObjective` needs — this step seamlessly plugs "multi-step interaction" back into the standard RLVR pipeline.

## Objective (the math)

An episode is an observation-action sequence $\tau = (o_0, a_0, o_1, a_1, \dots, o_{T-1}, a_{T-1})$, where actions are sampled by the policy $a_t \sim \pi_\theta(\cdot \mid o_t)$. The scalar reward of this trajectory is the terminal outcome plus a weighted process reward:

$$
R(\tau) \;=\; R_{\text{outcome}}(\tau) \;+\; \lambda_{\text{proc}} \sum_{t=0}^{T-1} r_t
$$

where $R_{\text{outcome}}$ is the success/failure reward (1 for success, 0 for failure, or a continuous score from the verifier), $r_t$ is the process reward of step $t$ (including $-\text{step\_penalty}$), and $\lambda_{\text{proc}}$ is the process-reward weight.

For a single task, sample a group of $G$ trajectories $\{\tau_i\}_{i=1}^G$; GRPO uses **in-group standardization** to obtain the advantage (no value network needed):

$$
A_i \;=\; \frac{R(\tau_i) - \operatorname{mean}\big(\{R(\tau_j)\}_{j=1}^G\big)}{\operatorname{std}\big(\{R(\tau_j)\}_{j=1}^G\big) + \varepsilon}
$$

The policy-gradient objective (same form as single-turn GRPO, treating all action tokens of each trajectory as a generation supervised by $A_i$):

$$
\mathcal{L}(\theta) \;=\; -\,\mathbb{E}_i\!\left[\min\!\Big(\rho_i\, A_i,\ \operatorname{clip}(\rho_i,\,1-\epsilon,\,1+\epsilon)\,A_i\Big)\right] \;+\; \beta\, \mathrm{KL}\!\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big)
$$

Notation:

- $\rho_i = \dfrac{\pi_\theta(\tau_i)}{\pi_{\theta_{\text{old}}}(\tau_i)}$ is the importance ratio of the new vs. old policy over this trajectory's actions;
- $A_i$ is the in-group advantage from above, shared across every token in the episode;
- $\epsilon$ (`clip_range`, default 0.2) clips the ratio to prevent a single update from being too large;
- $\beta$ (`kl_coef`) reins the policy in so it doesn't drift too far from the reference model;
- $\varepsilon$ (`eps`, default 1e-6) prevents division by zero for a zero-variance group — this is exactly why, when a group's trajectory rewards are all equal, the advantage is identically 0 (no contrastive information, no update).

The key intuition: the reward is episode-level $R(\tau)$, but the gradient is distributed via $\rho_i$ to every action token in the trajectory — this is agentic **credit assignment**.

## Data format

The input side is **task samples** `Sample` (not pre-generated trajectories):

```python
from trainall.types import Sample

Sample(prompt="reach 7", reference=7.0)   # prompt=task description, reference=ground truth for judging success
```

The environment uses `Sample.reference` as the success criterion. After `AgenticRunner` runs the episode, it flattens it into a `Trajectory`, and that is what gets fed to GRPO:

```python
from trainall.types import Trajectory

Trajectory(
    prompt="reach 7",
    response="calculator: 3 + 4\nanswer: 7",  # all actions concatenated
    reward=1.0,                                # outcome + weighted process
    group_id=0,                                # shared across trajectories of the same task
    advantage=None,                            # to be filled in by compute_group_advantages
    meta={"success": True, "num_steps": 2,
          "outcome_reward": 1.0, "process_reward": 0.0},
)
```

`compute_group_advantages` fills in each trajectory's `.advantage` in place. After that, `GRPOObjective` collates this batch of `Trajectory` into a policy-gradient `Batch` (`input_ids` / `attention_mask` / `response_mask` / `rewards` / `group_ids`), and the gradient acts only on the action tokens where `response_mask=1`.

## Using it in trainall

The following runs on CPU without torch: it uses `ExpressionEnv` (a reproducible environment that calls a calculator and succeeds upon reaching the target number) plus two scripted callable policies (in a real setting, replaced by a sampled LM), runs out episodes, flattens them into `Trajectory` objects sharing a group, and then computes in-group advantages — exactly the form fed to `GRPOObjective`.

```python
from trainall.rl import (
    AgenticRunner, MultiStepEnv, ToolRegistry, CalculatorTool,
    compute_group_advantages,
)
from trainall.rl.environment import ExpressionEnv
from trainall.types import Sample, Trajectory

# 1) Reproducible tool environment: reach the target number with the calculator,
#    then submit the answer; success = exact numeric match.
env = ExpressionEnv()          # action space = {calculator}; verifiable reward

# 2) A policy is just observation -> action (a string the environment understands).
#    The real policy is a sampled LM; here we script two.
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

# 3) Drive a single episode and watch the observe -> act -> result -> reward trajectory.
ep = env.rollout(good_policy(), sample=Sample(prompt="reach 7", reference=7.0), max_steps=5)
print("episode  success:", ep.success, "total_reward:", ep.total_reward, "steps:", len(ep))

# 4) Build a GRPO group: run the same prompt N times -> N Trajectory objects sharing group_id.
#    AgenticRunner.run drives the whole episode; ._to_trajectory flattens it
#    (response = concatenated actions, reward = episode success/failure).
sample = Sample(prompt="reach 7", reference=7.0)
trajs = []
for make in (good_policy, good_policy, bad_policy, bad_policy):
    runner = AgenticRunner(ExpressionEnv(), make(), max_steps=5)
    episode = runner.run(sample)
    trajs.append(runner._to_trajectory(episode, sample, group_id=0))

for t in trajs:
    print(f"  reward={t.reward:.1f} success={t.meta['success']} steps={t.meta['num_steps']}")

# 5) outcome reward -> in-group advantage, exactly the input GRPOObjective needs.
compute_group_advantages(trajs)
print("advantages:", [round(t.advantage, 3) for t in trajs])
```

Actual run output:

```
episode  success: True total_reward: 1.0 steps: 2
  reward=1.0 success=True steps=2
  reward=1.0 success=True steps=2
  reward=0.0 success=False steps=2
  reward=0.0 success=False steps=2
advantages: [1.0, 1.0, -1.0, -1.0]
```

The two successful trajectories get advantage $+1$, the two failed ones $-1$ — hand this batch of advantage-carrying `Trajectory` to `GRPOObjective`, and the policy is pushed toward the correct action sequence of "compute first, then answer." To re-score (e.g. swapping in a different reward model or verifier), pass a `Reward` to `AgenticRunner(reward=...)`; to densify the signal, tune `process_reward_weight` and the environment's `step_penalty`.

## When to use / when not

**Good fit:**

- Tasks that require **interaction with the outside world** to solve: tool calls (calculator/code/retrieval), multi-turn database or API access, web operations, code repair that needs execution feedback.
- A **reproducible, verifiable** environment with success/failure criteria (numeric match, unit tests pass, correct SQL result).
- You can afford the compute of **multi-episode sampling** (run each task N times, each run with multiple steps).

**Not a fit:**

- Tasks answerable in a single step with no intermediate results needed — single-turn [RLVR / GRPO](06-rlvr-grpo.md) is cheaper.
- Rewards that cannot be judged automatically (open-ended writing, subjective preference) — that is the domain of [preference optimization](04-preference-optimization.md) or [RLHF](05-rlhf.md).
- A non-reproducible environment (network/time/random side effects that cannot be mocked) — the reward is noise; make the environment deterministic first.
- No base model yet that reliably follows instructions and produces tool-call formats — first use [SFT](03-sft.md) to teach the tool-call format, then move to agentic RL.

## Pitfalls & practical notes

- **SFT first, then RL**: the policy must first be able to "output valid tool-call strings," otherwise almost no trajectory gets through at the start, the reward is all 0, and learning stalls. Use a small amount of trajectories for an SFT cold start (tool-call format + submission format).
- **Zero-variance group = zero gradient**: if all N trajectories in a group succeed or all fail, the in-group std ≈ 0, the advantages are all 0, and the group is wasted. Monitor the success rate; the ideal range is a mix of successes and failures within a group (about 0.2–0.8); use a [curriculum](08-distillation-and-selfplay.md) to tune difficulty into this range.
- **Control episode length**: long-horizon sparse reward is the number-one source of variance. Set a sensible `max_steps`, add `step_penalty` to encourage short paths, and densify the signal with process reward when necessary.
- **Error propagation amplifies**: an early misstep in a trajectory contaminates the entire suffix, and GRPO can only coarsely push the whole thing down. If you find the model repeatedly failing at the same step, consider adding a process reward targeted at that step, or splitting it into shorter subtasks.
- **Sandboxing and timeouts**: tools execute model-generated content (code, expressions), so you must isolate them (subprocess + timeout, like `PythonTool`) and disable dangerous calls (`CalculatorTool` rejects name/attribute/function calls), otherwise RL will actively find and exploit side effects to "farm points" (reward hacking).
- **Don't reward only the terminal step**: pure outcome reward is too sparse for long tasks; pure process reward is easy to game (the model learns to "look busy" without solving the problem). Balance the two via `process_reward_weight`, and make sure the process reward itself is verifiable.
- **Reproducibility is the baseline**: fix seeds, mock external dependencies, ensure the same (sample, policy) reproduces the same trajectory. In a non-reproducible environment, no reward curve can be trusted.

## Related

- [RLVR / GRPO](06-rlvr-grpo.md): the single-turn policy-gradient objective and in-group advantage that this article reuses.
- [RLHF](05-rlhf.md): the PPO route that uses a reward model rather than a verifiable verifier.
- [Preference optimization](04-preference-optimization.md): the rollout-free offline alignment alternative.
- [Process supervision (PRM)](09-process-supervision.md): the process reward model that scores intermediate steps, sharing the same lineage as the process reward in this article.
- [Distillation & self-play](08-distillation-and-selfplay.md): generating agentic training tasks with curriculum and self-play.
- [SFT](03-sft.md): the tool-call cold start before agentic RL.
- [LoRA / QLoRA](10-lora-qlora.md): parameter-efficient fine-tuning during the RL stage.
- Glossary: [GRPO](../../GLOSSARY.md#grpo), [RLVR](../../GLOSSARY.md#rlvr), [Trajectory](../../GLOSSARY.md#trajectory), [reward hacking](../../GLOSSARY.md#reward-hacking).
- Back to the [methods index](README.md).
