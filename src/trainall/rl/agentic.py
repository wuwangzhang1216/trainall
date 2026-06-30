"""Drive a policy through an environment and collate episodes for RL.

:class:`AgenticRunner` is the agentic analogue of :class:`Rollout`: instead of
a single generation it runs a *multi-turn* loop — the policy observes, acts,
the environment responds — producing an :class:`~trainall.types.Episode`.

For policy-gradient training those episodes are flattened into
:class:`~trainall.types.Trajectory` objects (``response`` = the concatenated
actions, ``reward`` = the episode outcome, optionally blended with a per-step
*process* reward), so the same GRPO/PPO machinery that consumes single-turn
rollouts also consumes agentic ones.

Everything is pure-python: the policy is any callable ``observation -> action``
(a string-producing model wrapper or, in tests, a scripted function), so this
runs on CPU with no torch.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

from ..base import Environment, Reward
from ..types import Episode, Sample, Trajectory, Transition

Policy = Callable[[Any], Any]


class AgenticRunner:
    """Run a policy through an :class:`Environment` into scored trajectories.

    Parameters
    ----------
    env:
        The environment to act in (reset/step contract).
    policy:
        A callable ``observation -> action`` (action is a string the env
        understands).  May be a model wrapper or a scripted test policy.
    reward:
        Optional :class:`~trainall.base.Reward` re-scoring the produced
        trajectories (RLVR / reward-model).  When ``None`` the environment's
        own outcome reward is used.
    max_steps:
        Per-episode step budget.
    process_reward_weight:
        Weight on the summed per-step (process) reward added to the outcome
        reward when forming a trajectory's scalar reward.
    """

    def __init__(
        self,
        env: Environment,
        policy: Policy,
        reward: Optional[Reward] = None,
        max_steps: int = 16,
        *,
        process_reward_weight: float = 0.0,
    ) -> None:
        self.env = env
        self.policy = policy
        self.reward = reward
        self.max_steps = max_steps
        self.process_reward_weight = process_reward_weight

    # ------------------------------------------------------------------ #
    # Single episode
    # ------------------------------------------------------------------ #
    def run(self, sample: Optional[Sample] = None) -> Episode:
        """Drive the policy through one episode and return it.

        Records each turn as a :class:`Transition` and stores the running list
        of actions in ``episode.meta['actions']`` so :meth:`collect` can
        reconstruct the response text.
        """
        obs = self.env.reset(sample)
        ep = Episode()
        actions: List[str] = []
        observations: List[Any] = [obs]
        for _ in range(self.max_steps):
            action = self.policy(obs)
            actions.append(str(action))
            next_obs, reward, done, info = self.env.step(action)
            ep.add(Transition(observation=obs, action=action, reward=reward, done=done, info=info))
            observations.append(next_obs)
            obs = next_obs
            if done:
                ep.success = bool(info.get("success", reward > 0))
                break
        ep.meta["actions"] = actions
        ep.meta["observations"] = observations
        if sample is not None:
            ep.meta["prompt"] = sample.prompt or sample.text or ""
        return ep

    # ------------------------------------------------------------------ #
    # Batch collection for GRPO / PPO
    # ------------------------------------------------------------------ #
    def collect(self, samples: Sequence[Sample]) -> List[Trajectory]:
        """Run one episode per sample and map each to a :class:`Trajectory`.

        Each sample becomes its own ``group_id`` (so the same prompt run
        multiple times — pass it repeatedly — forms a GRPO group).  The
        trajectory's ``reward`` is the episode outcome plus the weighted
        process reward.
        """
        trajectories: List[Trajectory] = []
        for gid, sample in enumerate(samples):
            ep = self.run(sample)
            trajectories.append(self._to_trajectory(ep, sample, gid))
        return trajectories

    def _to_trajectory(self, ep: Episode, sample: Sample, group_id: Any) -> Trajectory:
        actions: List[str] = ep.meta.get("actions", [])
        response = "\n".join(actions)
        prompt = (sample.prompt or sample.text or "") if sample else ""

        outcome = self._outcome_reward(ep)
        process = sum(t.reward for t in ep.transitions)
        reward = outcome + self.process_reward_weight * process

        return Trajectory(
            prompt=prompt,
            response=response,
            reward=float(reward),
            group_id=group_id,
            meta={
                "success": ep.success,
                "num_steps": len(ep),
                "outcome_reward": outcome,
                "process_reward": process,
            },
        )

    def _outcome_reward(self, ep: Episode) -> float:
        """Outcome scalar: a wrapped Reward if given, else env total reward."""
        if self.reward is not None:
            actions = ep.meta.get("actions", [])
            traj = Trajectory(
                prompt=ep.meta.get("prompt", ""),
                response="\n".join(actions),
                meta={"success": ep.success},
            )
            return float(self.reward.score([traj])[0])
        # Outcome = reward of the terminal transition (env success reward).
        if ep.transitions:
            return float(ep.transitions[-1].reward)
        return 0.0


__all__ = ["AgenticRunner"]
