"""Deterministic episode rollouts shared by evaluation and rendering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from weir.core.interfaces import AlgorithmPlugin, SimBackend

ResetHook = Callable[[], None]
StepHook = Callable[[int], None]


@dataclass(frozen=True, slots=True)
class EpisodeStats:
    """Outcome of one deterministic episode."""

    reward: float
    length: int
    progress: float
    completed: bool


def run_episode(
    sim: SimBackend,
    algorithm: AlgorithmPlugin,
    *,
    seed: int,
    max_steps: int,
    on_reset: ResetHook | None = None,
    on_step: StepHook | None = None,
) -> EpisodeStats:
    """Roll out the policy for one episode, optionally observing reset/step events.

    ``on_reset`` runs after the reset so a renderer can capture the initial frame;
    ``on_step`` runs after each step with the 1-based step index. An episode is
    ``completed`` when the sim signals termination or truncation rather than the
    step budget running out.
    """
    if max_steps < 0:
        raise ValueError(f"max_steps must be non-negative, got {max_steps}")
    observation = sim.reset(seed=seed)
    if on_reset is not None:
        on_reset()
    reward = 0.0
    progress = 0.0
    for step_index in range(1, max_steps + 1):
        action = algorithm.act(observation, deterministic=True)
        step = sim.step(action)
        progress += sim.progress(observation, step.observation)
        observation = step.observation
        reward += float(step.reward)
        if on_step is not None:
            on_step(step_index)
        if step.terminated or step.truncated:
            return EpisodeStats(reward, step_index, progress, True)
    return EpisodeStats(reward, max_steps, progress, False)


def rollout_metrics(
    sim: SimBackend,
    algorithm: AlgorithmPlugin,
    episodes: int,
    *,
    seed: int,
    max_steps: int,
) -> dict[str, float]:
    """Roll out deterministic episodes, summarizing reward, length, and task progress.

    Progress is whatever the run's task defines via ``SimBackend.progress`` (for
    ``walk_forward`` it is forward displacement; tasks without a progress notion
    report ``0.0``). An episode counts as completed when it ends on a sim
    signalled termination or truncation rather than exhausting the per-episode
    step budget.
    """
    stats = [
        run_episode(sim, algorithm, seed=seed + episode, max_steps=max_steps)
        for episode in range(episodes)
    ]
    return {
        "mean_reward": sum(item.reward for item in stats) / episodes,
        "mean_episode_length": sum(item.length for item in stats) / episodes,
        "mean_progress": sum(item.progress for item in stats) / episodes,
        "episodes_completed": float(sum(item.completed for item in stats)),
    }
