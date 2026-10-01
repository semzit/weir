from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from weir.cli.utils import compose_config, guarded_main
from weir.core.interfaces import AlgorithmPlugin, SimBackend
from weir.core.run import MANIFEST_NAME, Run
from weir.core.utils import config_to_dict, create_algorithm

_FIXED_OVERRIDE_KEYS = ("agent", "algo")


def rollout_metrics(
    sim: SimBackend,
    algorithm: AlgorithmPlugin,
    episodes: int,
    *,
    seed: int,
    max_steps: int,
) -> dict[str, float]:
    """Roll out deterministic episodes, summarizing reward, length, and forward progress.

    Forward displacement is the change in observation[0] (root x position). An
    episode counts as completed when it ends on a sim signalled termination or
    truncation rather than exhausting the per-episode step budget.
    """
    total_reward = 0.0
    total_length = 0
    total_forward = 0.0
    completed = 0
    for episode in range(episodes):
        observation = sim.reset(seed=seed + episode)
        initial_x = float(np.asarray(observation)[0])
        episode_reward = 0.0
        episode_length = 0
        finished = False
        for _ in range(max_steps):
            action = algorithm.act(observation, deterministic=True)
            step = sim.step(action)
            observation = step.observation
            episode_reward += float(step.reward)
            episode_length += 1
            if step.terminated or step.truncated:
                finished = True
                break
        total_reward += episode_reward
        total_length += episode_length
        total_forward += float(np.asarray(observation)[0]) - initial_x
        if finished:
            completed += 1
    return {
        "mean_reward": total_reward / episodes,
        "mean_episode_length": total_length / episodes,
        "total_forward_distance": total_forward,
        "mean_forward_distance_per_episode": total_forward / episodes,
        "episodes_completed": float(completed),
    }


def run_eval(
    checkpoint: Path,
    *,
    episodes: int,
    seed: int,
    max_steps: int,
    overrides: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate a checkpoint, driven by its manifest.

    ``--override`` layers on top of the manifest: the run's agent/task are used
    as the base so e.g. ``--override task.params.x=1`` tweaks the run's own
    task.
    """
    run = Run.open(checkpoint)
    manifest = run.config
    if manifest is None:
        raise ValueError(
            f"No manifest ({MANIFEST_NAME}) next to {checkpoint}: this checkpoint "
            "predates run manifests; re-train to create one"
        )
    if not overrides:
        sim = run.sim()
        run.validate(sim)
        algorithm = run.algorithm()
    else:
        _reject_fixed_overrides(overrides)
        base_agent = str(manifest["agent"]["name"])
        base_task = str(manifest["task"]["name"])
        cfg = compose_config(base_agent, base_task, overrides)
        agent_config = config_to_dict(cfg.agent, "model")
        sim_config = config_to_dict(cfg.sim)
        task_config = config_to_dict(cfg.task)
        algorithm_config = config_to_dict(cfg.algo)
        sim = Run.build_sim(sim_config)
        sim.load(agent_config, {**sim_config, "task": task_config})
        run.validate(sim)
        algorithm = create_algorithm(str(algorithm_config["plugin"]))
        algorithm.load(checkpoint)
    try:
        return rollout_metrics(sim, algorithm, episodes, seed=seed, max_steps=max_steps)
    finally:
        sim.close()


def _reject_fixed_overrides(overrides: list[str]) -> None:
    """Reject overrides that target pieces fixed by the checkpoint."""
    for override in overrides:
        key = override.split("=")[0]
        if key in _FIXED_OVERRIDE_KEYS or key.startswith(_FIXED_OVERRIDE_KEYS):
            raise ValueError(
                f"Override {key!r} targets the {key.split('.')[0]}, which is fixed by the "
                "checkpoint. Evaluation overrides may change the task or sim "
                "(e.g. --override task.params.x=1 or sim.robust=true)."
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weir-eval",
        description="Evaluate a trained checkpoint with deterministic rollouts.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to the algorithm checkpoint."
    )
    parser.add_argument("--episodes", type=int, default=5, help="Episodes to roll out.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the rollouts.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override of the run's task or sim, repeatable (agent and algo are "
        "fixed by the checkpoint).",
    )
    return parser


def _run_eval_from_args(args: argparse.Namespace) -> dict[str, Any]:
    metrics = run_eval(
        args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        overrides=args.override,
    )
    run = Run.open(args.checkpoint)
    config = run.config or {}
    agent_name = str(config.get("agent", {}).get("name", "unknown"))
    task_name = str(config.get("task", {}).get("name", "unknown"))
    return {
        "metrics": metrics,
        "checkpoint": str(args.checkpoint),
        "agent": agent_name,
        "task": task_name,
    }


def _print_metrics(result: dict[str, Any]) -> None:
    for key, value in result["metrics"].items():
        print(f"{key}: {value:.3f}")


def main(argv: list[str] | None = None) -> int:
    return guarded_main(
        build_parser(),
        _run_eval_from_args,
        "eval",
        on_success=_print_metrics,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
