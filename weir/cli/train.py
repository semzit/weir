from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from weir.cli.utils import setup_logging
from weir.core.interfaces import Shape, SimBackend
from weir.core.run import Run
from weir.core.utils import CONFIG_DIR, config_to_dict, create_algorithm, log_event
from weir.envs.gym_env import GymEnv

logger = logging.getLogger("weir")


def _shape_record(shape: Shape) -> dict[str, object]:
    return {"dims": list(shape.dims), "dtype": shape.dtype}


def _write_manifest(
    checkpoint: Path,
    *,
    agent: dict[str, Any],
    task: dict[str, Any],
    sim_config: dict[str, Any],
    algorithm_config: dict[str, Any],
    seed: int,
    total_steps: int,
    observation_shape: Shape,
    action_shape: Shape,
) -> Path:
    """Write the run manifest so downstream tools can rebuild the environment."""
    manifest = checkpoint.with_suffix(".meta.json")
    payload = {
        "version": 1,
        "agent": agent,
        "task": task,
        "sim": sim_config,
        "algo": algorithm_config,
        "train": {"seed": seed, "total_steps": total_steps},
        "observation_shape": _shape_record(observation_shape),
        "action_shape": _shape_record(action_shape),
    }
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest


class _ManifestCallback(BaseCallback):
    """Write a manifest beside each periodic checkpoint as it appears."""

    def __init__(self, run_dir: Path, manifest_kwargs: dict[str, Any]) -> None:
        super().__init__()
        self._run_dir = run_dir
        self._manifest_kwargs = manifest_kwargs
        self._seen: set[Path] = set()

    def _on_step(self) -> bool:
        for checkpoint in self._run_dir.glob("checkpoint_*_steps.zip"):
            if checkpoint in self._seen:
                continue
            self._seen.add(checkpoint)
            _write_manifest(checkpoint, **self._manifest_kwargs)
        return True


def _make_env(
    sim: SimBackend,
    sim_config: dict[str, Any],
    task: dict[str, Any],
    agent: dict[str, Any],
    n_envs: int,
) -> GymEnv | DummyVecEnv:
    """Build the training environment, vectorized when n_envs > 1.

    DummyVecEnv runs the envs in one process (sequential stepping). The sim
    is cheap (~0.05 ms/step) relative to the policy (~1 ms), so parallelism
    is not the win here — experience diversity (N different starting states
    per update) and batched policy inference are. SubprocVecEnv was measured
    slower: its per-step IPC overhead dwarfs the sim time. Each factory
    builds a fresh sim from the config, so only plain config dicts are
    shared (the caller's ``sim`` is never moved).
    """
    if n_envs <= 1:
        return GymEnv(sim)

    def factory() -> GymEnv:
        extra = Run.build_sim(sim_config)
        extra.load(agent, {**sim_config, "task": task})
        return GymEnv(extra)

    return DummyVecEnv([factory for _ in range(n_envs)])


def run(cfg: DictConfig) -> dict[str, Any]:
    """Train the algorithm on the simulator for the configured number of steps."""
    agent = config_to_dict(cfg.agent, "model")
    sim_config = config_to_dict(cfg.sim)
    task = config_to_dict(cfg.task)
    algorithm_config = config_to_dict(cfg.algo, "checkpoint")
    algorithm_config["seed"] = int(cfg.train.seed)

    sim = Run.build_sim(sim_config)
    algorithm = create_algorithm(str(algorithm_config["plugin"]))

    sim.load(agent, {**sim_config, "task": task})
    observation_shape = sim.observation_shape()
    action_shape = sim.action_shape()
    algorithm.configure(observation_shape, action_shape, algorithm_config)

    total_steps = int(cfg.train.total_steps)

    log_event(
        logger,
        "train.start",
        sim=sim_config["plugin"],
        algo=algorithm_config["plugin"],
        agent=agent.get("name"),
        observation_shape=observation_shape,
        action_shape=action_shape,
        total_steps=total_steps,
    )

    n_envs = int(algorithm_config.get("n_envs", 1))
    env = _make_env(sim, sim_config, task, agent, n_envs)
    manifest_kwargs = {
        "agent": agent,
        "task": task,
        "sim_config": sim_config,
        "algorithm_config": algorithm_config,
        "seed": int(cfg.train.seed),
        "total_steps": total_steps,
        "observation_shape": observation_shape,
        "action_shape": action_shape,
    }
    metrics = algorithm.learn(
        env, total_steps, callback=_ManifestCallback(Path.cwd(), manifest_kwargs)
    )

    checkpoint = Path.cwd() / "checkpoint.zip"
    algorithm.save(checkpoint)
    _write_manifest(checkpoint, **manifest_kwargs)
    env.close()
    if n_envs > 1:
        sim.close()  # the main-process sim is unused by the vectorized envs

    log_event(
        logger,
        "train.complete",
        steps=total_steps,
        checkpoint=str(checkpoint),
        **metrics,
    )
    return {"steps": total_steps, **metrics}


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="train")
def main(cfg: DictConfig) -> None:
    setup_logging()
    run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
