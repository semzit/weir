from __future__ import annotations

from pathlib import Path

from weir.core.configs import RandomizationConfig
from weir.core.rollout import rollout_metrics
from weir.core.run import Run, build_test_env
from weir.core.utils import compose_config, config_to_dict, create_algorithm

_FIXED_OVERRIDE_KEYS = ("agent", "algo")


def run_eval(
    checkpoint: Path,
    *,
    episodes: int,
    seed: int,
    max_steps: int,
    overrides: list[str] | None = None,
    perturbation: RandomizationConfig | None = None,
) -> dict[str, float]:
    """Open a checkpoint and evaluate it."""
    return evaluate_run(
        Run.open(checkpoint),
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
        overrides=overrides,
        perturbation=perturbation,
    )


def evaluate_run(
    run: Run,
    *,
    episodes: int,
    seed: int,
    max_steps: int,
    overrides: list[str] | None = None,
    perturbation: RandomizationConfig | None = None,
) -> dict[str, float]:
    """Evaluate an open run, driven by its manifest.

    ``--override`` layers on top of the manifest: the run's agent/task are used
    as the base so e.g. ``--override task.params.x=1`` tweaks the run's own
    task. ``perturbation`` enables disturbance pushes on top of the manifest's
    hardening settings.
    """
    manifest = run.require_config()
    if not overrides:
        agent_config = dict(manifest["agent"])
        sim_config = dict(manifest["sim"])
        task_config = dict(manifest["task"])
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
        algorithm = create_algorithm(str(algorithm_config["plugin"]))
        algorithm.load(run.checkpoint)
    env = build_test_env(agent_config, sim_config, task_config, perturbation=perturbation)
    try:
        run.validate(env.sim)
        return rollout_metrics(env.sim, algorithm, episodes, seed=seed, max_steps=max_steps)
    finally:
        env.sim.close()


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
