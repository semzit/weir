from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from weir.algo.ppo import PPOAlgorithm
from weir.cli.eval import run_eval
from weir.cli.train import run
from weir.core.run import Run
from weir.core.utils import CONFIG_DIR
from weir.envs.backends.mujoco import MuJoCoSim
from weir.envs.wrappers.randomized import RandomizedSim

ROOT = Path(__file__).parents[2]
CONFIG_RELATIVE = str(Path(os.path.relpath(CONFIG_DIR, Path(__file__).parent)))

RANDOMIZATION = {
    "mass_scale": [0.8, 1.2],
    "friction_scale": [0.5, 1.5],
    "damping_scale": [0.5, 1.5],
    "noise_std": 0.0,
    "action_noise_std": 0.0,
    "latency_steps": 0,
    "perturbation_force": 0.0,
    "perturbation_prob": 0.0,
}


def make_config(overrides: list[str] | None = None) -> DictConfig:
    with initialize(version_base=None, config_path=CONFIG_RELATIVE):
        return compose(config_name="train", overrides=overrides or [])


def test_config_composes_defaults() -> None:
    cfg = make_config()
    assert cfg.agent.name == "cartpole"
    assert cfg.task.name == "balance"
    assert cfg.sim.plugin == "mujoco"
    assert cfg.algo.plugin == "ppo"
    assert cfg.train.total_steps == 100000


def test_config_agent_override_swaps_model() -> None:
    cfg = make_config(["agent=humanoid"])
    assert cfg.agent.name == "berkeley_humanoid"
    assert "berkeley_humanoid.xml" in str(cfg.agent.model)


def test_config_task_override_applies_params() -> None:
    cfg = make_config(["task=standing"])
    assert cfg.task.name == "standing"
    assert cfg.task.params.min_height == 0.3


def test_run_trains_a_short_run(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = make_config(["train.total_steps=128", "algo.n_steps=64"])
    monkeypatch.chdir(tmp_path)
    with caplog.at_level("INFO", logger="weir"):
        result = run(cfg)

    assert result["steps"] == 128
    assert (tmp_path / "checkpoint.zip").exists()
    assert (tmp_path / "checkpoint.meta.json").exists()
    events = [record.message for record in caplog.records]
    assert "train.start" in events
    assert "train.complete" in events
    assert caplog.records[0].sim == "mujoco"
    assert caplog.records[0].algo == "ppo"


def test_run_manifest_drives_eval(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    run(make_config(["train.total_steps=32", "algo.n_steps=32"]))

    checkpoint = tmp_path / "checkpoint.zip"
    assert (tmp_path / "checkpoint.meta.json").exists()
    metrics = run_eval(checkpoint, episodes=1, seed=0, max_steps=20)
    assert set(metrics) == {
        "mean_reward",
        "mean_episode_length",
        "total_forward_distance",
        "mean_forward_distance_per_episode",
        "episodes_completed",
    }
    assert metrics["mean_episode_length"] == 20.0


def test_run_eval_override_layers_on_manifest_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    run(make_config(["train.total_steps=32", "algo.n_steps=32"]))
    checkpoint = tmp_path / "checkpoint.zip"

    metrics = run_eval(
        checkpoint, episodes=1, seed=0, max_steps=10, overrides=["train.total_steps=5"]
    )
    assert metrics["mean_episode_length"] == 10.0


def test_run_eval_rejects_agent_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    run(make_config(["train.total_steps=32", "algo.n_steps=32"]))
    checkpoint = tmp_path / "checkpoint.zip"

    with pytest.raises(ValueError, match="fixed by the checkpoint"):
        run_eval(checkpoint, episodes=1, seed=0, max_steps=10, overrides=["agent=humanoid"])


def test_run_eval_rejects_algo_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    run(make_config(["train.total_steps=32", "algo.n_steps=32"]))
    checkpoint = tmp_path / "checkpoint.zip"

    with pytest.raises(ValueError, match="fixed by the checkpoint"):
        run_eval(checkpoint, episodes=1, seed=0, max_steps=10, overrides=["algo.n_steps=128"])


def test_run_eval_accepts_sim_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    run(make_config(["train.total_steps=32", "algo.n_steps=32"]))
    checkpoint = tmp_path / "checkpoint.zip"

    metrics = run_eval(checkpoint, episodes=1, seed=0, max_steps=10, overrides=["sim.robust=true"])
    assert metrics["mean_episode_length"] == 10.0


def test_run_resumes_from_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    run(make_config(["train.total_steps=32", "algo.n_steps=32"]))
    checkpoint = tmp_path / "checkpoint.zip"
    assert checkpoint.exists()

    cfg = make_config(
        [
            "train.total_steps=32",
            "algo.n_steps=32",
            f"algo.checkpoint={checkpoint}",
        ]
    )
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    monkeypatch.chdir(resume_dir)
    result = run(cfg)
    assert result["steps"] == 32


def test_build_sim_wraps_when_robust() -> None:
    assert isinstance(Run.build_sim({"plugin": "mujoco", "robust": False}), MuJoCoSim)
    hardened = Run.build_sim({"plugin": "mujoco", "robust": True, "randomization": RANDOMIZATION})
    assert isinstance(hardened, RandomizedSim)


def test_run_is_reproducible_with_seed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    overrides = ["train.total_steps=64", "algo.n_steps=32"]

    def train_once(subdir: str, seed: int) -> Path:
        run_dir = tmp_path / subdir
        run_dir.mkdir()
        monkeypatch.chdir(run_dir)
        run(make_config([*overrides, f"train.seed={seed}"]))
        return run_dir / "checkpoint.zip"

    def deterministic_actions(checkpoint: Path) -> np.ndarray:
        algo = PPOAlgorithm()
        algo.load(checkpoint)
        return algo.act(np.zeros(4, dtype=np.float32), deterministic=True)

    first = train_once("first", seed=7)
    second = train_once("second", seed=7)
    different = train_once("different", seed=8)

    assert np.array_equal(deterministic_actions(first), deterministic_actions(second))
    assert not np.array_equal(deterministic_actions(first), deterministic_actions(different))
