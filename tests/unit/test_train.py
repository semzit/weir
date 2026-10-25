from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from weir.algo.ppo import PPOAlgorithm
from weir.core.application.evaluate import run_eval
from weir.core.application.train import run
from weir.core.utils import CONFIG_DIR

ROOT = Path(__file__).parents[2]
CONFIG_RELATIVE = str(Path(os.path.relpath(CONFIG_DIR, Path(__file__).parent)))


def make_config(overrides: list[str] | None = None) -> DictConfig:
    with initialize(version_base=None, config_path=CONFIG_RELATIVE):
        return compose(config_name="train", overrides=overrides or [])


def test_config_composition() -> None:
    cfg = make_config()
    assert cfg.agent.name == "cartpole"
    assert cfg.task.name == "balance"
    assert cfg.sim.plugin == "mujoco"
    assert cfg.algo.plugin == "ppo"
    assert cfg.train.total_steps == 100000

    agent_cfg = make_config(["agent=humanoid"])
    assert agent_cfg.agent.name == "berkeley_humanoid"
    assert "berkeley_humanoid.xml" in str(agent_cfg.agent.model)

    task_cfg = make_config(["task=standing"])
    assert task_cfg.task.name == "standing"
    assert task_cfg.task.params.min_height == 0.3


def test_train_then_eval(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = make_config(["train.total_steps=32", "algo.n_steps=32"])
    monkeypatch.chdir(tmp_path)
    with caplog.at_level("INFO", logger="weir"):
        result = run(cfg)

    assert result["steps"] == 32
    assert (tmp_path / "checkpoint.zip").exists()
    assert (tmp_path / "checkpoint.meta.json").exists()
    events = [record.message for record in caplog.records]
    assert "train.start" in events
    assert "train.complete" in events
    assert caplog.records[0].sim == "mujoco"
    assert caplog.records[0].algo == "ppo"

    checkpoint = tmp_path / "checkpoint.zip"
    metrics = run_eval(checkpoint, episodes=1, seed=0, max_steps=20)
    assert set(metrics) == {
        "mean_reward",
        "mean_episode_length",
        "mean_progress",
        "episodes_completed",
    }
    assert metrics["mean_episode_length"] == 20.0
    assert metrics["mean_progress"] == 0.0


def test_run_eval_override_rules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    run(make_config(["train.total_steps=32", "algo.n_steps=32"]))
    checkpoint = tmp_path / "checkpoint.zip"

    with pytest.raises(ValueError, match="fixed by the checkpoint"):
        run_eval(checkpoint, episodes=1, seed=0, max_steps=10, overrides=["agent=humanoid"])
    with pytest.raises(ValueError, match="fixed by the checkpoint"):
        run_eval(checkpoint, episodes=1, seed=0, max_steps=10, overrides=["algo.n_steps=128"])

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
