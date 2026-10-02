from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from weir.cli.eval import main, rollout_metrics
from weir.core.interfaces import Shape, SimStep


class FakeSim:
    """SimBackend double whose obs[0] advances by delta and terminates after K steps."""

    def __init__(self, delta: float = 1.0, terminate_after: int = 3) -> None:
        self.delta = delta
        self.terminate_after = terminate_after
        self.steps = 0
        self.agent_config: dict[str, Any] | None = None
        self.sim_config: dict[str, Any] | None = None

    def load(self, agent_config: dict[str, Any], sim_config: dict[str, Any]) -> None:
        self.agent_config = agent_config
        self.sim_config = sim_config

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.steps = 0
        return np.zeros(2, dtype=np.float32)

    def step(self, actions: Any) -> SimStep:
        self.steps += 1
        return SimStep(
            observation=np.array([self.delta * self.steps, 0.0], dtype=np.float32),
            reward=self.delta,
            terminated=self.steps >= self.terminate_after,
            truncated=False,
        )

    def observation_shape(self) -> Shape:
        return Shape(dims=(2,), dtype="float32")

    def action_shape(self) -> Shape:
        return Shape(dims=(1,), dtype="float32")

    def close(self) -> None:
        pass


class SeededFakeSim(FakeSim):
    """FakeSim whose initial root x depends on the reset seed."""

    def reset(self, seed: int | None = None) -> np.ndarray:
        super().reset(seed)
        initial_x = 0.0 if seed is None else float(seed % 3)
        return np.array([initial_x, 0.0], dtype=np.float32)


class FakeAlgorithm:
    """AlgorithmPlugin double that always acts with a zero action."""

    def __init__(self) -> None:
        self.deterministic: list[bool] = []

    def configure(
        self, observation_shape: Shape, action_shape: Shape, config: dict[str, Any]
    ) -> None:
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.config = config

    def learn(self, env: Any, total_steps: int, callback: Any | None = None) -> dict[str, float]:
        return {"total_steps": float(total_steps)}

    def act(self, observations: Any, deterministic: bool = False) -> np.ndarray:
        self.deterministic.append(deterministic)
        return np.zeros(1, dtype=np.float32)

    def save(self, path: Path) -> None:
        Path(path).write_text("fake-checkpoint", encoding="utf-8")

    def load(self, path: Path) -> None:
        pass

    def export_policy(self) -> Any:
        return None


def test_rollout_metrics_matches_hand_computed_values() -> None:
    metrics = rollout_metrics(
        FakeSim(delta=1.0, terminate_after=3),
        FakeAlgorithm(),
        episodes=2,
        seed=0,
        max_steps=100,
    )
    assert metrics == {
        "mean_reward": 3.0,
        "mean_episode_length": 3.0,
        "total_forward_distance": 6.0,
        "mean_forward_distance_per_episode": 3.0,
        "episodes_completed": 2.0,
    }


def test_rollout_metrics_uses_deterministic_actions() -> None:
    algorithm = FakeAlgorithm()
    rollout_metrics(FakeSim(), algorithm, episodes=1, seed=0, max_steps=100)
    assert algorithm.deterministic == [True, True, True]


def test_rollout_metrics_caps_episodes_at_max_steps() -> None:
    metrics = rollout_metrics(
        FakeSim(terminate_after=5),
        FakeAlgorithm(),
        episodes=2,
        seed=0,
        max_steps=2,
    )
    assert metrics["mean_reward"] == 2.0
    assert metrics["mean_episode_length"] == 2.0
    assert metrics["total_forward_distance"] == 4.0
    assert metrics["episodes_completed"] == 0.0


def test_rollout_metrics_is_seed_deterministic() -> None:
    first = rollout_metrics(SeededFakeSim(), FakeAlgorithm(), episodes=4, seed=7, max_steps=100)
    second = rollout_metrics(SeededFakeSim(), FakeAlgorithm(), episodes=4, seed=7, max_steps=100)
    other = rollout_metrics(SeededFakeSim(), FakeAlgorithm(), episodes=4, seed=8, max_steps=100)
    assert first == second
    assert first != other


def test_cli_evaluates_and_prints_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"fake-checkpoint")
    manifest = tmp_path / "checkpoint.meta.json"
    manifest.write_text(
        json.dumps(
            {
                "agent": {"name": "cartpole", "model": "irrelevant"},
                "task": {"name": "balance", "params": {}},
                "sim": {"plugin": "mujoco"},
                "algo": {"plugin": "ppo"},
                "observation_shape": {"dims": [2], "dtype": "float32"},
                "action_shape": {"dims": [1], "dtype": "float32"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("weir.core.run.Run.build_sim", lambda _name: FakeSim())
    monkeypatch.setattr("weir.cli.eval.create_algorithm", lambda _name: FakeAlgorithm())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weir-eval",
            "--checkpoint",
            str(checkpoint),
            "--episodes",
            "2",
            "--override",
            "task.params.x_threshold=2.5",
        ],
    )

    assert main() == 0
    captured = capsys.readouterr()
    assert "mean_reward: 3.000" in captured.out
    assert "episodes_completed: 2.000" in captured.out


def test_cli_returns_nonzero_on_missing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("weir.core.run.Run.build_sim", lambda _name: FakeSim())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weir-eval",
            "--checkpoint",
            str(tmp_path / "missing.zip"),
        ],
    )

    assert main() == 1
