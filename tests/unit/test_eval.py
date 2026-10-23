from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from test_rollout import FakeAlgorithm, FakeSim

from weir.cli.eval import main
from weir.core.configs import RandomizationConfig


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
    monkeypatch.setattr("weir.core.run.create_sim", lambda _name: FakeSim())
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


def test_cli_passes_perturbation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"fake-checkpoint")
    (tmp_path / "checkpoint.meta.json").write_text(
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
    seen: dict[str, object] = {}

    def fake_run_eval(checkpoint: Path, **kwargs: object) -> dict[str, float]:
        seen.update(kwargs)
        return {
            "mean_reward": 1.0,
            "mean_episode_length": 1.0,
            "mean_progress": 1.0,
            "episodes_completed": 1.0,
        }

    monkeypatch.setattr("weir.cli.eval.run_eval", fake_run_eval)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weir-eval",
            "--checkpoint",
            str(checkpoint),
            "--perturb-force",
            "5",
            "--perturb-prob",
            "1",
            "--perturb-body",
            "2",
        ],
    )

    assert main() == 0
    perturbation = seen["perturbation"]
    assert isinstance(perturbation, RandomizationConfig)
    assert perturbation.perturbation_force == 5.0
    assert perturbation.perturbation_prob == 1.0
    assert perturbation.perturbation_body == 2


def test_cli_returns_nonzero_on_missing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
