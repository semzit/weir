from __future__ import annotations

import json
from pathlib import Path

import pytest

from weir.core.run import Run
from weir.envs.utils import MODELS_DIR

CART_POLE = MODELS_DIR / "cartpole.xml"
BERKELEY = MODELS_DIR / "menagerie" / "berkeley_humanoid" / "berkeley_humanoid.xml"

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


def make_run(
    tmp_path: Path,
    *,
    model: Path = CART_POLE,
    obs_dims: tuple[int, ...] = (4,),
    act_dims: tuple[int, ...] = (1,),
    agent_name: str = "test_agent",
) -> Run:
    manifest = tmp_path / "checkpoint.meta.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "agent": {"name": agent_name, "model": str(model)},
                "task": {"name": "survive", "params": {}},
                "sim": {
                    "plugin": "mujoco",
                    "time_limit": 10.0,
                    "initial_noise": 0.0,
                    "robust": False,
                    "randomization": RANDOMIZATION,
                },
                "algo": {"plugin": "ppo"},
                "train": {"seed": 0, "total_steps": 100},
                "observation_shape": {"dims": obs_dims, "dtype": "float32"},
                "action_shape": {"dims": act_dims, "dtype": "float32"},
            }
        ),
        encoding="utf-8",
    )
    return Run.open(tmp_path / "checkpoint.zip")


def test_open_without_manifest_is_legacy() -> None:
    run = Run.open(Path("/nowhere/checkpoint.zip"))
    assert run.config is None
    assert run.obs_dim() is None
    assert run.plugin("algo") is None


def test_open_reads_manifest(tmp_path: Path) -> None:
    run = make_run(tmp_path, obs_dims=(25,), act_dims=(6,))
    assert run.config is not None
    assert run.obs_dim() == 25
    assert run.plugin("algo") == "ppo"
    assert run.config["agent"]["name"] == "test_agent"


def test_sim_builds_from_manifest(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    sim = run.sim()
    assert sim.observation_shape().dims == (4,)
    assert sim.action_shape().dims == (1,)
    sim.close()


def test_validate_passes_on_matching_shapes(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    sim = run.sim()
    run.validate(sim)
    sim.close()


def test_validate_rejects_mismatched_observation_shape(tmp_path: Path) -> None:
    run = make_run(tmp_path, obs_dims=(25,), act_dims=(1,))
    sim = run.sim()
    with pytest.raises(ValueError, match="trained with observation shape"):
        run.validate(sim)
    sim.close()


def test_validate_rejects_mismatched_action_shape(tmp_path: Path) -> None:
    run = make_run(tmp_path, obs_dims=(4,), act_dims=(6,))
    sim = run.sim()
    with pytest.raises(ValueError, match="trained with action shape"):
        run.validate(sim)
    sim.close()


def test_validate_ignores_humanoid_nq_injection(tmp_path: Path) -> None:
    run = make_run(tmp_path, model=BERKELEY, obs_dims=(37,), act_dims=(12,))
    sim = run.sim()
    run.validate(sim)
    sim.close()
