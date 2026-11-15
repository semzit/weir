from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import RANDOMIZATION

from weir.core.configs import RandomizationConfig
from weir.core.run import Run
from weir.envs.backends.mujoco import MuJoCoSim
from weir.envs.utils import MODELS_DIR
from weir.envs.wrappers.randomized import RandomizedSim

CART_POLE = MODELS_DIR / "cartpole.xml"
BERKELEY = MODELS_DIR / "menagerie" / "berkeley_humanoid" / "berkeley_humanoid.xml"


def make_run(
    tmp_path: Path,
    *,
    model: Path = CART_POLE,
    obs_dims: tuple[int, ...] = (4,),
    act_dims: tuple[int, ...] = (1,),
    agent_name: str = "test_agent",
    robust: bool = False,
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
                    "robust": robust,
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
    assert run.checkpoint == Path("/nowhere/checkpoint.zip")
    assert run.config is None
    assert run.obs_dim() is None
    assert run.plugin("algo") is None


def test_require_config_rejects_legacy_checkpoint() -> None:
    run = Run.open(Path("/nowhere/checkpoint.zip"))
    with pytest.raises(ValueError, match="No manifest"):
        run.require_config()


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


def test_sim_builds_from_relative_manifest_model(tmp_path: Path) -> None:
    run = make_run(tmp_path, model=Path("cartpole.xml"))
    sim = run.sim()
    try:
        assert sim.observation_shape().dims == (4,)
    finally:
        sim.close()


def test_test_env_exposes_frames_for_render(tmp_path: Path) -> None:
    env = make_run(tmp_path).test_env()
    try:
        assert env.frames is env.sim
    finally:
        env.sim.close()


def test_test_env_keeps_camera_when_robust(tmp_path: Path) -> None:
    env = make_run(tmp_path, robust=True).test_env()
    try:
        assert isinstance(env.sim, RandomizedSim)
        assert isinstance(env.frames, MuJoCoSim)
    finally:
        env.sim.close()


def test_test_env_applies_perturbation(tmp_path: Path) -> None:
    perturbation = RandomizationConfig.neutral(force=5.0, prob=1.0, body=2)
    env = make_run(tmp_path).test_env(perturbation=perturbation)
    try:
        assert isinstance(env.sim, RandomizedSim)
        assert isinstance(env.frames, MuJoCoSim)
    finally:
        env.sim.close()


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
