from __future__ import annotations

import numpy as np
import pytest
from conftest import NEUTRAL_RANDOMIZATION, RANDOMIZATION

from weir.core.interfaces import DomainRandomizable, SimBackend
from weir.envs.backends.mujoco import MuJoCoSim
from weir.envs.utils import MODELS_DIR
from weir.envs.wrappers.randomized import RandomizedSim

CART_POLE = MODELS_DIR / "cartpole.xml"
BERKELEY = MODELS_DIR / "menagerie" / "berkeley_humanoid" / "berkeley_humanoid.xml"


def make_wrapped(config: dict, model_path=CART_POLE) -> RandomizedSim:
    inner = MuJoCoSim()
    sim = RandomizedSim(inner, {**RANDOMIZATION, **config})
    sim.load(
        {"name": "test", "model": str(model_path)},
        {
            "task": {"name": "survive", "params": {}},
            "time_limit": 10.0,
            "initial_noise": 0.0,
        },
    )
    return sim


def make_bare(model_path=CART_POLE) -> MuJoCoSim:
    sim = MuJoCoSim()
    sim.load(
        {"name": "test", "model": str(model_path)},
        {
            "task": {"name": "survive", "params": {}},
            "time_limit": 10.0,
            "initial_noise": 0.0,
        },
    )
    return sim


def test_wrapped_sim_conforms_to_protocol() -> None:
    assert isinstance(make_wrapped(RANDOMIZATION), SimBackend)


def test_mujoco_sim_is_domain_randomizable() -> None:
    assert isinstance(make_bare(CART_POLE), DomainRandomizable)


def test_neutral_config_is_pass_through() -> None:
    def roll(sim: SimBackend) -> list[np.ndarray]:
        obs = [sim.reset(seed=3).copy()]
        action = np.zeros(12, dtype=np.float32)
        for _ in range(5):
            obs.append(sim.step(action).observation.copy())
        return obs

    bare = roll(make_bare(BERKELEY))
    wrapped = roll(make_wrapped(NEUTRAL_RANDOMIZATION, BERKELEY))
    assert np.array_equal(bare, wrapped)


def test_missing_randomization_keys_raise() -> None:
    with pytest.raises(ValueError, match="validation error"):
        RandomizedSim(MuJoCoSim(), {})


def test_observation_noise_is_zero_mean() -> None:
    bare = make_bare(BERKELEY)
    wrapped = make_wrapped({"noise_std": 0.5}, BERKELEY)
    bare.reset(seed=1)
    wrapped.reset(seed=1)
    action = np.zeros(12, dtype=np.float32)
    diffs = np.stack(
        [wrapped.step(action).observation - bare.step(action).observation for _ in range(40)]
    )
    assert diffs.shape == (40, 37)
    assert abs(float(diffs.mean())) < 0.15
    assert abs(float(diffs.std()) - 0.5) < 0.15


def test_domain_randomization_changes_dynamics() -> None:
    bare = make_bare(CART_POLE)
    wrapped = make_wrapped({"mass_scale": [2.0, 2.0]}, CART_POLE)
    bare.reset(seed=0)
    wrapped.reset(seed=0)
    action = np.ones(1, dtype=np.float32)
    bare_obs = bare.step(action).observation
    wrapped_obs = wrapped.step(action).observation
    assert not np.array_equal(bare_obs, wrapped_obs)


def test_latency_delays_actions() -> None:
    bare = make_bare(CART_POLE)
    expected = make_bare(CART_POLE)
    wrapped = make_wrapped({"latency_steps": 1}, CART_POLE)
    for sim in (bare, expected, wrapped):
        sim.reset(seed=0)

    action = np.array([1.0], dtype=np.float32)
    bare_result = bare.step(action)
    zeros_result = expected.step(np.zeros(1, dtype=np.float32))
    wrapped_result = wrapped.step(action)

    assert np.allclose(wrapped_result.observation, zeros_result.observation, atol=1e-6)
    assert not np.array_equal(wrapped_result.observation, bare_result.observation)


def test_same_seed_reproduces_hardened_trajectory() -> None:
    def roll(seed: int) -> list[np.ndarray]:
        sim = make_wrapped({"mass_scale": [0.8, 1.2], "noise_std": 0.1}, BERKELEY)
        obs = [sim.reset(seed=seed).copy()]
        action = np.zeros(12, dtype=np.float32)
        for _ in range(5):
            obs.append(sim.step(action).observation.copy())
        return obs

    for x, y in zip(roll(11), roll(11), strict=True):
        assert np.array_equal(x, y)


def test_mujoco_sim_domain_params_roundtrip() -> None:
    sim = make_bare(CART_POLE)
    params = sim.domain_params()
    assert params["body_mass"].size > 0

    scaled = np.asarray(params["body_mass"]) * 2.0
    sim.apply_domain_params({"body_mass": scaled})
    again = sim.domain_params()
    assert np.allclose(again["body_mass"], scaled)
