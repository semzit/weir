from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from weir.core.interfaces import SimStep
from weir.envs.backends.mujoco import MuJoCoSim
from weir.envs.utils import MODELS_DIR

MODELS = MODELS_DIR
CART_POLE = MODELS / "cartpole.xml"
MENAGERIE = MODELS / "menagerie" / "berkeley_humanoid" / "berkeley_humanoid.xml"


def make_sim(model_path: Path, *, task: dict | None = None, **sim_config: object) -> MuJoCoSim:
    sim = MuJoCoSim()
    task_config = task if task is not None else {"name": "survive", "params": {}}
    sim.load(
        {"name": model_path.stem, "model": str(model_path)},
        {"time_limit": 10.0, "initial_noise": 0.0, **sim_config, "task": task_config},
    )
    return sim


def test_shapes_match_model() -> None:
    sim = make_sim(CART_POLE)
    shape = sim.observation_shape()
    obs = sim.reset()
    assert obs.shape == shape.dims
    assert obs.dtype == np.float32
    assert shape.dtype == "float32"

    action_shape = sim.action_shape()
    assert action_shape.dims == (1,)
    assert action_shape.low is not None
    assert action_shape.high is not None
    assert action_shape.low.shape == (1,)


_NO_CTRLRANGE_XML = """\
<mujoco model="no_ctrlrange">
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.01"/>
    <body name="pole" pos="0 0 0.1">
      <joint name="hinge" type="hinge" axis="0 1 0" range="-1.0 1.0"/>
      <geom name="arm" type="capsule" size="0.02 0.2" pos="0 0 0.2"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="arm" joint="hinge" gear="10"/>
  </actuator>
</mujoco>
"""


def test_action_shape_rejects_degenerate_ctrlrange(tmp_path: Path) -> None:
    model = tmp_path / "no_ctrlrange.xml"
    model.write_text(_NO_CTRLRANGE_XML, encoding="utf-8")
    sim = make_sim(model)
    with pytest.raises(ValueError, match="degenerate"):
        sim.action_shape()
    sim.close()


def test_step_returns_simstep_and_advances() -> None:
    sim = make_sim(CART_POLE)
    before = sim.reset()
    result = sim.step(np.array([1.0], dtype=np.float32))
    assert isinstance(result, SimStep)
    assert result.observation.shape == sim.observation_shape().dims
    assert isinstance(result.reward, float)
    assert isinstance(result.terminated, bool)
    assert isinstance(result.truncated, bool)
    assert not np.array_equal(before, result.observation)


def test_seeded_resets_and_trajectories_are_reproducible() -> None:
    sim_a = make_sim(MENAGERIE)
    sim_b = make_sim(MENAGERIE)
    a = sim_a.reset()
    b = sim_b.reset()
    assert np.array_equal(a, b)
    sim_a.close()
    sim_b.close()

    sim_a = make_sim(MENAGERIE, initial_noise=0.1)
    sim_b = make_sim(MENAGERIE, initial_noise=0.1)
    sim_c = make_sim(MENAGERIE, initial_noise=0.1)
    a = sim_a.reset(seed=1)
    b = sim_b.reset(seed=2)
    c = sim_c.reset(seed=1)
    assert np.array_equal(a, c)
    assert not np.array_equal(a, b)
    sim_a.close()
    sim_b.close()
    sim_c.close()

    def trajectory(seed: int) -> list[np.ndarray]:
        sim = make_sim(MENAGERIE, initial_noise=0.05)
        obs = [sim.reset(seed=seed).copy()]
        for _ in range(5):
            obs.append(sim.step(np.zeros(12, dtype=np.float32)).observation.copy())
        sim.close()
        return obs

    for x, y in zip(trajectory(7), trajectory(7), strict=True):
        assert np.array_equal(x, y)


def test_truncates_at_time_limit() -> None:
    sim = make_sim(CART_POLE, time_limit=0.05)
    sim.reset()
    result = None
    for _ in range(10):
        result = sim.step(np.zeros(1, dtype=np.float32))
    assert result is not None
    assert result.truncated is True


def test_use_before_load_raises() -> None:
    sim = MuJoCoSim()
    with pytest.raises(RuntimeError, match="load"):
        sim.reset()
    with pytest.raises(RuntimeError, match="load"):
        sim.observation_shape()


def test_unknown_task_raises() -> None:
    sim = MuJoCoSim()
    with pytest.raises(ValueError, match="Unknown task"):
        sim.load(
            {"name": "x", "model": str(CART_POLE)},
            {
                "task": {"name": "nope"},
                "time_limit": 10.0,
                "initial_noise": 0.0,
            },
        )


def test_walk_forward_task_terminates_on_fall() -> None:
    sim = make_sim(
        MENAGERIE,
        task={"name": "walk_forward", "params": {"min_height": 0.9}},
    )
    sim.reset(seed=0)
    result = None
    for _ in range(300):
        sim.apply_perturbation(np.array([300.0, 0.0, 0.0]))
        result = sim.step(np.zeros(12, dtype=np.float32))
        if result.terminated:
            break
    assert result is not None
    assert isinstance(result, SimStep)
    assert result.terminated is True
    assert result.observation[2] < 0.9


def test_walk_forward_injects_nq_from_model() -> None:
    def reward(push: bool) -> float:
        sim = make_sim(
            MENAGERIE,
            task={"name": "walk_forward", "params": {"min_height": 0.2}},
        )
        sim.reset(seed=0)
        for _ in range(4):
            if push:
                sim.apply_perturbation(np.array([300.0, 0.0, 0.0]))
            result = sim.step(np.zeros(12, dtype=np.float32))
        return float(result.reward)

    # The pushed rollout has qvel[0] > 0; a higher reward proves the forward
    # term reads it via the injected nq=19 (a wrong nq would read a joint
    # position, ~0, and the two rewards would be equal).
    assert reward(push=True) > reward(push=False)


def test_apply_perturbation_targets_the_requested_body() -> None:
    """The body index selects the model body: 1 is the cart, 2 is the pole."""

    def observation_after_push(body: int) -> np.ndarray:
        sim = make_sim(CART_POLE)
        sim.reset(seed=0)
        sim.apply_perturbation(np.array([50.0, 0.0, 0.0]), body=body)
        result = sim.step(np.zeros(1, dtype=np.float32))
        sim.close()
        return result.observation

    cart_push = observation_after_push(1)
    pole_push = observation_after_push(2)
    # obs = [cart_x, pole_angle, cart_v, pole_omega]
    assert abs(cart_push[2]) > abs(pole_push[2])
    assert abs(pole_push[3]) > abs(cart_push[3])


def test_humanoid_rollout_conforms_to_shapes() -> None:
    sim = make_sim(MENAGERIE, task={"name": "standing", "params": {"min_height": 0.2}})
    obs_shape = sim.observation_shape()
    act_shape = sim.action_shape()
    low = act_shape.low
    high = act_shape.high
    assert low is not None and high is not None
    rng = np.random.default_rng(7)
    obs = sim.reset(seed=7)
    assert obs.shape == obs_shape.dims
    for _ in range(10):
        action = rng.uniform(low, high).astype(np.float32)
        result = sim.step(action)
        assert result.observation.shape == obs_shape.dims
        assert isinstance(result.reward, float)
        assert isinstance(result.terminated, bool)
        assert isinstance(result.truncated, bool)
    sim.close()


def test_initial_noise_randomizes_unseeded_resets() -> None:
    sim = make_sim(CART_POLE, initial_noise=0.0)
    a = sim.reset()
    b = sim.reset()
    assert np.array_equal(a, b)

    noisy = make_sim(MENAGERIE, initial_noise=0.1)
    c = noisy.reset()
    d = noisy.reset()
    assert not np.array_equal(c, d)
    noisy.close()


def test_initial_noise_works_without_freejoint() -> None:
    plain = make_sim(CART_POLE)
    noisy = make_sim(CART_POLE, initial_noise=0.1)
    assert not np.array_equal(plain.reset(seed=1), noisy.reset(seed=1))
    assert np.array_equal(noisy.reset(seed=1), noisy.reset(seed=1))
    plain.close()
    noisy.close()


def test_initial_noise_perturbs_joints_but_not_root() -> None:
    plain = make_sim(MENAGERIE)
    noisy = make_sim(MENAGERIE, initial_noise=0.5)
    baseline = plain.reset(seed=1)
    perturbed = noisy.reset(seed=1)
    assert np.array_equal(baseline[:7], perturbed[:7])
    assert not np.array_equal(baseline[7:], perturbed[7:])
    plain.close()
    noisy.close()


def test_load_requires_complete_sim_config() -> None:
    sim = MuJoCoSim()
    with pytest.raises(ValueError, match="task"):
        sim.load(
            {"name": "x", "model": str(CART_POLE)},
            {"time_limit": 10.0, "initial_noise": 0.0},
        )
    with pytest.raises(ValueError, match="validation error"):
        sim.load(
            {"name": "x", "model": str(CART_POLE)},
            {"task": {"name": "survive", "params": {}}},
        )
