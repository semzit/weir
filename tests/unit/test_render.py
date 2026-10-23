from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from conftest import PPO_CONFIG

from weir.cli.render import main as render_main
from weir.cli.render import render_episode
from weir.core.configs import RandomizationConfig
from weir.core.interfaces import SimStep
from weir.core.run import TestEnv as LoadedTestEnv
from weir.core.utils import create_algorithm
from weir.envs.backends.mujoco import MuJoCoSim
from weir.envs.utils import MODELS_DIR
from weir.envs.wrappers.randomized import RandomizedSim

CART_POLE = MODELS_DIR / "cartpole.xml"


def make_sim() -> MuJoCoSim:
    sim = MuJoCoSim()
    sim.load(
        {"name": "cartpole", "model": str(CART_POLE)},
        {
            "task": {"name": "survive", "params": {}},
            "time_limit": 5.0,
            "initial_noise": 0.0,
        },
    )
    return sim


def test_render_frame_returns_rgb_uint8() -> None:
    sim = make_sim()
    try:
        frame = sim.render_frame(width=160, height=120)
    finally:
        sim.close()
    assert frame.shape == (120, 160, 3)
    assert frame.dtype == np.uint8


def test_render_frame_resizes_renderer() -> None:
    sim = make_sim()
    try:
        small = sim.render_frame(width=64, height=48)
        large = sim.render_frame(width=160, height=120)
    finally:
        sim.close()
    assert small.shape == (48, 64, 3)
    assert large.shape == (120, 160, 3)


def test_render_episode_writes_mp4(tmp_path: Path) -> None:
    sim = make_sim()
    algo = create_algorithm("ppo")
    algo.configure(sim.observation_shape(), sim.action_shape(), PPO_CONFIG)
    output = tmp_path / "episode.mp4"
    result = render_episode(
        sim,
        sim,
        algo,
        output,
        frames_to_capture=10,
        width=160,
        height=120,
        fps=15,
    )
    sim.close()
    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0
    assert output.read_bytes()[4:8] == b"ftyp"


def test_render_episode_stops_at_termination(tmp_path: Path) -> None:
    sim = make_sim()
    algo = create_algorithm("ppo")
    algo.configure(sim.observation_shape(), sim.action_shape(), PPO_CONFIG)
    output = tmp_path / "short.mp4"
    render_episode(
        sim,
        sim,
        algo,
        output,
        frames_to_capture=1000,
        width=160,
        height=120,
        fps=15,
        seed=0,
    )
    sim.close()
    assert output.exists()


def test_render_episode_paces_frames_to_realtime(tmp_path: Path) -> None:
    sim = make_sim()  # cartpole: dt = 0.02
    algo = create_algorithm("ppo")
    algo.configure(sim.observation_shape(), sim.action_shape(), PPO_CONFIG)
    output = tmp_path / "paced.mp4"
    render_episode(
        sim,
        sim,
        algo,
        output,
        frames_to_capture=6,
        width=160,
        height=120,
        fps=30,
        seed=0,
    )
    sim.close()
    # dt=0.02, fps=30 -> interval = round(1 / 0.6) = 2 sim steps per frame
    frames = imageio.mimread(output)
    assert len(frames) == 6


def test_render_episode_applies_perturbations(tmp_path: Path) -> None:
    """With perturb options, the sim receives pushes and they are released."""

    class RecordingSim(MuJoCoSim):
        def __init__(self) -> None:
            super().__init__()
            self.load(
                {"name": "cartpole", "model": str(CART_POLE)},
                {
                    "task": {"name": "survive", "params": {}},
                    "time_limit": 5.0,
                    "initial_noise": 0.0,
                },
            )
            self.pushes: list[tuple[float, int]] = []
            self.clears = 0

        def reset(self, seed: int | None = None) -> np.ndarray:
            return np.zeros(4, dtype=np.float32)

        @property
        def dt(self) -> float:
            return 0.02

        def step(self, action: object) -> object:
            step = type(
                "Step",
                (),
                {
                    "observation": np.zeros(4, dtype=np.float32),
                    "reward": 0.0,
                    "terminated": False,
                    "truncated": False,
                },
            )
            return step()

        def render_frame(self, width: int = 0, height: int = 0) -> np.ndarray:
            return np.zeros((height, width, 3), dtype=np.uint8)

        def apply_perturbation(self, force: object, body: int = 1) -> None:
            force_array = np.asarray(force, dtype=float)
            if np.all(force_array == 0.0):
                self.clears += 1
            else:
                self.pushes.append((float(force_array[0]), body))

    class FakeAlgorithm:
        def act(self, observation: object, deterministic: bool = False) -> np.ndarray:
            return np.zeros(1, dtype=np.float32)

    sim = RecordingSim()
    algo = FakeAlgorithm()
    env = RandomizedSim(
        sim,
        RandomizationConfig.neutral(force=5.0, prob=1.0, body=2).model_dump(),
    )
    render_episode(
        env,
        sim,
        algo,  # type: ignore[arg-type]
        tmp_path / "perturbed.mp4",
        frames_to_capture=4,
        frame_interval=1,
        width=160,
        height=120,
        fps=15,
        seed=0,
    )
    sim.close()
    assert len(sim.pushes) == 3  # initial frame consumes one frame slot
    assert all(body == 2 for _, body in sim.pushes)
    assert sim.clears == 3  # every push was released the same step


def test_render_episode_feeds_updated_observations(tmp_path: Path) -> None:
    """The policy must see each post-step observation, not just the reset state."""

    class RecordingAlgorithm:
        def __init__(self) -> None:
            self.seen: list[bytes] = []

        def act(self, observation: object, deterministic: bool = False) -> np.ndarray:
            self.seen.append(np.asarray(observation, dtype=np.float32).tobytes())
            return np.ones(1, dtype=np.float32)  # push the cart so the state evolves

    sim = make_sim()  # cartpole, survive: never terminates
    algo = RecordingAlgorithm()
    output = tmp_path / "recorded.mp4"
    render_episode(
        sim,
        sim,
        algo,  # type: ignore[arg-type]
        output,
        frames_to_capture=4,
        frame_interval=1,
        width=160,
        height=120,
        fps=15,
        seed=0,
    )
    sim.close()
    assert len(algo.seen) == 3  # initial frame consumes one frame slot
    assert len(set(algo.seen)) == 3  # every act saw a different state


def test_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSim(MuJoCoSim):
        def reset(self, seed: int | None = None) -> np.ndarray:
            return np.zeros(4, dtype=np.float32)

        @property
        def dt(self) -> float:
            return 0.02

        def step(self, action: object) -> SimStep:
            return SimStep(
                observation=np.zeros(4, dtype=np.float32),
                reward=0.0,
                terminated=False,
                truncated=False,
            )

        def progress(self, prev_observation: object, observation: object) -> float:
            return 0.0

        def render_frame(self, width: int = 0, height: int = 0) -> np.ndarray:
            return np.zeros((height, width, 3), dtype=np.uint8)

        def close(self) -> None: ...

    class FakeAlgorithm:
        def act(self, observation: object, deterministic: bool = False) -> np.ndarray:
            return np.zeros(1, dtype=np.float32)

    seen: dict[str, object] = {}

    class FakeRun:
        config = {"agent": {"name": "x"}}

        def test_env(self, *, perturbation: RandomizationConfig | None = None) -> LoadedTestEnv:
            seen["perturbation"] = perturbation
            sim = FakeSim()
            return LoadedTestEnv(sim=sim, frames=sim)

        def validate(self, sim: object) -> None: ...

        def algorithm(self) -> FakeAlgorithm:
            return FakeAlgorithm()

        @classmethod
        def open(cls, _checkpoint: object) -> FakeRun:
            return cls()

    monkeypatch.setattr("weir.cli.render.Run", FakeRun)
    output = tmp_path / "cli.mp4"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weir-render",
            "--checkpoint",
            str(tmp_path / "checkpoint.zip"),
            "--output",
            str(output),
            "--frames",
            "10",
            "--frame-interval",
            "1",
            "--width",
            "160",
            "--height",
            "120",
            "--fps",
            "15",
            "--perturb-force",
            "5",
            "--perturb-prob",
            "1",
            "--perturb-body",
            "2",
        ],
    )
    assert render_main() == 0
    assert output.exists()
    assert output.stat().st_size > 0
    perturbation = seen["perturbation"]
    assert isinstance(perturbation, RandomizationConfig)
    assert perturbation.perturbation_force == 5.0
    assert perturbation.perturbation_body == 2


def test_cli_requires_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["weir-render"])
    with pytest.raises(SystemExit):
        render_main()


def test_cli_rejects_legacy_checkpoint_without_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"fake-checkpoint")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weir-render",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "ckpt.mp4"),
        ],
    )

    assert render_main() == 1


def test_cli_plays_back_manifest_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSim(MuJoCoSim):
        def reset(self, seed: int | None = None) -> np.ndarray:
            return np.zeros(4, dtype=np.float32)

        @property
        def dt(self) -> float:
            return 0.02

        def step(self, action: object) -> object:
            step = type(
                "Step",
                (),
                {
                    "observation": np.zeros(4, dtype=np.float32),
                    "reward": 0.0,
                    "terminated": True,
                    "truncated": False,
                },
            )
            return step()

        def progress(self, prev_observation: object, observation: object) -> float:
            return 0.0

        def render_frame(self, width: int = 0, height: int = 0) -> np.ndarray:
            return np.zeros((height, width, 3), dtype=np.uint8)

        def close(self) -> None: ...

    class FakeAlgorithm:
        def act(self, observation: object, deterministic: bool = False) -> np.ndarray:
            return np.zeros(1, dtype=np.float32)

    class FakeRun:
        config = {"agent": {"name": "x"}}

        def test_env(self, *, perturbation: RandomizationConfig | None = None) -> LoadedTestEnv:
            sim = FakeSim()
            return LoadedTestEnv(sim=sim, frames=sim)

        def validate(self, sim: object) -> None: ...

        def algorithm(self) -> FakeAlgorithm:
            return FakeAlgorithm()

        @classmethod
        def open(cls, _checkpoint: object) -> FakeRun:
            return cls()

    monkeypatch.setattr("weir.cli.render.Run", FakeRun)
    output = tmp_path / "playback.mp4"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weir-render",
            "--checkpoint",
            str(tmp_path / "checkpoint.zip"),
            "--output",
            str(output),
            "--frame-interval",
            "1",
        ],
    )
    assert render_main() == 0
    assert output.exists()
