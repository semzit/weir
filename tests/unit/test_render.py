from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from weir.cli.render import main as render_main
from weir.cli.render import render_episode
from weir.core.utils import create_algorithm
from weir.envs.backends.mujoco import MuJoCoSim
from weir.envs.utils import MODELS_DIR

CART_POLE = MODELS_DIR / "cartpole.xml"

PPO_CONFIG = {
    "net_arch": [64, 64],
    "learning_rate": 3e-4,
    "n_steps": 64,
    "batch_size": 32,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "device": "cpu",
    "n_envs": 1,
}


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
    render_episode(
        sim,
        algo,  # type: ignore[arg-type]
        tmp_path / "perturbed.mp4",
        frames_to_capture=4,
        frame_interval=1,
        width=160,
        height=120,
        fps=15,
        seed=0,
        perturb_force=5.0,
        perturb_prob=1.0,  # push every step
        perturb_body=2,
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
    output = tmp_path / "cli.mp4"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weir-render",
            "--model",
            str(CART_POLE),
            "--task",
            "survive",
            "--output",
            str(output),
            "--frames",
            "10",
            "--width",
            "160",
            "--height",
            "120",
            "--fps",
            "15",
        ],
    )
    assert render_main() == 0
    assert output.exists()
    assert output.stat().st_size > 0


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
                    "terminated": True,
                    "truncated": False,
                },
            )
            return step()

        def render_frame(self, width: int = 0, height: int = 0) -> np.ndarray:
            return np.zeros((height, width, 3), dtype=np.uint8)

        def close(self) -> None: ...

    class FakeAlgorithm:
        def act(self, observation: object, deterministic: bool = False) -> np.ndarray:
            return np.zeros(1, dtype=np.float32)

    class FakeRun:
        config = {"agent": {"name": "x"}}

        def sim(self) -> FakeSim:
            return FakeSim()

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


def test_cli_rejects_env_flags_with_manifest_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "checkpoint.meta.json"
    manifest.write_text('{"agent": {"name": "x"}}', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weir-render",
            "--checkpoint",
            str(tmp_path / "checkpoint.zip"),
            "--task-param",
            "x=1",
        ],
    )
    assert render_main() == 1


def test_cli_rejects_malformed_task_param(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["weir-render", "--model", str(CART_POLE), "--task", "survive", "--task-param", "nq"],
    )
    assert render_main() == 1


def test_cli_rejects_unknown_task_param(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["weir-render", "--model", str(CART_POLE), "--task", "survive", "--task-param", "x=1"],
    )
    assert render_main() == 1


def test_cli_requires_model_and_task_for_demo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["weir-render", "--model", str(CART_POLE)])
    assert render_main() == 1

    monkeypatch.setattr(sys, "argv", ["weir-render", "--task", "survive"])
    assert render_main() == 1
