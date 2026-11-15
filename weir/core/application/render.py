from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio

from weir.core.configs import RandomizationConfig
from weir.core.interfaces import AlgorithmPlugin, FrameSource, SimBackend
from weir.core.rollout import run_episode
from weir.core.run import Run


def run_render(
    checkpoint: Path,
    output_path: Path,
    *,
    frames_to_capture: int | None = None,
    frame_interval: int | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    seed: int = 0,
    perturbation: RandomizationConfig | None = None,
) -> Path:
    """Render a checkpoint's rollout using its manifest environment."""
    run = Run.open(checkpoint)
    env = run.test_env(perturbation=perturbation)
    try:
        if env.frames is None:
            raise ValueError(
                "render requires a simulator with render_frame; "
                "the manifest's backend does not expose it"
            )
        run.validate(env.sim)
        return render_episode(
            env.sim,
            env.frames,
            run.algorithm(),
            output_path,
            frames_to_capture=frames_to_capture,
            frame_interval=frame_interval,
            width=width,
            height=height,
            fps=fps,
            seed=seed,
        )
    finally:
        env.sim.close()


def render_episode(
    sim: SimBackend,
    frames: FrameSource,
    algo: AlgorithmPlugin,
    output_path: Path,
    *,
    frames_to_capture: int | None = None,
    frame_interval: int | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    seed: int = 0,
) -> Path:
    """Roll out a policy and write the captured frames to an mp4 at output_path.

    By default one frame is captured per ``1 / (dt * fps)`` simulation steps,
    so the video plays at roughly realtime speed regardless of the sim
    timestep. Pass ``frame_interval=1`` to capture every step; pass
    ``frames_to_capture=None`` to keep rolling until the episode ends.

    ``sim`` may be hardened (e.g. a ``RandomizedSim``); ``frames`` is the
    underlying simulator that actually renders the camera.
    """
    interval = (
        frame_interval if frame_interval is not None else max(1, round(1.0 / (frames.dt * fps)))
    )
    captured: list[Any] = []

    def capture() -> None:
        captured.append(frames.render_frame(width, height))

    def capture_interval(step_index: int) -> None:
        if step_index % interval == 0:
            capture()

    max_steps = (
        sys.maxsize if frames_to_capture is None else max(0, (frames_to_capture - 1) * interval)
    )
    stats = run_episode(
        sim,
        algo,
        seed=seed,
        max_steps=max_steps,
        on_reset=capture,
        on_step=capture_interval,
    )
    if frames_to_capture is not None and len(captured) < frames_to_capture:
        print(
            f"weir-render: episode ended after {stats.length} steps; captured "
            f"{len(captured)} of {frames_to_capture} requested frames",
            file=sys.stderr,
        )
    imageio.mimsave(output_path, captured, fps=fps)
    return output_path
