from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio

from weir.cli.utils import add_perturbation_args, guarded_main, perturbation_from_args
from weir.core.interfaces import AlgorithmPlugin, FrameSource, SimBackend
from weir.core.rollout import run_episode
from weir.core.run import MANIFEST_NAME, Run


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


def _add_video_args(parser: argparse.ArgumentParser) -> None:
    """Video-encoding flags: frames, frame interval, size, fps."""
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=None,
        help="Simulation steps per video frame; defaults to realtime pacing (1 / dt / fps).",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weir-render",
        description="Render a trained checkpoint's MuJoCo rollout to an mp4 video.",
    )
    parser.add_argument("--output", default="video.mp4", help="Output file path.")
    _add_video_args(parser)
    add_perturbation_args(parser)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trained checkpoint to play back; renders the run's own environment.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the run.")
    return parser


def _run_render_from_args(args: argparse.Namespace) -> dict[str, Any]:
    run = Run.open(args.checkpoint)
    if run.config is None:
        raise ValueError(
            f"no manifest ({MANIFEST_NAME}) next to {args.checkpoint}: "
            "this checkpoint predates run manifests; re-train to create one"
        )
    env = run.test_env(perturbation=perturbation_from_args(args))
    try:
        if env.frames is None:
            raise ValueError(
                "render requires a simulator with render_frame; "
                "the manifest's backend does not expose it"
            )
        run.validate(env.sim)
        output_path = render_episode(
            env.sim,
            env.frames,
            run.algorithm(),
            Path(args.output),
            frames_to_capture=args.frames,
            frame_interval=args.frame_interval,
            width=args.width,
            height=args.height,
            fps=args.fps,
            seed=args.seed,
        )
    finally:
        env.sim.close()
    return {"output": str(output_path)}


def _print_output(result: dict[str, Any]) -> None:
    print(f"Rendered rollout to {result['output']}")


def main() -> int:
    return guarded_main(
        build_parser(),
        _run_render_from_args,
        "render",
        on_success=_print_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
