from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from weir.cli.utils import add_perturbation_args, guarded_main, perturbation_from_args
from weir.core.application.render import render_episode
from weir.core.run import MANIFEST_NAME, Run


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
