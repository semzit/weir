from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from weir.cli.utils import load_config_section
from weir.core.interfaces import AlgorithmPlugin, DomainRandomizable
from weir.core.run import MANIFEST_NAME, Run
from weir.core.utils import create_algorithm, resolve_model_path
from weir.envs.backends.mujoco import MuJoCoSim


def render_episode(
    sim: MuJoCoSim,
    algo: AlgorithmPlugin,
    output_path: Path,
    *,
    frames_to_capture: int | None = None,
    frame_interval: int | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    seed: int = 0,
    perturb_force: float = 0.0,
    perturb_prob: float = 0.0,
    perturb_body: int = 1,
) -> Path:
    """Roll out a policy and write the frames to an mp4 file at output_path.

    By default one frame is captured per ``1 / (dt * fps)`` simulation steps,
    so the video plays at roughly realtime speed regardless of the sim
    timestep. Pass ``frame_interval=1`` to capture every step.

    When ``perturb_force > 0``, random perturbation pushes (magnitude drawn
    from ``N(0, perturb_force)``) are applied to the given body with
    probability ``perturb_prob`` each step, showing the policy recovering
    from disturbances. Requires a ``DomainRandomizable`` sim (MuJoCoSim).
    """
    if frame_interval is None:
        frame_interval = max(1, round(1.0 / (sim.dt * fps)))
    rng = np.random.default_rng(seed + 1)
    frames: list[Any] = []
    observation = sim.reset(seed=seed)
    frames.append(sim.render_frame(width, height))
    steps = 0
    while frames_to_capture is None or len(frames) < frames_to_capture:
        if perturb_force > 0 and perturb_prob > 0 and rng.random() < perturb_prob:
            if isinstance(sim, DomainRandomizable):
                sim.apply_perturbation(rng.normal(0.0, perturb_force, size=3), body=perturb_body)
                perturbed = True
            else:
                perturbed = False
        else:
            perturbed = False
        action = algo.act(observation, deterministic=True)
        result = sim.step(action)
        if perturbed:
            sim.apply_perturbation(np.zeros(3), body=perturb_body)  # one-step push, then release
        observation = result.observation
        steps += 1
        if steps % frame_interval == 0:
            frames.append(sim.render_frame(width, height))
        if result.terminated or result.truncated:
            break
    if frames_to_capture is not None and len(frames) < frames_to_capture:
        print(
            f"weir-render: episode ended after {steps} steps; captured "
            f"{len(frames)} of {frames_to_capture} requested frames",
            file=sys.stderr,
        )
    if not frames:
        raise RuntimeError(
            "no frames were captured; the episode ended before the first frame interval "
            f"({frame_interval} steps). Try --frame-interval 1"
        )
    imageio.mimsave(output_path, frames, fps=fps)
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
        description="Render a MuJoCo rollout to an mp4 video.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model XML path for demo renders (no --checkpoint).",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task name for demo renders (no --checkpoint).",
    )
    parser.add_argument(
        "--task-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Task parameter for demo renders, repeatable (no --checkpoint).",
    )
    parser.add_argument("--output", default="video.mp4", help="Output file path.")
    _add_video_args(parser)
    parser.add_argument(
        "--perturb-force",
        type=float,
        default=0.0,
        help="Random push magnitude; show the policy recovering from disturbances.",
    )
    parser.add_argument(
        "--perturb-prob",
        type=float,
        default=0.05,
        help="Per-step probability of a perturbation push (only with --perturb-force).",
    )
    parser.add_argument(
        "--perturb-body",
        type=int,
        default=1,
        help="Body index to push (1=root; for cartpole 2 is the pole).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trained checkpoint to play back; renders the run's own environment.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the run.")
    return parser


def _parse_task_params(entries: list[str]) -> dict[str, object]:
    params: dict[str, object] = {}
    for entry in entries:
        key, sep, value = entry.partition("=")
        if not key or not sep:
            raise ValueError(f"Invalid task param (expected KEY=VALUE): {entry!r}")
        try:
            params[key] = float(value) if "." in value else int(value)
        except ValueError:
            params[key] = value
    return params


_DEMO_FLAGS = (
    ("--model", "model"),
    ("--task", "task"),
    ("--task-param", "task_param"),
)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.checkpoint:
        return _render_demo(args)

    run = Run.open(args.checkpoint)
    if run.config is None:
        print(
            f"weir-render: no manifest ({MANIFEST_NAME}) next to {args.checkpoint}: "
            "this checkpoint predates run manifests; re-train to create one",
            file=sys.stderr,
        )
        return 1

    given = [flag for flag, attr in _DEMO_FLAGS if getattr(args, attr)]
    if given:
        print(
            f"weir-render: {' and '.join(given)} cannot be combined with --checkpoint: "
            "rendering plays back the checkpoint's own environment",
            file=sys.stderr,
        )
        return 1

    sim = None
    try:
        sim = run.sim()
        run.validate(sim)
        if not isinstance(sim, MuJoCoSim):
            raise ValueError(
                f"Render requires a MuJoCoSim, but the manifest uses {type(sim).__name__}"
            )
        algo = run.algorithm()
    except (RuntimeError, TypeError, ValueError) as error:
        if sim is not None:
            sim.close()
        print(f"weir-render: {error}", file=sys.stderr)
        return 1
    return _play(sim, algo, args)


def _render_demo(args: argparse.Namespace) -> int:
    sim = MuJoCoSim()
    try:
        if not args.model:
            raise ValueError("--model is required when rendering without a checkpoint")
        if not args.task:
            raise ValueError("--task is required when rendering without a checkpoint")
        sim.load(
            {"name": "render", "model": resolve_model_path(args.model)},
            {
                "task": {
                    "name": args.task,
                    "params": _parse_task_params(args.task_param),
                },
                "time_limit": 5.0,
                "initial_noise": 0.0,
            },
        )
        algo = create_algorithm("ppo")
        if args.checkpoint:
            algo.load(args.checkpoint)
        else:
            algo.configure(
                sim.observation_shape(),
                sim.action_shape(),
                load_config_section("algo/ppo", "checkpoint"),
            )
    except (RuntimeError, TypeError, ValueError) as error:
        sim.close()
        print(f"weir-render: {error}", file=sys.stderr)
        return 1
    return _play(sim, algo, args)


def _play(sim: MuJoCoSim, algo: AlgorithmPlugin, args: argparse.Namespace) -> int:
    try:
        output_path = render_episode(
            sim,
            algo,
            Path(args.output),
            frames_to_capture=args.frames,
            frame_interval=args.frame_interval,
            width=args.width,
            height=args.height,
            fps=args.fps,
            seed=args.seed,
            perturb_force=args.perturb_force,
            perturb_prob=args.perturb_prob,
            perturb_body=args.perturb_body,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        print(f"weir-render: {error}", file=sys.stderr)
        return 1
    finally:
        sim.close()
    print(f"Rendered rollout to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
