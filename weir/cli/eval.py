from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from weir.cli.utils import add_perturbation_args, guarded_main, perturbation_from_args
from weir.core.application.evaluate import run_eval
from weir.core.run import Run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weir-eval",
        description="Evaluate a trained checkpoint with deterministic rollouts.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to the algorithm checkpoint."
    )
    parser.add_argument("--episodes", type=int, default=5, help="Episodes to roll out.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the rollouts.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode.")
    add_perturbation_args(parser)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override of the run's task or sim, repeatable (agent and algo are "
        "fixed by the checkpoint).",
    )
    return parser


def _run_eval_from_args(args: argparse.Namespace) -> dict[str, Any]:
    metrics = run_eval(
        args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        overrides=args.override,
        perturbation=perturbation_from_args(args),
    )
    run = Run.open(args.checkpoint)
    config = run.config or {}
    agent_name = str(config.get("agent", {}).get("name", "unknown"))
    task_name = str(config.get("task", {}).get("name", "unknown"))
    return {
        "metrics": metrics,
        "checkpoint": str(args.checkpoint),
        "agent": agent_name,
        "task": task_name,
    }


def _print_metrics(result: dict[str, Any]) -> None:
    for key, value in result["metrics"].items():
        print(f"{key}: {value:.3f}")


def main() -> int:
    return guarded_main(
        build_parser(),
        _run_eval_from_args,
        "eval",
        on_success=_print_metrics,
    )


if __name__ == "__main__":
    raise SystemExit(main())
