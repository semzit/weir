from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from typing import Any

from weir.core.configs import RandomizationConfig
from weir.core.utils import log_event

logger = logging.getLogger("weir")


def setup_logging() -> None:
    """Configure INFO-level structured logging shared by all entry points."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def add_perturbation_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared disturbance flags used by evaluation and rendering."""
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


def perturbation_from_args(args: argparse.Namespace) -> RandomizationConfig | None:
    """Build a perturbation config from the shared flags, or None when disabled."""
    if args.perturb_force <= 0:
        return None
    return RandomizationConfig.neutral(
        force=args.perturb_force,
        prob=args.perturb_prob,
        body=args.perturb_body,
    )


def guarded_main(
    parser: argparse.ArgumentParser,
    runner: Callable[[argparse.Namespace], dict[str, Any]],
    event: str,
    *,
    on_success: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Parse args, run, and log the outcome; return 0 on success, 1 on failure."""
    args = parser.parse_args()
    setup_logging()
    try:
        result = runner(args)
    except Exception as error:
        log_event(logger, f"{event}.failed", error=str(error))
        print(f"{event}: {error}", file=sys.stderr)
        return 1
    log_event(logger, f"{event}.complete", **result)
    if on_success:
        on_success(result)
    return 0
