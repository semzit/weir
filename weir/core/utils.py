from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from weir.algo.ppo import PPOAlgorithm
from weir.core.interfaces import AlgorithmPlugin, SimBackend
from weir.envs.backends.mujoco import MuJoCoSim

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"

SIMS: dict[str, type[SimBackend]] = {
    "mujoco": MuJoCoSim,
}

ALGORITHMS: dict[str, type[AlgorithmPlugin]] = {
    "ppo": PPOAlgorithm,
}


def create_sim(name: str) -> SimBackend:
    try:
        return SIMS[name]()
    except KeyError as error:
        raise ValueError(f"Unknown sim backend: {name!r}") from error


def create_algorithm(name: str) -> AlgorithmPlugin:
    try:
        return ALGORITHMS[name]()
    except KeyError as error:
        raise ValueError(f"Unknown algorithm: {name!r}") from error


def resolve_model_path(path: str) -> str:
    """Resolve a config model path to an absolute path against the repo root."""
    model_path = Path(path)
    if model_path.is_absolute():
        return str(model_path)
    return str((ROOT / model_path).resolve())


def config_to_dict(section: Any, *path_keys: str) -> dict[str, Any]:
    """Coerce an OmegaConf section into a resolved plain dict.

    Values under *path_keys are resolved to absolute paths via
    resolve_model_path; missing or empty (e.g. null) values are left as-is.
    """
    resolved = cast(dict[str, Any], OmegaConf.to_container(section, resolve=True) or {})
    for key in path_keys:
        if resolved.get(key):
            resolved[key] = resolve_model_path(str(resolved[key]))
    return resolved


_RESERVED = {
    "name",
    "msg",
    "message",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
}


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Log an INFO event with structured fields, avoiding reserved LogRecord keys."""
    extra = {key: value for key, value in fields.items() if key not in _RESERVED}
    logger.info(event, extra=extra)
