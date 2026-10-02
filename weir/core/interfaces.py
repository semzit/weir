from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from gymnasium import Env

if TYPE_CHECKING:
    from torch import nn

Observation = np.ndarray
Action = np.ndarray


@dataclass(frozen=True, slots=True)
class Shape:
    """Dimensions, dtype, and optional per-dimension bounds of an observation or action space."""

    dims: tuple[int, ...]
    dtype: str
    low: np.ndarray | None = None
    high: np.ndarray | None = None


@dataclass(slots=True)
class SimStep:
    """One simulator transition: next observation plus reward and termination flags."""

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool


@runtime_checkable
class SimBackend(Protocol):
    """Simulator boundary: reset/step a physics model and describe its interface."""

    def load(self, agent_config: dict[str, Any], sim_config: dict[str, Any]) -> None: ...
    def reset(self, seed: int | None = None) -> Observation: ...
    def step(self, actions: Action) -> SimStep: ...
    def observation_shape(self) -> Shape: ...
    def action_shape(self) -> Shape: ...
    def close(self) -> None: ...


@runtime_checkable
class AlgorithmPlugin(Protocol):
    """Learning and policy boundary.

    Each implementation owns its training loop via ``learn`` (an SB3-backed
    implementation calls ``model.learn``; a hand-written one would collect its
    own rollouts). ``act`` is used for evaluation and recording.
    """

    def configure(
        self,
        observation_shape: Shape,
        action_shape: Shape,
        config: dict[str, Any],
    ) -> None: ...
    def learn(
        self,
        env: Env | Any,
        total_steps: int,
        callback: Any | None = None,
    ) -> dict[str, float]: ...
    def act(self, observations: Any, deterministic: bool = False) -> Any: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
    def export_policy(self) -> nn.Module: ...


@runtime_checkable
class DomainRandomizable(Protocol):
    """Optional simulator capability used by the hardening wrapper.

    A sim that exposes its physics domain parameters lets a wrapper randomize
    them per episode, and accepts external wrenches for perturbation pushes.
    """

    def domain_params(self) -> dict[str, Any]: ...
    def apply_domain_params(self, params: dict[str, Any]) -> None: ...
    def apply_perturbation(self, force: Action) -> None: ...
