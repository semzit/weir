"""Pydantic validation for the config dicts that cross plugin boundaries.

Each config surface is validated as soon as it enters the code: required
keys are enforced by the schema, so a missing or mistyped key fails loudly
instead of being silently defaulted.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field


class TaskConfig(BaseModel):
    """The task section nested inside the sim config passed to ``SimBackend.load``."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class SimConfig(BaseModel):
    """Validated sim config for ``SimBackend.load``.

    ``extra="allow"`` tolerates the section's plugin/robust/randomization keys,
    which are consumed by the launcher rather than the backend.
    """

    model_config = ConfigDict(extra="allow")

    time_limit: float
    initial_noise: float
    dt: float | None = None
    task: TaskConfig


class RandomizationConfig(BaseModel):
    """Validated hardening knobs for ``RandomizedSim``; every knob is required."""

    mass_scale: tuple[float, float]
    friction_scale: tuple[float, float]
    damping_scale: tuple[float, float]
    noise_std: float
    action_noise_std: float
    latency_steps: int
    perturbation_force: float
    perturbation_prob: float
    perturbation_body: int

    @classmethod
    def neutral(
        cls,
        *,
        force: float = 0.0,
        prob: float = 0.0,
        body: int = 1,
    ) -> Self:
        """A no-op hardening config with only the perturbation knobs optionally enabled."""
        return cls(
            mass_scale=(1.0, 1.0),
            friction_scale=(1.0, 1.0),
            damping_scale=(1.0, 1.0),
            noise_std=0.0,
            action_noise_std=0.0,
            latency_steps=0,
            perturbation_force=force,
            perturbation_prob=prob,
            perturbation_body=body,
        )


class AlgorithmConfig(BaseModel):
    """Validated algorithm hyperparameters for ``AlgorithmPlugin.configure``.

    ``extra="allow"`` tolerates the section's ``plugin`` key, which selects the
    algorithm implementation rather than tuning it.
    """

    model_config = ConfigDict(extra="allow")

    net_arch: list[int]
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    device: str
    n_envs: int
    seed: int | None = None
    checkpoint: str | None = None
    checkpoint_freq: int | None = None
