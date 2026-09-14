from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from weir.core.contracts import Action, DomainRandomizable, Observation, Shape, SimBackend, SimStep

_FIELD_MAP = {
    "mass_scale": "body_mass",
    "friction_scale": "geom_friction",
    "damping_scale": "dof_damping",
}

_REQUIRED_KEYS = (
    "mass_scale",
    "friction_scale",
    "damping_scale",
    "noise_std",
    "action_noise_std",
    "latency_steps",
    "perturbation_force",
    "perturbation_prob",
)


class RandomizedSim(SimBackend):
    """SimBackend decorator that hardens any simulator for sim-to-real transfer.

    Wraps an inner SimBackend and layers on, per episode: domain randomization
    (via the optional DomainRandomizable capability), observation/action noise,
    action latency, and random perturbation pushes. The randomization config
    must declare every knob explicitly; nothing is silently defaulted.
    """

    def __init__(self, inner: SimBackend, config: dict[str, Any]) -> None:
        missing = [key for key in _REQUIRED_KEYS if key not in config]
        if missing:
            raise ValueError(f"Randomization config missing keys: {missing}")
        self._inner = inner
        self._config = config
        self._rng = np.random.default_rng(0)
        self._latency: deque[np.ndarray] = deque()

    def load(self, agent_config: dict[str, Any], sim_config: dict[str, Any]) -> None:
        self._inner.load(agent_config, sim_config)

    def reset(self, seed: int | None = None) -> Observation:
        self._rng = np.random.default_rng(seed)
        if isinstance(self._inner, DomainRandomizable):
            self._randomize_domain()
        self._latency.clear()
        return self._inner.reset(seed=seed)

    def step(self, actions: Action) -> SimStep:
        action = np.asarray(actions, dtype=np.float32)
        action = self._add_noise(action, self._config["action_noise_std"])
        action = self._apply_latency(action)
        inner = self._inner
        perturbation_force = self._config["perturbation_force"]
        if isinstance(inner, DomainRandomizable):
            self._maybe_perturb(inner, perturbation_force)
        result = inner.step(action)
        if isinstance(inner, DomainRandomizable) and perturbation_force > 0:
            inner.apply_perturbation(np.zeros(3))
        observation = self._add_noise(result.observation, self._config["noise_std"])
        return SimStep(
            observation=observation,
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
        )

    def observation_shape(self) -> Shape:
        return self._inner.observation_shape()

    def action_shape(self) -> Shape:
        return self._inner.action_shape()

    def close(self) -> None:
        self._inner.close()

    def _randomize_domain(self) -> None:
        inner = self._inner
        assert isinstance(inner, DomainRandomizable)
        params = inner.domain_params()
        for key, field in _FIELD_MAP.items():
            low, high = self._config[key]
            factor = self._rng.uniform(float(low), float(high))
            params[field] = np.asarray(params[field]) * factor
        inner.apply_domain_params(params)

    def _add_noise(self, values: np.ndarray, std: float) -> np.ndarray:
        if std <= 0:
            return values
        return (
            np.asarray(values, dtype=np.float32) + self._rng.normal(0.0, std, size=values.shape)
        ).astype(np.float32)

    def _apply_latency(self, action: np.ndarray) -> np.ndarray:
        delay = int(self._config["latency_steps"])
        if delay <= 0:
            return action
        self._latency.append(action.copy())
        if len(self._latency) <= delay:
            return np.zeros_like(action)
        return self._latency.popleft()

    def _maybe_perturb(self, inner: DomainRandomizable, force: float) -> None:
        prob = self._config["perturbation_prob"]
        if force <= 0 or self._rng.random() >= prob:
            return
        inner.apply_perturbation(self._rng.normal(0.0, force, size=3))
