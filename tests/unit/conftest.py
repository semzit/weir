"""Shared test fixtures and constants.

These are plain values imported by the unit tests (the tests/unit directory is
on pytest's sys.path, so ``from conftest import ...`` resolves). Keeping them in
one place avoids duplicating the PPO and randomization configs per test file.
"""

from __future__ import annotations

PPO_CONFIG = {
    "net_arch": [64, 64],
    "learning_rate": 3e-4,
    "n_steps": 64,
    "batch_size": 32,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "device": "cpu",
    "n_envs": 1,
}

RANDOMIZATION = {
    "mass_scale": [0.8, 1.2],
    "friction_scale": [0.5, 1.5],
    "damping_scale": [0.5, 1.5],
    "noise_std": 0.0,
    "action_noise_std": 0.0,
    "latency_steps": 0,
    "perturbation_force": 0.0,
    "perturbation_prob": 0.0,
}

NEUTRAL_RANDOMIZATION = {
    "mass_scale": [1.0, 1.0],
    "friction_scale": [1.0, 1.0],
    "damping_scale": [1.0, 1.0],
    "noise_std": 0.0,
    "action_noise_std": 0.0,
    "latency_steps": 0,
    "perturbation_force": 0.0,
    "perturbation_prob": 0.0,
}
