from __future__ import annotations

import numpy as np
import torch
from gymnasium import Env
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from weir.core.contracts import Shape
from weir.envs.utils import shape_to_box


def sample_action(
    action_shape: Shape,
    rng: np.random.Generator,
    deterministic: bool = False,
) -> np.ndarray:
    """Sample an in-bounds action, or its midpoint when deterministic."""
    dims = tuple(action_shape.dims)
    low = action_shape.low
    high = action_shape.high
    if low is None or high is None:
        raise ValueError(
            "Action shape has no bounds; cannot sample an action. "
            "The sim backend must report per-dimension low/high."
        )
    if deterministic:
        return ((low + high) / 2.0).astype(np.float32)
    return rng.uniform(low, high, size=dims).astype(np.float32)


class SpacesOnly(Env):
    """Minimal env exposing only the spaces SB3 needs to build its policy."""

    def __init__(self, observation_shape: Shape, action_shape: Shape) -> None:
        self.observation_space = shape_to_box(observation_shape)
        self.action_space = shape_to_box(action_shape)


class DeterministicPolicy(nn.Module):
    """Inference-only wrapper: forward maps observations to deterministic mean actions."""

    def __init__(self, policy: ActorCriticPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        device = next(self.policy.parameters()).device
        actions, _, _ = self.policy(observations.to(device), deterministic=True)
        return actions
