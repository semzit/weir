from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weir.core.interfaces import Action, Observation
from weir.core.tasks.utils import rotate_vector


@dataclass(slots=True)
class WalkForwardTask:
    """Walk forward along world +x while staying upright; terminate on falling.

    The reward blend ports standard legged-locomotion reward terms from Isaac
    Lab (BSD-3-Clause), https://github.com/isaac-sim/IsaacLab:

    - ``alive_reward``: ``is_alive``,
      source/isaaclab/isaaclab/envs/mdp/rewards.py:32
    - ``action_rate_l2`` (squared action delta penalty):
      source/isaaclab/isaaclab/envs/mdp/rewards.py:252
    - uprightness (cosine of the base tilt): ``upright_posture_bonus`` via
      ``obs.base_up_proj``,
      source/isaaclab_tasks/.../manager_based/classic/humanoid/mdp/rewards.py:22
    - heading (base forward projected on the goal direction):
      ``move_to_target_bonus`` via ``obs.base_heading_proj``, same file:30
    - forward velocity (heading-projected forward speed): in the spirit of
      the velocity-tracking terms, e.g. ``track_lin_vel_xy_exp``,
      source/isaaclab/isaaclab/envs/mdp/rewards.py:304
    - ``progress_reward`` (per-step forward displacement, uses
      ``prev_observation``): Isaac Lab's ``progress_reward`` class,
      source/isaaclab_tasks/.../manager_based/classic/humanoid/mdp/rewards.py:35
    - height (root height, keeps the torso up against slow sinks):
      in the spirit of Isaac Lab's ``base_height_l2``,
      source/isaaclab/isaaclab/envs/mdp/rewards.py:101

    Each term is adapted to this project's stateless observation layout —
    no asset/command tensors, just the root freejoint slice: obs[0] is the
    root x position, obs[2] is the body height above the floor, obs[3:7]
    is the root quaternion in (w, x, y, z) order, and obs[nq] is the root
    linear x-velocity (qvel[0]).
    """

    nq: int = 19
    min_height: float = 0.9
    forward_coef: float = 2.0
    heading_coef: float = 1.0
    upright_coef: float = 0.5
    alive_reward: float = 1.0
    action_rate_coef: float = 0.02
    progress_coef: float = 0.0
    height_coef: float = 0.0

    def reward(
        self,
        observation: Observation,
        action: Action,
        prev_action: Action | None = None,
        prev_observation: Observation | None = None,
    ) -> float:
        if self.terminated(observation):
            return 0.0
        forward_vel = float(observation[self.nq])
        heading = float(rotate_vector(observation[3:7], (1.0, 0.0, 0.0))[0])
        uprightness = float(rotate_vector(observation[3:7], (0.0, 0.0, 1.0))[2])
        action_rate = 0.0
        if prev_action is not None:
            delta = np.asarray(action, dtype=np.float32) - np.asarray(prev_action, dtype=np.float32)
            action_rate = self.action_rate_coef * float(np.sum(np.square(delta)))
        progress = 0.0
        if prev_observation is not None:
            progress = self.progress_coef * (float(observation[0]) - float(prev_observation[0]))
        height = self.height_coef * float(observation[2])
        return (
            self.alive_reward
            + self.forward_coef * forward_vel * heading
            + self.heading_coef * heading
            + self.upright_coef * uprightness
            + progress
            + height
            - action_rate
        )

    def terminated(self, observation: Observation) -> bool:
        return float(observation[2]) < self.min_height
