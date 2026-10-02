from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weir.core.interfaces import Action, Observation


@dataclass(slots=True)
class BalanceTask:
    """CartPole balance: reward every step, terminate when the pole falls.

    The reward and termination semantics are those of Gymnasium's CartPole-v1
    (Barto, Sutton & Anderson's cart-pole problem, via Sutton et al.'s
    pole.c):

    | Gymnasium (Farama Foundation), file
    | ``gymnasium/envs/classic_control/cartpole.py`` (BSD-3-Clause),
    | https://github.com/Farama-Foundation/Gymnasium — itself credited to
    | "Classic cart-pole system implemented by Rich Sutton et al.",
    | http://incompleteideas.net/sutton/book/code/pole.c
    | (permalink: https://perma.cc/C9ZM-652R).

    Cited lines: ``theta_threshold_radians`` and ``x_threshold`` at
    L135-136 of ``CartPoleEnv.__init__``; ``terminated`` at L198-204 and
    the per-step rewards at L206/211/220 of ``step``.

    Semantics: ``+1`` reward on every step (including the termination step)
    while ``|x| <= 2.4`` and ``|theta| <= 12 deg``, terminating otherwise.
    One adaptation: our observation layout is MuJoCo's ``qpos||qvel`` order
    ``[x, theta, x_dot, theta_dot]`` — Gymnasium's state is ``[x, x_dot,
    theta, theta_dot]`` — so ``terminated`` reads ``obs[0]`` and ``obs[1]``.
    """

    x_threshold: float = 2.4
    theta_threshold: float = 12.0 * 2.0 * np.pi / 360.0  # 12 degrees in radians

    def reward(
        self,
        observation: Observation,
        action: Action,
        prev_action: Action | None = None,
        prev_observation: Observation | None = None,
    ) -> float:
        return 1.0

    def terminated(self, observation: Observation) -> bool:
        x, theta = float(observation[0]), float(observation[1])
        return bool(abs(x) > self.x_threshold or abs(theta) > self.theta_threshold)
