from __future__ import annotations

from dataclasses import dataclass

from weir.core.interfaces import Action, Observation


@dataclass(slots=True)
class StandingTask:
    """Stay upright and on the floor: survive while standing, terminate on falling.

    Assumes the observation begins with the root freejoint: qpos[2] is the
    body height above the floor.
    """

    min_height: float = 0.8
    live_reward: float = 1.0

    def reward(
        self,
        observation: Observation,
        action: Action,
        prev_action: Action | None = None,
        prev_observation: Observation | None = None,
    ) -> float:
        return 0.0 if self.terminated(observation) else self.live_reward

    def terminated(self, observation: Observation) -> bool:
        return float(observation[2]) < self.min_height
