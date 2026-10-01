from __future__ import annotations

from dataclasses import dataclass

from weir.core.interfaces import Action, Observation


@dataclass(slots=True)
class SurviveTask:
    """Reward survival for every step; never terminates on its own.

    Note: a constant reward gives PPO no gradient — nothing can be learned
    from this task. It exists for demo renders, not training.
    """

    live_reward: float = 1.0

    def reward(
        self,
        observation: Observation,
        action: Action,
        prev_action: Action | None = None,
        prev_observation: Observation | None = None,
    ) -> float:
        return self.live_reward

    def terminated(self, observation: Observation) -> bool:
        return False
