"""Environment tasks: reward and termination logic, one module per task."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from weir.core.interfaces import Action, Observation
from weir.core.tasks.balance import BalanceTask
from weir.core.tasks.forward_gait import WalkForwardTask
from weir.core.tasks.standing import StandingTask
from weir.core.tasks.survive import SurviveTask

__all__ = [
    "BalanceTask",
    "StandingTask",
    "SurviveTask",
    "TASKS",
    "Task",
    "TaskFactory",
    "WalkForwardTask",
]


class Task(Protocol):
    """Environment task: maps simulator state to reward and termination.

    ``prev_action`` and ``prev_observation`` are from the previous step
    (None on the first step of an episode), enabling action-rate penalties
    and progress-based rewards.
    """

    def reward(
        self,
        observation: Observation,
        action: Action,
        prev_action: Action | None = None,
        prev_observation: Observation | None = None,
    ) -> float: ...
    def terminated(self, observation: Observation) -> bool: ...


TaskFactory = Callable[..., Task]

TASKS: dict[str, TaskFactory] = {
    "survive": SurviveTask,
    "standing": StandingTask,
    "balance": BalanceTask,
    "walk_forward": WalkForwardTask,
}
