"""Run artifacts: self-describing training runs (checkpoint plus manifest).

A manifest (``checkpoint.meta.json``) is written next to the checkpoint by the
training entry point, recording the resolved configuration and the
observation/action shapes the policy was trained with. Evaluation, rendering,
and export rebuild the environment from the manifest instead of re-specifying
flags, so a checkpoint can no longer be paired with the wrong model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from weir.core.contracts import AlgorithmPlugin, SimBackend
from weir.core.utils import create_algorithm, create_sim
from weir.envs.wrappers.randomized import RandomizedSim

MANIFEST_NAME = "checkpoint.meta.json"


class Run:
    """A training run: a checkpoint plus its manifest, if present."""

    def __init__(self, checkpoint: Path, config: dict[str, Any] | None) -> None:
        self._checkpoint = Path(checkpoint)
        self._config = config

    @staticmethod
    def build_sim(sim_config: dict[str, Any]) -> SimBackend:
        """Construct the simulator named in config, wrapping it when hardened."""
        if "robust" not in sim_config:
            raise ValueError("sim config is missing the 'robust' key")
        sim = create_sim(str(sim_config["plugin"]))
        if sim_config["robust"]:
            randomization = sim_config.get("randomization")
            if randomization is None:
                raise ValueError(
                    "sim config is missing the 'randomization' block (required when robust)"
                )
            sim = RandomizedSim(sim, randomization)
        return sim

    @classmethod
    def open(cls, checkpoint: Path | str) -> Run:
        """Open a checkpoint, loading the manifest that sits beside it (if any)."""
        checkpoint = Path(checkpoint)
        manifest = checkpoint.with_suffix(".meta.json")
        if manifest.is_file():
            config = json.loads(manifest.read_text(encoding="utf-8"))
            return cls(checkpoint, config)
        return cls(checkpoint, None)

    @property
    def config(self) -> dict[str, Any] | None:
        """The resolved training config from the manifest, or None."""
        return self._config

    def algorithm(self) -> AlgorithmPlugin:
        """Create the training algorithm plugin and load the checkpoint weights."""
        plugin = self.plugin("algo")
        if plugin is None:
            raise ValueError(
                f"No 'algo.plugin' in manifest ({MANIFEST_NAME}) next to {self._checkpoint}"
            )
        algorithm = create_algorithm(plugin)
        algorithm.load(self._checkpoint)
        return algorithm

    def sim(self) -> SimBackend:
        """Build the simulator the run was trained with (hardening included)."""
        config = self._require_config()
        sim_config = dict(config["sim"])
        agent_config = dict(config["agent"])
        task_config = dict(config["task"])
        sim = self.build_sim(sim_config)
        sim.load(agent_config, {**sim_config, "task": task_config})
        return sim

    def obs_dim(self) -> int | None:
        """The flat observation dimension the policy was trained with."""
        if self._config is None:
            return None
        return int(self._config["observation_shape"]["dims"][0])

    def plugin(self, section: str) -> str | None:
        """The plugin name recorded in a config section (e.g. 'algo')."""
        if self._config is None:
            return None
        return str(self._config.get(section, {}).get("plugin", ""))

    def validate(self, sim: SimBackend) -> None:
        """Raise if the simulator's shapes disagree with the manifest."""
        if self._config is None:
            return
        expected_obs = tuple(self._config["observation_shape"]["dims"])
        expected_act = tuple(self._config["action_shape"]["dims"])
        obs = sim.observation_shape().dims
        act = sim.action_shape().dims
        if obs != expected_obs:
            raise ValueError(
                f"Checkpoint was trained with observation shape {expected_obs} "
                f"but this simulator produces {obs}"
            )
        if act != expected_act:
            raise ValueError(
                f"Checkpoint was trained with action shape {expected_act} "
                f"but this simulator produces {act}"
            )

    def _require_config(self) -> dict[str, Any]:
        if self._config is None:
            raise ValueError(
                f"No manifest ({MANIFEST_NAME}) next to {self._checkpoint}; "
                "pass the configuration explicitly instead."
            )
        return self._config
