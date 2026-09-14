from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from torch import nn

from weir.algo.utils import DeterministicPolicy, SpacesOnly
from weir.core.contracts import AlgorithmPlugin, Shape

_REQUIRED_KEYS = (
    "net_arch",
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "clip_range",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "device",
    "n_envs",
)


class PPOAlgorithm(AlgorithmPlugin):
    """PPO via stable-baselines3, exposed behind the AlgorithmPlugin protocol."""

    def configure(
        self,
        observation_shape: Shape,
        action_shape: Shape,
        config: dict[str, Any],
    ) -> None:
        missing = [key for key in _REQUIRED_KEYS if key not in config]
        if missing:
            raise ValueError(f"Missing algorithm config keys: {missing}")
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self._checkpoint_freq: int | None = None
        seed = config.get("seed")
        self._seed = int(seed) if seed is not None else None
        self._n_envs = int(config["n_envs"])
        checkpoint = config.get("checkpoint")
        if checkpoint:
            self._model = PPO.load(str(checkpoint))
            if self._n_envs != 1:
                self._model.n_envs = self._n_envs
            return
        self._checkpoint_freq = config.get("checkpoint_freq")
        net_arch = list(config["net_arch"])
        self._model = PPO(
            "MlpPolicy",
            SpacesOnly(observation_shape, action_shape),
            policy_kwargs={"net_arch": net_arch},
            device=str(config["device"]),
            learning_rate=float(config["learning_rate"]),
            n_steps=int(config["n_steps"]),
            batch_size=int(config["batch_size"]),
            n_epochs=int(config["n_epochs"]),
            gamma=float(config["gamma"]),
            gae_lambda=float(config["gae_lambda"]),
            clip_range=float(config["clip_range"]),
            ent_coef=float(config["ent_coef"]),
            vf_coef=float(config["vf_coef"]),
            max_grad_norm=float(config["max_grad_norm"]),
            seed=self._seed,
        )
        self._model.n_envs = self._n_envs

    def learn(
        self,
        env: Env,
        total_steps: int,
        callback: Any | None = None,
    ) -> dict[str, float]:
        self._require_model()
        self._model.set_env(env)
        if self._seed is not None:
            self._model.set_random_seed(self._seed)
        if self._n_envs > 1:
            # SB3 2.9 allocates the rollout buffer at construction (n_envs=1);
            # recreate it for the vectorized environment.
            model = self._model
            buffer_class = model.rollout_buffer_class or RolloutBuffer
            model.rollout_buffer = buffer_class(
                model.n_steps,
                cast(Any, model.observation_space),
                cast(Any, model.action_space),
                device=model.device,
                gamma=model.gamma,
                gae_lambda=model.gae_lambda,
                n_envs=env.num_envs,
            )
        callbacks: list[Any] = []
        if self._checkpoint_freq:
            callbacks.append(
                CheckpointCallback(
                    save_freq=int(self._checkpoint_freq),
                    save_path=str(Path.cwd()),
                    name_prefix="checkpoint",
                )
            )
        if callback is not None:
            callbacks.append(callback)
        self._model.learn(
            total_timesteps=total_steps,
            progress_bar=False,
            callback=callbacks or None,
        )
        return {"total_steps": float(total_steps)}

    def act(self, observations: Any, deterministic: bool = False) -> Any:
        self._require_model()
        action, _ = self._model.predict(
            np.asarray(observations, dtype=np.float32), deterministic=deterministic
        )
        return action

    def save(self, path: Path) -> None:
        self._require_model()
        self._model.save(str(path))

    def load(self, path: Path) -> None:
        self._model = PPO.load(str(path))

    def export_policy(self) -> nn.Module:
        self._require_model()
        return DeterministicPolicy(self._model.policy)

    def _require_model(self) -> None:
        if getattr(self, "_model", None) is None:
            raise RuntimeError("PPOAlgorithm.configure() must be called before use")
