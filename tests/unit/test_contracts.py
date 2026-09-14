from pathlib import Path

import numpy as np
import torch

from weir.algo.ppo import PPOAlgorithm
from weir.core.contracts import AlgorithmPlugin, DomainRandomizable, Shape, SimBackend
from weir.envs.backends.mujoco import MuJoCoSim

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


def test_implementations_conform_to_protocols() -> None:
    # Both protocols are @runtime_checkable, so isinstance performs the
    # structural conformance check at runtime.
    assert isinstance(MuJoCoSim(), SimBackend)
    assert isinstance(MuJoCoSim(), DomainRandomizable)
    assert isinstance(PPOAlgorithm(), AlgorithmPlugin)


def test_shape_defaults() -> None:
    shape = Shape(dims=(4,), dtype="float32")
    assert shape.low is None
    assert shape.high is None


def test_ppo_algorithm_acts_in_shape() -> None:
    algo = PPOAlgorithm()
    algo.configure(
        Shape(dims=(4,), dtype="float32"),
        Shape(dims=(1,), dtype="float32", low=np.array([-1.0]), high=np.array([1.0])),
        PPO_CONFIG,
    )
    action = algo.act(np.zeros(4, dtype=np.float32))
    assert action.shape == (1,)
    assert action.dtype == np.float32


def test_ppo_algorithm_save_load_roundtrip(tmp_path: Path) -> None:
    algo = PPOAlgorithm()
    algo.configure(
        Shape(dims=(4,), dtype="float32"),
        Shape(dims=(1,), dtype="float32", low=np.array([-1.0]), high=np.array([1.0])),
        PPO_CONFIG,
    )
    path = tmp_path / "model.zip"
    algo.save(path)
    assert path.exists()

    loaded = PPOAlgorithm()
    loaded.load(path)
    observation = np.zeros(4, dtype=np.float32)
    assert np.allclose(
        algo.act(observation, deterministic=True),
        loaded.act(observation, deterministic=True),
    )


def test_ppo_algorithm_export_policy(tmp_path: Path) -> None:
    algo = PPOAlgorithm()
    algo.configure(
        Shape(dims=(4,), dtype="float32"),
        Shape(dims=(1,), dtype="float32", low=np.array([-1.0]), high=np.array([1.0])),
        PPO_CONFIG,
    )
    policy = algo.export_policy()
    with torch.no_grad():
        output = policy(torch.zeros(2, 4))
    assert output.shape == (2, 1)

    onnx_path = tmp_path / "policy.onnx"
    torch.onnx.export(policy, (torch.zeros(1, 4),), onnx_path, dynamo=False)
    assert onnx_path.exists()
