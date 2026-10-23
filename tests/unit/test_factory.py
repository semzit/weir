import pytest

from weir.core.utils import create_algorithm, create_sim


def test_factories_create_plugins_and_reject_unknown() -> None:
    assert create_sim("mujoco").__class__.__name__ == "MuJoCoSim"
    assert create_algorithm("ppo").__class__.__name__ == "PPOAlgorithm"

    with pytest.raises(ValueError, match="Unknown sim backend"):
        create_sim("isaac_lab")
    with pytest.raises(ValueError, match="Unknown algorithm"):
        create_algorithm("sac")
