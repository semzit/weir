from pathlib import Path

import mujoco
import pytest

from weir.envs.utils import MODELS_DIR


def _model_path(name: str) -> Path:
    path = MODELS_DIR / name
    assert path.exists(), f"missing model asset: {path}"
    return path


@pytest.mark.parametrize(
    "xml_path,nu,nq",
    [
        (_model_path("cartpole.xml"), 1, 2),
        (_model_path("menagerie/berkeley_humanoid/berkeley_humanoid.xml"), 12, 19),
    ],
    ids=["cartpole", "berkeley_humanoid"],
)
def test_model_loads_and_steps(xml_path: Path, nu: int, nq: int) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.time > 0.0
    assert model.nu == nu
    assert model.nq == nq


def test_berkeley_humanoid_uses_position_actuators() -> None:
    model = mujoco.MjModel.from_xml_path(
        str(_model_path("menagerie/berkeley_humanoid/berkeley_humanoid.xml"))
    )
    for i in range(model.nu):
        assert model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
