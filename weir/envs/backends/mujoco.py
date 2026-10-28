from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from weir.core.configs import SimConfig
from weir.core.interfaces import Action, Observation, Shape, SimBackend, SimStep
from weir.core.tasks import TASKS, Task
from weir.envs.utils import resolve_model_asset


def _apply_actuator_gains(model: mujoco.MjModel, gains: dict[str, Any]) -> None:
    """Override position-actuator gains by matching actuator-name suffixes."""
    if not isinstance(gains, dict):
        raise ValueError("actuator_gains must be a mapping")
    for suffix, raw in gains.items():
        if not isinstance(raw, dict):
            raise ValueError(f"actuator_gains[{suffix!r}] must be a mapping")
        try:
            kp = float(raw["kp"])
            kv = float(raw["kv"])
        except KeyError as error:
            raise ValueError(f"actuator_gains[{suffix!r}] requires kp and kv") from error
        matched = 0
        for index in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            if name is None or not name.endswith(str(suffix)):
                continue
            model.actuator_gainprm[index, 0] = kp
            model.actuator_biasprm[index, 1] = -kp
            model.actuator_biasprm[index, 2] = -kv
            matched += 1
        if matched == 0:
            raise ValueError(f"actuator_gains suffix {suffix!r} matched no actuators")


def _perturb_joint_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    noise: float,
    rng: np.random.Generator,
) -> None:
    """Add noise to non-root joint coordinates; normalize perturbed ball quaternions."""
    for joint in range(model.njnt):
        kind = model.jnt_type[joint]
        if kind == mujoco.mjtJoint.mjJNT_FREE:
            continue
        address = int(model.jnt_qposadr[joint])
        if kind == mujoco.mjtJoint.mjJNT_BALL:
            quat = data.qpos[address : address + 4] + rng.normal(0.0, noise, 4)
            data.qpos[address : address + 4] = quat / np.linalg.norm(quat)
        else:
            data.qpos[address] += rng.normal(0.0, noise)


class MuJoCoSim(SimBackend):
    """MuJoCo-backed SimBackend for single-environment rollouts."""

    def __init__(self) -> None:
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._task: Task | None = None
        self._time_limit = float("inf")
        self._renderer: mujoco.Renderer | None = None
        self._last_action: Action | None = None
        self._last_observation: Observation | None = None
        self._initial_noise = 0.0
        self._action_repeat = 1
        self._action_scale: float | None = None
        self._action_offset: np.ndarray | None = None
        self._keyframe = -1

    def load(self, agent_config: dict[str, Any], sim_config: dict[str, Any]) -> None:
        cfg = SimConfig.model_validate(sim_config)
        model_path = str(agent_config["model"])
        if not Path(model_path).is_absolute():
            model_path = resolve_model_asset(model_path)
        model = mujoco.MjModel.from_xml_path(model_path)
        self._model = model
        if cfg.dt is not None:
            model.opt.timestep = cfg.dt
        self._keyframe = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if self._keyframe < 0 and model.nkey > 0:
            self._keyframe = 0
        self._action_scale = None
        self._action_offset = None
        action_scale = agent_config.get("action_scale")
        if action_scale is not None:
            self._action_scale = float(action_scale)
            if self._action_scale <= 0:
                raise ValueError("action_scale must be positive")
            if self._keyframe < 0:
                raise ValueError("action_scale requires a keyframe")
            self._action_offset = model.key_ctrl[self._keyframe].copy()
        self._action_repeat = 1
        control_dt = agent_config.get("control_dt")
        if control_dt is not None:
            repeat = round(float(control_dt) / float(model.opt.timestep))
            if repeat < 1 or not np.isclose(repeat * model.opt.timestep, float(control_dt)):
                raise ValueError("control_dt must be a positive multiple of the physics timestep")
            self._action_repeat = repeat
        actuator_gains = agent_config.get("actuator_gains")
        if actuator_gains is not None:
            _apply_actuator_gains(model, actuator_gains)
        self._data = mujoco.MjData(model)
        task_name = cfg.task.name
        task_params = dict(cfg.task.params)
        try:
            task_type = TASKS[task_name]
        except KeyError as error:
            raise ValueError(f"Unknown task: {task_name!r}") from error
        if "nq" in inspect.signature(task_type).parameters and "nq" not in task_params:
            task_params["nq"] = model.nq
        self._task = task_type(**task_params)
        self._time_limit = cfg.time_limit
        self._initial_noise = cfg.initial_noise

    def reset(self, seed: int | None = None) -> Observation:
        model = self._require_model()
        data = self._require_data()
        self._last_action = None
        self._last_observation = None
        if self._keyframe >= 0:
            mujoco.mj_resetDataKeyframe(model, data, self._keyframe)
        else:
            mujoco.mj_resetData(model, data)
        if self._initial_noise > 0:
            _perturb_joint_qpos(model, data, self._initial_noise, np.random.default_rng(seed))
        mujoco.mj_forward(model, data)
        return self._observe(data)

    def step(self, actions: Action) -> SimStep:
        model = self._require_model()
        data = self._require_data()
        task = self._require_task()
        action = np.asarray(actions, dtype=float)
        if self._action_scale is not None:
            assert self._action_offset is not None
            action = np.clip(action, -1.0, 1.0)
            data.ctrl[:] = self._action_offset + self._action_scale * action
        else:
            data.ctrl[:] = action
        for _ in range(self._action_repeat):
            mujoco.mj_step(model, data)
        observation = self._observe(data)
        action_f32 = action.astype(np.float32)
        step = SimStep(
            observation=observation,
            reward=float(
                task.reward(observation, action_f32, self._last_action, self._last_observation)
            ),
            terminated=bool(task.terminated(observation)),
            truncated=bool(data.time >= self._time_limit),
        )
        self._last_action = action_f32
        self._last_observation = observation.copy()
        return step

    def progress(self, prev_observation: Observation | None, observation: Observation) -> float:
        """Delegate to the task's progress definition (0 for tasks without one)."""
        return self._require_task().progress(prev_observation, observation)

    def observation_shape(self) -> Shape:
        model = self._require_model()
        return Shape(dims=(model.nq + model.nv,), dtype="float32")

    def action_shape(self) -> Shape:
        model = self._require_model()
        if self._action_scale is not None:
            return Shape(
                dims=(model.nu,),
                dtype="float32",
                low=np.full(model.nu, -1.0, dtype=np.float32),
                high=np.full(model.nu, 1.0, dtype=np.float32),
            )
        ctrlrange = model.actuator_ctrlrange
        low = ctrlrange[:, 0].astype(np.float32)
        high = ctrlrange[:, 1].astype(np.float32)
        if np.all(low == high):
            raise ValueError(
                "Every actuator has a degenerate ctrlrange (low == high); "
                "set an explicit range/ctrlrange on each actuator"
            )
        return Shape(
            dims=(model.nu,),
            dtype="float32",
            low=low,
            high=high,
        )

    @property
    def dt(self) -> float:
        """The control timestep between actions in seconds."""
        return float(self._require_model().opt.timestep) * self._action_repeat

    def render_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Render the current state with MuJoCo's offscreen renderer (RGB uint8).

        Follows the root body with the camera and injects a directional light
        for models that define none (MuJoCo's default scene light is a
        zero-intensity headlight, which renders everything as dark silhouettes).
        """
        model = self._require_model()
        data = self._require_data()
        if (
            self._renderer is None
            or self._renderer.width != width
            or self._renderer.height != height
        ):
            if self._renderer is not None:
                self._renderer.close()
            try:
                self._renderer = mujoco.Renderer(model, height=height, width=width)
            except Exception as error:
                self._renderer = None
                raise RuntimeError(
                    "Offscreen rendering is unavailable in this environment: could not "
                    f"create a mujoco.Renderer ({type(error).__name__}: {error})"
                ) from error
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat = data.body(1).xpos.copy()  # root body
        # Frame the whole model: distance scales with the model's bounding
        # radius (e.g. the cartpole's pole tip must stay in view).
        camera.distance = 3.5 * model.stat.extent
        camera.azimuth = 90
        camera.elevation = -20
        mujoco.mj_forward(model, data)
        self._renderer.update_scene(data, camera=camera)
        if model.nlight == 0:
            light = self._renderer.scene.lights[0]
            light.type = 1  # directional
            light.headlight = 0
            light.intensity = 1.0
            light.pos = np.array([0.0, 0.0, 6.0])
            light.dir = np.array([0.0, 0.0, -1.0])
            light.diffuse = np.array([1.0, 1.0, 1.0])
            light.specular = np.array([0.5, 0.5, 0.5])
            light.ambient = np.array([0.3, 0.3, 0.3])
        return self._renderer.render()[..., :3].copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._data = None
        self._model = None

    def domain_params(self) -> dict[str, Any]:
        model = self._require_model()
        return {
            "body_mass": model.body_mass.copy(),
            "geom_friction": model.geom_friction.copy(),
            "dof_damping": model.dof_damping.copy(),
        }

    def apply_domain_params(self, params: dict[str, Any]) -> None:
        model = self._require_model()
        data = self._require_data()
        if "body_mass" in params:
            model.body_mass[:] = params["body_mass"]
        if "geom_friction" in params:
            model.geom_friction[:] = params["geom_friction"]
        if "dof_damping" in params:
            model.dof_damping[:] = params["dof_damping"]
        mujoco.mj_forward(model, data)

    def apply_perturbation(self, force: Action, body: int = 1) -> None:
        data = self._require_data()
        # xfrc_applied row 0 is the world body; 1 is the root/movable body
        data.xfrc_applied[body, :3] = np.asarray(force, dtype=float)[:3]
        data.xfrc_applied[body, 3:] = 0.0

    def _observe(self, data: mujoco.MjData) -> Observation:
        return np.concatenate([data.qpos, data.qvel]).astype(np.float32)

    def _require_model(self) -> mujoco.MjModel:
        if self._model is None:
            raise RuntimeError("MuJoCoSim.load() must be called before use")
        return self._model

    def _require_data(self) -> mujoco.MjData:
        if self._data is None:
            raise RuntimeError("MuJoCoSim.load() must be called before use")
        return self._data

    def _require_task(self) -> Task:
        if self._task is None:
            raise RuntimeError("MuJoCoSim.load() must be called before use")
        return self._task
