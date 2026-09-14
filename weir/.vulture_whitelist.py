# Vulture whitelist: names that are used structurally — gymnasium/API contracts,
# SB3 policy internals, MuJoCo struct fields — but never "called" by our code.
# Remove an entry only if the code it refers to is actually gone.

observation_space  # gymnasium Env interface (SpacesOnly, GymEnv)
action_space       # gymnasium Env interface (SpacesOnly, GymEnv)
metadata           # gymnasium Env class attribute
options            # gymnasium Env.reset(seed, options) API parameter
forward            # torch.nn.Module.forward override

# mujoco.MjvCamera / MjvLight struct fields set directly
lookat
distance
azimuth
elevation
headlight
intensity
pos
dir
diffuse
specular
ambient

_on_step  # SB3 BaseCallback override
rollout_buffer  # SB3 model attribute reallocated for vectorized envs

# Pydantic model fields in weir/core/configs.py — declared on the class,
# consumed by the validation metaclass rather than read as variables
model_config
mass_scale
friction_scale
damping_scale
noise_std
action_noise_std
latency_steps
perturbation_prob
