# Configuration guide

Everything about a run lives in YAML under this directory, composed by Hydra at launch.
This guide explains how the system fits together and what every knob does.

## How composition works

`train.yaml` is the entry point. Its `defaults` list picks exactly one file
from each of four config groups, and Hydra merges them into a single config:

```yaml
# train.yaml
defaults:
  - _self_
  - agent: cartpole      # -> agent/cartpole.yaml
  - task: balance        # -> task/balance.yaml
  - sim: mujoco          # -> sim/mujoco.yaml
  - algo: ppo            # -> algo/ppo.yaml
```

```
agent/               # one robot per file: cartpole.yaml, humanoid.yaml
task/                # one objective per file: balance, standing, walk_forward, ...
sim/                 # one backend per file: mujoco.yaml
algo/                # one algorithm per file: ppo.yaml
train.yaml           # composition + train.seed, train.total_steps
```

Each group file carries a `name`/`plugin` key that the factory maps to a concrete
implementation — the training loop never imports anything concrete itself.

## Overriding at the command line

Any composed value can be overridden with `key=value` arguments, without editing files:

```bash
# Swap a whole group (picks a different file from that group)
uv run weir-train agent=humanoid task=walk_forward

# Override a nested value inside a group
uv run weir-train task.params.min_height=1.0

# Override a top-level value
uv run weir-train train.total_steps=500000

# Combine as many as you like
uv run weir-train agent=humanoid task=walk_forward algo.n_steps=4096 train.total_steps=1000000
```

Rules:

- Values are typed: numbers parse as floats, `true`/`false` as booleans, `[64, 64]` as lists.
- Hydra refuses to *invent* keys — adding a brand-new key needs a `+` prefix:
  `+task.params.thing=1`.
- `algo.checkpoint=<path>` resumes training from an existing checkpoint (weights only;
  the rest of the config still applies).

## Reference

### `train.yaml`

| Key | Default | Meaning |
|---|---|---|
| `defaults.agent` | `cartpole` | Which file in `agent/` to use. |
| `defaults.task` | `balance` | Which file in `task/` to use. |
| `defaults.sim` | `mujoco` | Which file in `sim/` to use. |
| `defaults.algo` | `ppo` | Which file in `algo/` to use. |
| `train.seed` | `0` | RNG seed; fixed seeds give reproducible runs. |
| `train.total_steps` | `100000` | Total environment steps the algorithm trains for. |

### `agent/*.yaml` — the robot

| Key | Example | Meaning |
|---|---|---|
| `name` | `berkeley_humanoid` | Label recorded in run manifests and logs. |
| `model` | `weir/models/menagerie/...xml` | Path to the MuJoCo XML asset (repo-relative). |

### `task/*.yaml` — the objective

Each task file's `params` are constructor kwargs of the task class
(`weir/core/tasks/`). Reward semantics and source citations live in the task
docstrings.

**balance** (CartPole-v1 semantics, ported from Gymnasium):

| Param | Default | Meaning |
|---|---|---|
| `x_threshold` | `2.4` | Cart position limit (m); beyond this the episode ends. |
| `theta_threshold` | `0.2094` (12°) | Pole angle limit (rad); beyond this the episode ends. |

**standing**:

| Param | Default | Meaning |
|---|---|---|
| `min_height` | `0.8` | Root body height (`obs[2]`) below which the episode ends. |

**walk_forward** (Isaac Lab reward terms):

| Param | Default | Meaning |
|---|---|---|
| `min_height` | `0.9` | Root body height below which the episode ends. |
| `forward_coef` | `2.0` | Weight on forward speed in the heading direction. |
| `heading_coef` | `1.0` | Weight on facing the goal heading (world +x). |
| `upright_coef` | `0.5` | Weight on the base being vertical. |
| `alive_reward` | `1.0` | Per-step survival bonus. |
| `action_rate_coef` | `0.02` | Penalty on the squared change in action per step. |

**survive**: no params. A constant reward gives PPO no gradient — nothing can be
learned from this task; it exists for demo renders, not training.

### `sim/mujoco.yaml` — the simulator

| Key | Default | Meaning |
|---|---|---|
| `plugin` | `mujoco` | SimBackend implementation in the factory. |
| `time_limit` | `10.0` | Episode truncation (seconds of simulated time). |
| `robust` | `false` | Wrap the sim in the sim-to-real hardening decorator (`RandomizedSim`). |
| `randomization.mass_scale` | `[0.8, 1.2]` | Per-episode uniform factor range for body masses. |
| `randomization.friction_scale` | `[0.5, 1.5]` | Per-episode uniform factor range for geom friction. |
| `randomization.damping_scale` | `[0.5, 1.5]` | Per-episode uniform factor range for joint damping. |
| `randomization.noise_std` | `0.0` | Std of Gaussian noise added to observations. |
| `randomization.action_noise_std` | `0.0` | Std of Gaussian noise added to actions. |
| `randomization.latency_steps` | `0` | Action delay in whole steps (0 = none). |
| `randomization.perturbation_force` | `0.0` | Std of random root-body pushes (0 = none). |
| `randomization.perturbation_prob` | `0.0` | Per-step probability of a push. |

The `randomization` block only takes effect when `robust: true`.

### `algo/ppo.yaml` — the algorithm (PPO via stable-baselines3)

| Key | Default | Meaning |
|---|---|---|
| `plugin` | `ppo` | AlgorithmPlugin implementation in the factory. |
| `checkpoint` | `null` | Checkpoint to resume from (weights only); `null` trains from scratch. |
| `device` | `auto` | `auto` (CUDA when available, else CPU), `cpu`, or `cuda`. |
| `net_arch` | `[64, 64]` | Hidden layer sizes of the MLP policy. |
| `learning_rate` | `3.0e-4` | Adam learning rate. |
| `n_steps` | `2048` | Rollout buffer size (steps per policy update). |
| `batch_size` | `64` | Minibatch size for gradient updates. |
| `n_epochs` | `10` | Passes over the buffer per update. |
| `gamma` | `0.99` | Discount factor. |
| `gae_lambda` | `0.95` | GAE lambda for advantage estimation. |
| `clip_range` | `0.2` | PPO clipping epsilon. |
| `ent_coef` | `0.0` | Entropy bonus weight (exploration; 0 = none). |
| `vf_coef` | `0.5` | Value-function loss weight. |
| `max_grad_norm` | `0.5` | Gradient clipping norm. |

## Example runs

Commands pick a torch build via the `cpu` or `gpu` extra (see the root README's GPU
section); use the extra you synced with:

```bash
# Default: train the cartpole balancer
uv run --extra cpu weir-train

# Fast iteration on the toy: fewer steps, bigger minibatches
uv run --extra cpu weir-train train.total_steps=50000 algo.batch_size=128

# The real goal: humanoid walking, more exploration, on the GPU
uv run --extra gpu weir-train agent=humanoid task=walk_forward algo.ent_coef=0.02 train.total_steps=2000000

# Force the device if auto-detection isn't what you want
uv run --extra cpu weir-train agent=cartpole task=balance algo.device=cpu

# Hardened training (sim-to-real: mass/friction randomization, noise, pushes)
uv run --extra cpu weir-train agent=humanoid task=standing sim.robust=true

# Resume a run from its checkpoint
uv run --extra cpu weir-train agent=humanoid task=walk_forward algo.checkpoint=outputs/2026-08-15/12-00-00/checkpoint.zip
```
