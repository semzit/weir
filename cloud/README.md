# Cloud training on AWS Batch

This document explains the opt-in cloud training path. Local training is
unchanged: the cloud worker runs the same `weir-train` command inside a
container, then uploads the run directory to S3.

The short version:

- CI publishes a GPU image to GitHub Container Registry (GHCR).
- OpenTofu creates GPU and CPU AWS Batch queues, IAM roles, a private S3
  artifact bucket, CloudWatch logs, and a monthly EC2 budget.
- `cloud/weir` submits one Batch job per run and handles status, logs, artifact
  download, cancellation, and teardown.
- The container optionally checks for CUDA, trains under `/work/run`, and syncs
  artifacts to S3 every 60 seconds and again on exit.
- `WEIR_COMPUTE=gpu` uses the CUDA queue; `WEIR_COMPUTE=cpu` uses the parallel
  MuJoCo queue.

## Architecture

```text
Developer
  cloud/weir train ...
        |
        | submit-job
        v
  AWS Batch queues: weir-gpu / weir-cpu
        |
        +--> g5.xlarge (CUDA) or c7i.4xlarge (parallel MuJoCo)
               |
               +--> pull ghcr.io/<repo>:<git-sha>
               +--> cloud/entrypoint.sh
                      +--> CUDA check (GPU only)
                      +--> weir-train ... hydra.run.dir=/work/run
                      +--> aws s3 sync /work/run -> s3://.../runs/<run-id>/
               |
               +--> CloudWatch Logs: /aws/batch/weir

  Private S3 artifact bucket
        |
        +--> cloud/weir pull -> outputs/cloud/<run-id>/run/
```

OpenTofu has two layers:

```text
infra/bootstrap/   creates the encrypted, versioned S3 bucket for OpenTofu state
infra/             creates the Batch stack and reads state from that bucket
```

## Components

### `cloud/weir`

A small shell wrapper around AWS CLI, OpenTofu, and `jq`:

| Command | Purpose |
| --- | --- |
| `cloud/weir setup <region>` | Create the state bucket, write `infra/backend.hcl`, initialize the main backend. |
| `cloud/weir deploy` | Apply the Batch stack. Uses the current git SHA as the image tag unless `WEIR_IMAGE_TAG` is set. |
| `cloud/weir train [overrides...]` | Submit a job. Uses `WEIR_COMPUTE=gpu` (default) or `cpu`, prepends the matching `algo.device`, creates a run ID, and sets the S3 artifact URI. |
| `cloud/weir status [job-id]` | List jobs in the queue, or describe one job. |
| `cloud/weir logs [job-id] --follow` | Read the CloudWatch log stream for a job. |
| `cloud/weir pull [run-id\|job-id] [dest]` | Download run artifacts from S3. Defaults to `outputs/cloud/<run-id>/run`. |
| `cloud/weir cancel [run-id\|job-id]` | Terminate a running job. |
| `cloud/weir destroy` | Destroy the Batch stack. The state bucket is intentionally kept. |

Run `cloud/weir help` for the built-in summary.

### `infra/`

The main stack creates:

- A managed Batch compute environment using `g5.xlarge` on-demand instances.
- A Batch job queue named `weir-gpu`.
- A job definition named `weir-train`.
- A private S3 artifact bucket with encryption and public access blocked.
- IAM roles for Batch, the EC2 container instance, job artifact access, and ECS
  image/log execution.
- A CloudWatch log group with 30-day retention.
- An AWS Budget for EC2 spend with email alerts.

The stack uses the account's default VPC and subnets. Those subnets must have
outbound internet access so the worker can pull the image; the worker security
group has no inbound rules.

The bootstrap layer creates `weir-tofu-state-<account-id>` by default. It has
versioning, encryption, and public access blocked. The main stack stores its
state at `weir/terraform.tfstate` in that bucket.

### Container image

CI publishes the GPU image on pushes to `main`:

```text
ghcr.io/<owner>/<repo>:<full-git-sha>
ghcr.io/<owner>/<repo>:latest
```

The image installs the locked `gpu` extra. Use an immutable full-SHA tag for real
runs; `latest` is only convenient for a first smoke test.

AWS Batch does not receive GHCR credentials in this stack. The package must be
public so workers can pull it anonymously. GitHub Packages may create it as
private even when the repository is public; check the package settings after the
first publish.

The CPU queue uses the same image without a GPU reservation. It sets
`WEIR_REQUIRE_GPU=0`, so the CUDA check is skipped. For MuJoCo training, request
parallel environments explicitly:

```bash
WEIR_COMPUTE=cpu cloud/weir train agent=humanoid task=walk_forward \
  algo.n_envs=8 algo.vec_env=subproc train.total_steps=2000000
```

The CPU job requests 16 vCPU and 30 GiB on a `c7i.4xlarge` by default. The GPU
queue remains available for CUDA policy training or future MJX work.

### Artifacts

Remote layout:

```text
s3://weir-artifacts-<account-id>-<region>/
  runs/
    <run-id>/
      job-id.txt
      checkpoint.zip
      checkpoint.meta.json
      checkpoint_50000_steps.zip
      checkpoint_50000_steps.meta.json
      ...
  jobs/
    <job-id>/
      run-id.txt
```

The `jobs/<job-id>/run-id.txt` object is a reverse index so commands can accept
either a run ID or a Batch job ID.

Local metadata:

```text
outputs/cloud/<run-id>/
  job-id.txt
  artifact-uri.txt
  command.txt
  run/                 # after cloud/weir pull
```

`outputs/` is gitignored. The S3 bucket is private; the job role can only list
the bucket and read/write under `runs/`.

## Prerequisites

- AWS credentials with permission to manage IAM, S3, Batch, CloudWatch, and
  Budgets. Prefer a dedicated deploy profile over the account root user.
- For GPU runs: On-Demand G/VT vCPU quota of at least 4 in the target region.
  The quota code is `L-DB2E81BA`; it is commonly zero on new accounts.
- For CPU runs: On-Demand Standard vCPU quota of at least 16 in the target
  region. The default account quota is usually sufficient.
- A default VPC with working outbound internet.
- A public GHCR image at the configured repository.
- An email address for budget alerts.
- OpenTofu 1.10 or newer, AWS CLI v2, `jq`, and `git`.
- Local Docker is not required; CI builds the image.

## First-time setup

1. Publish the image by merging to `main`, or build and push it separately.
2. Make the GHCR package public.
3. Confirm the quota for the queue you plan to use:

   ```bash
   # GPU
   aws service-quotas get-service-quota \
     --service-code ec2 --quota-code L-DB2E81BA --region us-east-1
   # CPU (16 vCPU needed for the default CPU job)
   aws service-quotas get-service-quota \
     --service-code ec2 --quota-code L-1216C47A --region us-east-1
   ```

4. Create the state backend:

   ```bash
   cloud/weir setup us-east-1
   ```

5. Create and edit the local variables file:

   ```bash
   cp infra/terraform.tfvars.example infra/terraform.tfvars
   ```

   At minimum set `region` and `budget_email`. For a manual `tofu apply`, also
   set `image_tag`; `cloud/weir deploy` supplies the image tag explicitly (the
   current git SHA, or `WEIR_IMAGE_TAG`), so make sure the checkout matches the
   published image.

6. Apply the stack:

   ```bash
   cloud/weir deploy
   ```

7. Make sure the queue and compute environment reach a valid state before
   submitting a real run.

## What happens on `cloud/weir train`

1. The wrapper reads stack outputs from OpenTofu: region, artifact bucket, and
   the selected CPU/GPU queue plus job definition.
2. It creates a run ID such as `20261027T120000Z-c5d2fde`, or uses
   `WEIR_RUN_ID`.
3. It builds a container override:
   - `algo.device=cuda` or `algo.device=cpu` is prepended to the Hydra overrides.
   - `WEIR_ARTIFACTS_URI` is set to
     `s3://<artifact-bucket>/runs/<run-id>/`.
4. It submits the job to Batch and writes local metadata under
   `outputs/cloud/<run-id>/`.
5. It uploads `job-id.txt` into the run prefix and the reverse run-ID index into
   `jobs/<job-id>/run-id.txt`.

The job definition supplies these environment variables:

- `WEIR_RUN_DIR=/work/run`
- `WEIR_REQUIRE_GPU=1`
- `AWS_REGION=<region>`
- `HYDRA_FULL_ERROR=1`

Inside the container, `cloud/entrypoint.sh`:

1. Refuses to continue if CUDA is unavailable.
2. Starts a background `aws s3 sync` loop every 60 seconds.
3. Runs `weir-train "$@" hydra.run.dir=/work/run`.
4. Stops the loop and performs a final sync when the job exits or is cancelled.

The training process itself is unchanged. It writes checkpoints and manifests to
`/work/run`; periodic checkpoints are enabled with `algo.checkpoint_freq`.

## Job sizing and cost controls

- Instance: `g5.xlarge`, one NVIDIA GPU, on-demand only.
- Compute environment: 0 to 8 vCPUs, so at most two workers.
- Job: 4 vCPU, 8192 MiB memory, 1 GPU.
- Timeout: 6 hours per attempt.
- Retries: 5 attempts.
- Logs: 30 days.
- Budget: `$50/month` for EC2, with email alerts at 80% and 100%.

Important cost details:

- The budget is an alerting budget, not a hard spending cap. AWS can continue
  spending if a job runs longer.
- The budget filter covers EC2 compute only. S3 requests/storage and CloudWatch
  Logs are not included.
- Retries restart the job from scratch. A failed or timed-out job can multiply
  the cost.
- Spot instances and automatic checkpoint resume are not enabled.

A `g5.xlarge` is roughly one dollar per hour at list price. Most short experiments
cost cents, but measure the first real run rather than assuming a fixed
throughput.

## Staged training and early stopping

The wrapper does not currently download a previous S3 checkpoint into a new job.
Therefore:

- `algo.checkpoint=<path>` resumes weights only when that checkpoint is
  available inside the container.
- A Batch retry starts from scratch.
- Separate cloud jobs do not automatically continue from the previous job.

A practical staged workflow uses one capped job with frequent checkpoints:

```bash
cloud/weir train agent=humanoid task=walk_forward \
  train.total_steps=2000000 algo.ent_coef=0.02 \
  algo.checkpoint_freq=50000
```

While it runs:

1. Periodically run `cloud/weir pull <run-id>`.
2. Evaluate a checkpoint:

   ```bash
   uv run --extra cpu weir-eval \
     --checkpoint outputs/cloud/<run-id>/run/checkpoint_500000_steps.zip \
     --episodes 5 --max-steps 5000
   ```

3. Render a short video:

   ```bash
   MUJOCO_GL=egl uv run --extra cpu weir-render \
     --checkpoint outputs/cloud/<run-id>/run/checkpoint_500000_steps.zip \
     --output walk.mp4
   ```

4. When the result is good enough, cancel the job:

   ```bash
   cloud/weir cancel <run-id>
   ```

The job continues training while you evaluate, so cancel promptly once a
checkpoint passes. The final sync still runs after cancellation.

If you need true separate stages, add resume-from-S3 support first: download the
chosen checkpoint into `/work/run` in the container, append
`algo.checkpoint=<downloaded-path>`, and run only the additional steps.

## Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `missing command: tofu` / `jq` | Tools not on `PATH` | Install OpenTofu and `jq`, then retry. |
| Job stays `RUNNABLE` | No G/VT quota or no instance capacity | Check the service quota and Batch compute environment status. |
| `CannotPullContainerError` | GHCR package is private or the tag does not exist | Make the package public and confirm the image tag. |
| `CUDA is not available` | Wrong instance/image or GPU resource missing | Check the job definition GPU requirement and published image. |
| No log stream | Job has not started or the job ID is wrong | Use `cloud/weir status` and retry `logs` once the job starts. |
| `pull` finds no artifacts | Job failed before syncing, or artifact env is missing | Check logs and verify `WEIR_ARTIFACTS_URI` in the submitted job. |
| `destroy` fails on the artifact bucket | Bucket is not empty | Pull what you need, delete the objects, then destroy. |
| `default VPC not found` | Account/region has no default VPC | Create a default VPC or adapt the network data sources. |
| `AccessDenied` during setup/deploy | Credentials lack IAM/S3/Batch/Budget permissions | Use a deploy profile with the required permissions. |

CloudWatch logs show process startup, the CUDA check, artifact sync messages, and
the final training event. They do not show per-step learning progress; use
checkpoint cadence and local evaluation for that.

## Security and housekeeping

- Workers have no inbound network access; they only need outbound access to pull
  the image and write logs/artifacts.
- Artifacts are private. Do not put secrets in Hydra overrides or logs.
- `infra/terraform.tfvars`, `infra/backend.hcl`, and `outputs/` are gitignored.
- The state bucket is versioned and encrypted. It is intentionally not destroyed
  by `cloud/weir destroy`.
- Prefer a dedicated AWS deploy profile. Root credentials work but have far more
  access than the stack needs.
- To fully remove everything, destroy the main stack, empty and delete the
  artifact bucket, then decide whether to delete the state bucket and all of its
  object versions.

## Validation

```bash
tofu fmt -check -recursive infra
tofu -chdir=infra/bootstrap init -backend=false
tofu -chdir=infra/bootstrap validate
tofu -chdir=infra init -backend=false
tofu -chdir=infra validate
shellcheck cloud/*.sh
```

CI also runs the Python tests, ShellCheck, actionlint, OpenTofu validation, and a
container smoke test. See `.github/workflows/ci.yml`.

## Related documentation

- [`infra/README.md`](../infra/README.md) — manual OpenTofu steps and stack
  details.
- [`README.md`](../README.md) — project quick start and cloud command summary.
- [`configs/README.md`](../configs/README.md) — training configuration and
  example runs.
- [`cloud/`](./) — wrapper scripts and the container entrypoint.
- [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) — image publishing
  and validation.
