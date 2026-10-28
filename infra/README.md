# AWS Batch infrastructure

OpenTofu provisions a GPU and a CPU training stack: an S3 artifact bucket, IAM
roles, managed AWS Batch compute environments/queues, job definitions,
CloudWatch logs, and a monthly EC2 budget.

The stack deliberately uses the account's default VPC and subnets. Those subnets
must have outbound internet access so workers can pull the public GHCR image.
Workers get no inbound access.

## Prerequisites

- OpenTofu 1.10 or newer.
- AWS credentials with permission to create IAM, S3, Batch, CloudWatch, and
  Budgets resources.
- A public GHCR package at `ghcr.io/semzit/weir` (fork: change
  `image_repository`).
- For GPU runs: on-demand G-family vCPU quota of at least 4.
- For CPU runs: on-demand Standard vCPU quota of at least 16.
- An email address for budget alerts.
- A default VPC in the target region.

## Bootstrap state

The state bucket is created once with local state:

```bash
tofu -chdir=infra/bootstrap init
tofu -chdir=infra/bootstrap apply -var="region=us-east-1"
```

It defaults to `weir-tofu-state-<account-id>` and has versioning, encryption,
and public access blocked. Set `state_bucket_name` to override it.

## Deploy

```bash
bucket=$(tofu -chdir=infra/bootstrap output -raw state_bucket)
tofu -chdir=infra init \
  -backend-config="bucket=$bucket" \
  -backend-config="region=us-east-1"

cp infra/terraform.tfvars.example infra/terraform.tfvars
# set region, budget_email, and an immutable image_tag (the full git SHA)
tofu -chdir=infra apply
```

`image_tag=latest` is convenient for a first smoke test; use the full SHA for
real runs. Re-apply after publishing a new image so the job definition points at
it.

## Stack details

- GPU instance type: `g5.xlarge`, on-demand.
- GPU compute environment: 0 to 8 vCPUs, so at most two workers.
- GPU job: 4 vCPU, 8192 MiB memory, 1 GPU, 6-hour timeout, 5 attempts.
- GPU AMI image type: `ECS_AL2023_NVIDIA`.
- CPU instance type: `c7i.4xlarge`, on-demand.
- CPU compute environment: 0 to 16 vCPUs, one worker.
- CPU job: 16 vCPU, 30000 MiB memory, no GPU, 6-hour timeout, 5 attempts.
- CPU AMI image type: `ECS_AL2023`.
- Artifacts: `s3://<artifact-bucket>/runs/`.
- Logs: `/aws/batch/weir`, retained for 30 days.
- Budget: $50/month for EC2, alerts at 80% and 100%.

The job definitions set `WEIR_RUN_DIR`, `WEIR_REQUIRE_GPU`, `AWS_REGION`, and
`HYDRA_FULL_ERROR`. The submitter must set `WEIR_ARTIFACTS_URI` per run, for
example `s3://<artifact-bucket>/runs/<run-id>/`, and pass `algo.device=cuda` or
`algo.device=cpu` because the training config defaults to CPU. The CPU queue
uses 8 subprocess MuJoCo envs only when the submitter also passes
`algo.n_envs=8 algo.vec_env=subproc`.

## Outputs

```bash
tofu -chdir=infra output -raw job_queue
tofu -chdir=infra output -raw job_definition
tofu -chdir=infra output -raw cpu_job_queue
tofu -chdir=infra output -raw cpu_job_definition
tofu -chdir=infra output -raw artifact_uri_prefix
```

## Teardown

```bash
tofu -chdir=infra destroy
```

The artifact bucket has `force_destroy = false`; delete its objects first. The
state bucket is intentionally left in place.

## Validation

```bash
tofu fmt -check -recursive infra
tofu -chdir=infra/bootstrap init -backend=false
tofu -chdir=infra/bootstrap validate
tofu -chdir=infra init -backend=false
tofu -chdir=infra validate
```
