#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cloud/lib.sh
source "$SCRIPT_DIR/lib.sh"

require_cmd aws
require_cmd jq
require_cmd tofu
require_cmd git

ensure_deployed

region=$(tofu_out region)
bucket=$(tofu_out artifact_bucket)
artifact_prefix=$(tofu_out artifact_uri_prefix)

compute="${WEIR_COMPUTE:-gpu}"
case "$compute" in
  gpu)
    queue=$(tofu_out job_queue)
    job_definition=$(tofu_out job_definition)
    overrides=("algo.device=cuda")
    ;;
  cpu)
    queue=$(tofu_out cpu_job_queue)
    job_definition=$(tofu_out cpu_job_definition)
    overrides=("algo.device=cpu")
    ;;
  *)
    die "WEIR_COMPUTE must be 'gpu' or 'cpu', got: $compute"
    ;;
esac

run_id="${WEIR_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)}"
[[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid run id: $run_id"
artifact_uri="${artifact_prefix}${run_id}/"

overrides+=("$@")
command_json=$(jq -nc --args '$ARGS.positional' "${overrides[@]}")
container_overrides=$(jq -nc \
  --arg uri "$artifact_uri" \
  --argjson command "$command_json" \
  '{environment: [{name: "WEIR_ARTIFACTS_URI", value: $uri}], command: $command}')

job_id=$(aws batch submit-job \
  --job-name "weir-$run_id" \
  --job-queue "$queue" \
  --job-definition "$job_definition" \
  --container-overrides "$container_overrides" \
  --tags "Project=weir,RunId=$run_id" \
  --region "$region" \
  --query jobId \
  --output text)

run_dir="$OUTPUT_DIR/$run_id"
mkdir -p "$run_dir"
printf '%s\n' "$job_id" > "$run_dir/job-id.txt"
printf '%s\n' "$artifact_uri" > "$run_dir/artifact-uri.txt"
printf '%s\n' "${overrides[@]}" > "$run_dir/command.txt"

aws s3 cp "$run_dir/job-id.txt" "${artifact_uri}job-id.txt" \
  --region "$region" --only-show-errors ||
  printf 'warning: could not upload job-id.txt\n' >&2
printf '%s\n' "$run_id" |
  aws s3 cp - "s3://$bucket/jobs/$job_id/run-id.txt" \
    --region "$region" --only-show-errors ||
  printf 'warning: could not upload run-id.txt\n' >&2

printf 'job:       %s\n' "$job_id"
printf 'run:       %s\n' "$run_id"
printf 'artifacts: %s\n' "$artifact_uri"
printf 'status:    cloud/weir status %s\n' "$job_id"
printf 'logs:      cloud/weir logs %s --follow\n' "$job_id"
printf 'pull:      cloud/weir pull %s\n' "$run_id"
