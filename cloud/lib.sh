#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
INFRA_DIR="$REPO_ROOT/infra"
# shellcheck disable=SC2034
BOOTSTRAP_DIR="$INFRA_DIR/bootstrap"
# shellcheck disable=SC2034
BACKEND_HCL="$INFRA_DIR/backend.hcl"
OUTPUT_DIR="$REPO_ROOT/outputs/cloud"

die() {
  printf 'weir: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

tofu_out() {
  tofu -chdir="$INFRA_DIR" output -raw "$1"
}

ensure_deployed() {
  tofu -chdir="$INFRA_DIR" output -raw job_queue >/dev/null 2>&1 ||
    die "infra is not deployed; run cloud/weir deploy first"
}

latest_run_id() {
  local dir
  # shellcheck disable=SC2012
  dir=$(ls -1dt "$OUTPUT_DIR"/*/ 2>/dev/null | head -n 1 || true)
  [[ -n "$dir" ]] || die "no local runs; pass a run id or job id"
  basename "${dir%/}"
}

job_id_for_run() {
  local run_id="$1"
  local file="$OUTPUT_DIR/$run_id/job-id.txt"
  if [[ -f "$file" ]]; then
    cat "$file"
    return
  fi
  aws s3 cp "s3://$(tofu_out artifact_bucket)/runs/$run_id/job-id.txt" - \
    --region "$(tofu_out region)" --only-show-errors 2>/dev/null ||
    die "no job id found for run: $run_id"
}

latest_job_id() {
  job_id_for_run "$(latest_run_id)"
}

resolve_run_id() {
  local id="${1:-}"
  if [[ -z "$id" ]]; then
    latest_run_id
    return
  fi
  if [[ -d "$OUTPUT_DIR/$id" ]]; then
    printf '%s\n' "$id"
    return
  fi

  local remote
  if remote=$(aws s3 cp "s3://$(tofu_out artifact_bucket)/jobs/$id/run-id.txt" - \
      --region "$(tofu_out region)" --only-show-errors 2>/dev/null) &&
    [[ -n "$remote" ]]; then
    printf '%s\n' "$remote"
    return
  fi
  printf '%s\n' "$id"
}
