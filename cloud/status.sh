#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cloud/lib.sh
source "$SCRIPT_DIR/lib.sh"

require_cmd aws
require_cmd jq
require_cmd tofu

ensure_deployed

region=$(tofu_out region)
queue=$(tofu_out job_queue)

printf 'JOB_ID\tNAME\tSTATUS\tCREATED\tEXIT\tREASON\n'

if [[ $# -gt 0 ]]; then
  job_id="$1"
  if [[ -f "$OUTPUT_DIR/$job_id/job-id.txt" ]]; then
    job_id=$(cat "$OUTPUT_DIR/$job_id/job-id.txt")
  fi
  aws batch describe-jobs --jobs "$job_id" --region "$region" --output json |
    jq -r '
      .jobs[] |
      [
        .jobId,
        .jobName,
        .status,
        (if .createdAt then (.createdAt / 1000 | todate) else "-" end),
        ((.container.exitCode // "-") | tostring),
        (.statusReason // "-")
      ] | @tsv'
else
  aws batch list-jobs --job-queue "$queue" --region "$region" --output json |
    jq -r '
      .jobSummaryList[] |
      [
        .jobId,
        .jobName,
        .status,
        (if .createdAt then (.createdAt / 1000 | todate) else "-" end),
        ((.container.exitCode // "-") | tostring),
        (.statusReason // "-")
      ] | @tsv'
fi
