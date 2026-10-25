#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cloud/lib.sh
source "$SCRIPT_DIR/lib.sh"

require_cmd aws
require_cmd tofu

ensure_deployed

arg="${1:-}"
job_id="$arg"
if [[ -z "$job_id" ]]; then
  job_id=$(latest_job_id)
elif [[ -f "$OUTPUT_DIR/$arg/job-id.txt" ]]; then
  job_id=$(cat "$OUTPUT_DIR/$arg/job-id.txt")
fi

region=$(tofu_out region)
aws batch terminate-job --job-id "$job_id" --reason "${2:-cancelled by weir}" \
  --region "$region" --query jobId --output text
