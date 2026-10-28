#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cloud/lib.sh
source "$SCRIPT_DIR/lib.sh"

require_cmd aws
require_cmd tofu

job_id=""
follow=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --follow|-f)
      follow=1
      shift
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      [[ -z "$job_id" ]] || die "only one job id is supported"
      job_id="$1"
      shift
      ;;
  esac
done

ensure_deployed

if [[ -z "$job_id" ]]; then
  job_id=$(latest_job_id)
fi
if [[ -f "$OUTPUT_DIR/$job_id/job-id.txt" ]]; then
  job_id=$(cat "$OUTPUT_DIR/$job_id/job-id.txt")
fi

region=$(tofu_out region)
log_group=$(tofu_out log_group)
stream=$(aws batch describe-jobs --jobs "$job_id" --region "$region" \
  --query 'jobs[0].container.logStreamName' --output text)

if [[ -z "$stream" || "$stream" == "None" ]]; then
  die "no log stream for job $job_id yet"
fi

args=(--log-stream-names "$stream" --since 7d)
if [[ "$follow" == "1" ]]; then
  args+=(--follow)
fi
aws logs tail "$log_group" "${args[@]}" --region "$region"
