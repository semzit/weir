#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cloud/lib.sh
source "$SCRIPT_DIR/lib.sh"

require_cmd aws
require_cmd tofu

ensure_deployed

arg="${1:-}"
dest="${2:-}"
run_id=$(resolve_run_id "$arg")
bucket=$(tofu_out artifact_bucket)
region=$(tofu_out region)
prefix="s3://$bucket/runs/$run_id/"

listing=$(aws s3 ls "$prefix" --region "$region") ||
  die "could not list $prefix"
[[ -n "$listing" ]] || die "no artifacts at $prefix"

dest="${dest:-$OUTPUT_DIR/$run_id/run}"
mkdir -p "$dest"
aws s3 sync "$prefix" "$dest" --region "$region" --only-show-errors

printf 'pulled %s -> %s\n' "$prefix" "$dest"
