#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cloud/lib.sh
source "$SCRIPT_DIR/lib.sh"

require_cmd tofu

[[ -f "$BACKEND_HCL" ]] || die "run cloud/weir setup <region> first"

tofu -chdir="$INFRA_DIR" init -input=false -backend-config=backend.hcl
printf 'destroying the Batch stack; artifacts must be empty or force_destroy must be enabled\n'
tofu -chdir="$INFRA_DIR" destroy
