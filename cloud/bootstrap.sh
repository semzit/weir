#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cloud/lib.sh
source "$SCRIPT_DIR/lib.sh"

require_cmd tofu

region="${1:-${WEIR_REGION:-}}"
[[ -n "$region" ]] || die "usage: cloud/weir setup <region> (or set WEIR_REGION)"

tofu -chdir="$BOOTSTRAP_DIR" init -input=false
tofu -chdir="$BOOTSTRAP_DIR" apply -auto-approve -var="region=$region"

bucket=$(tofu -chdir="$BOOTSTRAP_DIR" output -raw state_bucket)
cat > "$BACKEND_HCL" <<EOF
bucket = "$bucket"
region = "$region"
EOF

tofu -chdir="$INFRA_DIR" init -input=false -backend-config=backend.hcl

printf 'state bucket: %s\n' "$bucket"
if [[ ! -f "$INFRA_DIR/terraform.tfvars" ]]; then
  printf 'next: cp infra/terraform.tfvars.example infra/terraform.tfvars\n'
  printf '      edit region, budget_email, and image_tag; then run cloud/weir deploy\n'
else
  printf 'next: cloud/weir deploy\n'
fi
