#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cloud/lib.sh
source "$SCRIPT_DIR/lib.sh"

require_cmd tofu
require_cmd git

[[ -f "$BACKEND_HCL" ]] || die "run cloud/weir setup <region> first"
[[ -f "$INFRA_DIR/terraform.tfvars" ]] ||
  die "copy infra/terraform.tfvars.example to infra/terraform.tfvars and edit it first"

image_tag="${WEIR_IMAGE_TAG:-$(git rev-parse HEAD)}"

tofu -chdir="$INFRA_DIR" init -input=false -backend-config=backend.hcl
tofu -chdir="$INFRA_DIR" apply -auto-approve -var="image_tag=$image_tag"

printf 'queue:     %s\n' "$(tofu_out job_queue)"
printf 'job def:   %s\n' "$(tofu_out job_definition)"
printf 'artifacts: %s\n' "$(tofu_out artifact_uri_prefix)"
printf 'image:     %s\n' "$(tofu_out image)"
