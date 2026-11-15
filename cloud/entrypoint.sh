#!/usr/bin/env bash
set -Eeuo pipefail

RUN_DIR="${WEIR_RUN_DIR:-/work/run}"
ARTIFACTS_URI="${WEIR_ARTIFACTS_URI:-}"
SYNC_PID=""
TRAIN_PID=""

sync_artifacts() {
  if [[ -n "$ARTIFACTS_URI" ]]; then
    aws s3 sync "$RUN_DIR" "$ARTIFACTS_URI" --only-show-errors
  fi
}

cleanup() {
  local status=$?
  if [[ -n "$SYNC_PID" ]]; then
    kill "$SYNC_PID" 2>/dev/null || true
    wait "$SYNC_PID" 2>/dev/null || true
  fi
  sync_artifacts || true
  exit "$status"
}

forward_signal() {
  if [[ -n "$TRAIN_PID" ]]; then
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
  fi
  exit 143
}

trap cleanup EXIT
trap forward_signal TERM INT

mkdir -p "$RUN_DIR"

if [[ "${WEIR_REQUIRE_GPU:-0}" == "1" ]]; then
  python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print(
    f"gpu.ready torch={torch.__version__} cuda={torch.version.cuda} "
    f"device={torch.cuda.get_device_name(0)}"
)
PY
fi

if [[ -n "$ARTIFACTS_URI" ]]; then
  echo "artifact sync enabled: $ARTIFACTS_URI"
  (
    while true; do
      sleep 60
      sync_artifacts || true
    done
  ) &
  SYNC_PID=$!
else
  echo "artifact sync disabled"
fi

weir-train "$@" hydra.run.dir="$RUN_DIR" &
TRAIN_PID=$!
wait "$TRAIN_PID"
