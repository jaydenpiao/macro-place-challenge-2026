#!/usr/bin/env bash
#
# Run the official air-gapped Docker evaluator and save reproducibility metadata.
#
# Usage:
#   scripts/run_cloud_parity.sh [run_id] [team] [placer_path]
#
# Defaults:
#   run_id:      cloud-parity-YYYYmmdd-HHMMSS
#   team:        jaydenpiao
#   placer_path: submissions/jaydenpiao/placer.py

set -euo pipefail

RUN_ID="${1:-cloud-parity-$(date -u +%Y%m%d-%H%M%S)}"
TEAM="${2:-jaydenpiao}"
PLACER_PATH="${3:-submissions/jaydenpiao/placer.py}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_PATH="$REPO_ROOT/configs/cloud_gpu.env"
RESULT_DIR="$REPO_ROOT/results/$RUN_ID"
DOCKER_LOG="$REPO_ROOT/eval_docker/results/${TEAM}.log"

cd "$REPO_ROOT"

set -a
source "$CONFIG_PATH"
set +a

python3 "$REPO_ROOT/scripts/check_cloud_parity_host.py" \
    --gpu-smoke-image "${JAYDEN_DOCKER_GPU_SMOKE_IMAGE:-nvidia/cuda:12.4.1-base-ubuntu22.04}"

git submodule update --init external/MacroPlacement

mkdir -p "$RESULT_DIR"

echo "run_id=$RUN_ID"
echo "team=$TEAM"
echo "placer=$PLACER_PATH"
echo "config=$CONFIG_PATH"

"$REPO_ROOT/eval_docker/run_eval.sh" "$TEAM" "$PLACER_PATH"

cp "$DOCKER_LOG" "$RESULT_DIR/run.log"

cat >"$RESULT_DIR/metadata.json" <<EOF
{
  "schema_version": 1,
  "run_id": "$RUN_ID",
  "team": "$TEAM",
  "placer_path": "$PLACER_PATH",
  "command": "scripts/run_cloud_parity.sh $RUN_ID $TEAM $PLACER_PATH",
  "git": {
    "commit": "$(git rev-parse HEAD)",
    "upstream_commit": "$(git rev-parse upstream/main 2>/dev/null || echo unknown)",
    "dirty": $(if [[ -n "$(git status --porcelain)" ]]; then echo true; else echo false; fi)
  },
  "env_knobs": {
    "JAYDEN_PLACER_SEED": "${JAYDEN_PLACER_SEED:-}",
    "JAYDEN_SEARCH_ITERS": "${JAYDEN_SEARCH_ITERS:-}",
    "JAYDEN_LEGAL_GAP": "${JAYDEN_LEGAL_GAP:-}",
    "JAYDEN_TRANSFORM": "${JAYDEN_TRANSFORM:-}",
    "JAYDEN_STRATEGY": "${JAYDEN_STRATEGY:-}",
    "JAYDEN_DOCKER_GPU_SMOKE_IMAGE": "${JAYDEN_DOCKER_GPU_SMOKE_IMAGE:-}"
  },
  "hardware": {
    "hostname": "$(hostname)",
    "nvidia_smi": "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd ';' -)"
  },
  "outputs": {
    "run_log": "results/$RUN_ID/run.log"
  }
}
EOF

echo "metadata written: $RESULT_DIR/metadata.json"
echo "docker log copied: $RESULT_DIR/run.log"
