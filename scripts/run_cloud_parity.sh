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

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker is required for cloud parity evaluation" >&2
    exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "error: nvidia-smi is required; use an NVIDIA GPU host" >&2
    exit 2
fi

git submodule update --init external/MacroPlacement

set -a
source "$CONFIG_PATH"
set +a

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
    "JAYDEN_TRANSFORM": "${JAYDEN_TRANSFORM:-}"
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
