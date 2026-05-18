#!/usr/bin/env bash
# Generic SGLang eval launcher for Sokoban.
#
# Usage:
#   MODEL_PATH=/path/to/model bash eval_sglang.sh <config.yaml> [hydra overrides...]
#
# Example:
#   MODEL_PATH=/root/models/Qwen2.5-VL-32B-Instruct \
#     bash eval_sglang.sh config_32b_pomdp.yaml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAGEN_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# ── Args ────────────────────────────────────────────────────────────────
CONFIG="${1:?Usage: bash eval_sglang.sh <config.yaml> [overrides...]}"
# Make path absolute if relative
if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${SCRIPT_DIR}/${CONFIG}"
fi
shift || true   # remaining args forwarded to run_eval as hydra overrides

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH env var to local model directory}"
PORT="${PORT:-30000}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

CONFIG_NAME="$(basename "${CONFIG}" .yaml)"
SERVER_LOG="${LOG_DIR}/sglang_${CONFIG_NAME}_server.log"
EVAL_LOG="${LOG_DIR}/sglang_${CONFIG_NAME}_eval.log"

# ── Launch SGLang server ─────────────────────────────────────────────────
echo "[1/3] Starting SGLang server..."
echo "      model : ${MODEL_PATH}"
echo "      port  : ${PORT}"
echo "      log   : ${SERVER_LOG}"
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  echo "Stopping SGLang server (pid ${SERVER_PID})..."
  kill "${SERVER_PID}" >/dev/null 2>&1 || true
  wait "${SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# ── Wait for server ready ────────────────────────────────────────────────
echo "[2/3] Waiting for server to be ready..."
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "  Server ready after ${i}s."
    break
  fi
  if [ "${i}" -eq 240 ]; then
    echo "ERROR: Server did not start within 240s. Check ${SERVER_LOG}."
    exit 1
  fi
  sleep 1
done

# ── Run eval ─────────────────────────────────────────────────────────────
echo "[3/3] Running eval with config: ${CONFIG}"
python -m vagen.evaluate.run_eval \
  --config "${CONFIG}" \
  backends.sglang.model="${MODEL_PATH}" \
  fileroot="${VAGEN_ROOT}" \
  "$@" \
  2>&1 | tee "${EVAL_LOG}"
