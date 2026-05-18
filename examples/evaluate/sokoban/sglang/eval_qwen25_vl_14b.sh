#!/usr/bin/env bash
# Run Sokoban eval with Qwen2.5-VL-14B via local SGLang server.
# Usage: bash eval_qwen25_vl_14b.sh [extra run_eval overrides...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAGEN_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-14B-Instruct}"
PORT="${PORT:-30000}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

SERVER_LOG="${LOG_DIR}/sglang_14b_server.log"
EVAL_LOG="${LOG_DIR}/sglang_14b_eval.log"

# ── Launch SGLang server ─────────────────────────────────────────────
echo "[1/3] Starting SGLang server (model: ${MODEL_PATH}, port: ${PORT})..."
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

# ── Wait for server ready ────────────────────────────────────────────
echo "[2/3] Waiting for server to be ready..."
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "  Server ready after ${i}s."
    break
  fi
  if [ "${i}" -eq 120 ]; then
    echo "ERROR: Server did not start within 120s. Check ${SERVER_LOG}."
    exit 1
  fi
  sleep 1
done

# ── Run eval ─────────────────────────────────────────────────────────
echo "[3/3] Running Sokoban eval..."
python -m vagen.evaluate.run_eval \
  --config "${SCRIPT_DIR}/config_14b.yaml" \
  backends.sglang.model="${MODEL_PATH}" \
  fileroot="${VAGEN_ROOT}" \
  "$@" \
  2>&1 | tee "${EVAL_LOG}"
