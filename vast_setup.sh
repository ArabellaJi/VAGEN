#!/usr/bin/env bash
# vast.ai one-shot setup for VAGEN Sokoban eval (eval-only, no training deps).
# Run once after the instance starts:
#   bash vast_setup.sh
set -euo pipefail

VAGEN_ROOT="${VAGEN_ROOT:-/root/VAGEN}"
REPO_URL="${REPO_URL:-https://github.com/YOURUSERNAME/VAGEN.git}"

echo "=== [1/4] System packages ==="
apt-get update -qq && apt-get install -y -qq git curl

echo "=== [2/4] Clone VAGEN repo ==="
if [ ! -d "${VAGEN_ROOT}/.git" ]; then
  git clone "${REPO_URL}" "${VAGEN_ROOT}"
else
  echo "Repo already exists at ${VAGEN_ROOT}, pulling latest..."
  git -C "${VAGEN_ROOT}" pull
fi

echo "=== [3/4] Install Python dependencies ==="
# SGLang (serves the 14B model via OpenAI-compatible API)
pip install --quiet "sglang[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/

# VAGEN eval-only deps
pip install --quiet \
  "omegaconf" \
  "openai>=1.0" \
  "Pillow" \
  "fire" \
  "gym-sokoban" \
  "gymnasium" \
  "uvicorn<0.41" \
  "qwen-vl-utils"

# Install vagen package itself (eval framework only)
pip install --quiet -e "${VAGEN_ROOT}"

echo "=== [4/4] Done! ==="
echo ""
echo "Next steps:"
echo "  cd ${VAGEN_ROOT}"
echo "  bash examples/evaluate/sokoban/sglang/eval_qwen25_vl_14b.sh"
