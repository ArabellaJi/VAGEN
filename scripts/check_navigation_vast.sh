#!/usr/bin/env bash
# Validate the graphics stack needed by VAGEN Navigation on a Vast.ai instance.
#
# Run from inside the Vast container after installing VAGEN dependencies:
#   bash scripts/check_navigation_vast.sh
#
# Optional:
#   NAV_GPU=1 bash scripts/check_navigation_vast.sh
#   USE_XVFB=1 bash scripts/check_navigation_vast.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
NAV_GPU="${NAV_GPU:-0}"
AI2THOR_CACHE_DIR="${AI2THOR_CACHE_DIR:-${HOME}/.ai2thor}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ -z "${VK_ICD_FILENAMES:-}" ]]; then
  shopt -s nullglob
  icd_files=(/usr/share/vulkan/icd.d/nvidia_icd*.json /etc/vulkan/icd.d/nvidia_icd*.json)
  shopt -u nullglob
  if (( ${#icd_files[@]} > 0 )); then
    export VK_ICD_FILENAMES="${icd_files[0]}"
  fi
fi

XVFB_PID=""
cleanup() {
  if [[ -n "${XVFB_PID}" ]]; then
    kill "${XVFB_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "${USE_XVFB:-0}" == "1" ]]; then
  if ! command -v Xvfb >/dev/null 2>&1; then
    echo "ERROR: USE_XVFB=1 was set, but Xvfb is not installed." >&2
    exit 1
  fi
  export DISPLAY="${DISPLAY:-:99}"
  Xvfb "${DISPLAY}" -screen 0 1280x1024x24 -nolisten tcp >/tmp/vagen-xvfb.log 2>&1 &
  XVFB_PID=$!
  sleep 1
  echo "Started Xvfb on DISPLAY=${DISPLAY} (pid=${XVFB_PID})."
fi

echo "=== GPU ==="
nvidia-smi

echo
echo "=== Vulkan ICD ==="
echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}"
if [[ -n "${VK_ICD_FILENAMES:-}" && ! -f "${VK_ICD_FILENAMES}" ]]; then
  echo "ERROR: VK_ICD_FILENAMES points to a missing file." >&2
  exit 1
fi

if command -v vulkaninfo >/dev/null 2>&1; then
  vulkaninfo --summary | sed -n '1,120p'
else
  echo "ERROR: vulkaninfo is not installed. AI2-THOR CloudRendering calls vulkaninfo before launching Unity." >&2
  echo "Install vulkan-tools in the conda env, load a Vulkan module if Delta provides one, or ask the admin to install it." >&2
  exit 1
fi

echo
echo "=== AI2-THOR cache ==="
echo "Expected cache directory: ${AI2THOR_CACHE_DIR}"
if [[ -d "${AI2THOR_CACHE_DIR}" ]]; then
  if [[ -L "${AI2THOR_CACHE_DIR}" ]]; then
    echo "Symlink target: $(readlink -f "${AI2THOR_CACHE_DIR}")"
  fi
  du -shL "${AI2THOR_CACHE_DIR}" 2>/dev/null || du -sh "${AI2THOR_CACHE_DIR}" 2>/dev/null || true
  find "${AI2THOR_CACHE_DIR}" -maxdepth 3 -type f \( -name 'thor-CloudRendering*.zip' -o -name 'AI2-THOR.x86_64' \) 2>/dev/null | sed -n '1,20p'
else
  echo "Cache directory does not exist yet. First Controller() call will download the Unity build."
fi

echo
echo "=== Python imports ==="
python - <<'PY'
import importlib.util
missing = [name for name in ("ai2thor", "PIL", "numpy") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing Python packages: {missing}")
print("Imports OK")
PY

echo
echo "=== AI2-THOR CloudRendering reset ==="
python - <<'PY'
import os
import pathlib
import time
import traceback

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

gpu = int(os.environ.get("NAV_GPU", "0"))
controller = None
try:
    print("Creating Controller. If a thor-CloudRendering zip appears here, this is first-run asset download, not reset latency.")
    t0 = time.perf_counter()
    controller = Controller(
        agentMode="default",
        gridSize=0.1,
        visibilityDistance=10,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=255,
        height=255,
        fieldOfView=100,
        platform=CloudRendering,
        gpu_device=gpu,
        server_timeout=300,
        server_start_timeout=300,
    )
    t_controller = time.perf_counter() - t0
    print(f"Controller ready in {t_controller:.1f}s")

    t1 = time.perf_counter()
    controller.reset(scene="FloorPlan1")
    t_reset = time.perf_counter() - t1
    frame = controller.last_event.frame
    print(f"AI2-THOR OK: frame_shape={frame.shape}, scene=FloorPlan1, gpu={gpu}, reset={t_reset:.2f}s")
except Exception:
    traceback.print_exc()
    player_logs = sorted(
        pathlib.Path.home().glob(".config/unity3d/Allen Institute for Artificial Intelligence/AI2-THOR/Player.log")
    )
    if player_logs:
        print(f"\nUnity Player.log: {player_logs[-1]}")
    raise
finally:
    if controller is not None:
        controller.stop()
PY

echo
echo "Navigation graphics smoke test passed."
