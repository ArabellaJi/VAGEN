#!/usr/bin/env bash
# Vast.ai launcher for VAGEN Navigation GRPO.
#
# This is the non-Slurm counterpart to run_navigation_grpo.sh. It assumes you
# are already inside a Vast.ai SSH/Jupyter container with the VAGEN conda/env
# activated and dependencies installed.
#
# Typical use:
#   CONDITION=quick bash run_navigation_grpo_vast.sh
#
# Useful knobs:
#   NAV_GPU=0 TRAIN_GPU=1 CONDITION=window3_thumb bash run_navigation_grpo_vast.sh
#   NAV_GPU=0 TRAIN_GPU=1 CONDITION=window NAV_HISTORY_WINDOW=5 NAV_THUMBNAIL_SCALE=0.25 bash run_navigation_grpo_vast.sh
#   PREDOWNLOAD_SCENES=1 CONDITION=quick bash run_navigation_grpo_vast.sh
#   USE_XVFB=1 CONDITION=quick bash run_navigation_grpo_vast.sh

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ -d /workspace ]]; then
  DEFAULT_RUN_ROOT="/workspace/vagen_runs"
else
  DEFAULT_RUN_ROOT="${PROJECT_ROOT}/runs/vast"
fi
RUN_ROOT="${RUN_ROOT:-${DEFAULT_RUN_ROOT}}"
HF_CACHE="${HF_CACHE:-${RUN_ROOT}/hf_cache}"
DEFAULT_TMP_ROOT="${VAGEN_TMP_ROOT:-/tmp}"
SCRIPTDIR="${PROJECT_ROOT}/examples/train/navigation"

CONDITION="${CONDITION:-quick}"
NAV_SERVER_PORT="${NAV_SERVER_PORT:-8000}"
NAV_SERVER_HOST="${NAV_SERVER_HOST:-127.0.0.1}"

GPU_COUNT_DETECTED=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT_DETECTED="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
fi

NAV_GPU="${NAV_GPU:-0}"
if [[ -z "${TRAIN_GPU:-}" ]]; then
  if [[ "${GPU_COUNT_DETECTED}" -ge 2 ]]; then
    TRAIN_GPU=1
  else
    TRAIN_GPU=0
  fi
fi

validate_gpu_index() {
  local label="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${label}=${value} is not a numeric GPU index."
    exit 1
  fi
  if [[ "${GPU_COUNT_DETECTED}" -gt 0 && "${value}" -ge "${GPU_COUNT_DETECTED}" ]]; then
    echo "ERROR: ${label}=${value}, but nvidia-smi reports only ${GPU_COUNT_DETECTED} GPU(s), indexed 0..$((GPU_COUNT_DETECTED - 1))."
    echo "For a single-GPU smoke test use: ALLOW_SINGLE_GPU_TRAIN=1 NAV_GPU=0 TRAIN_GPU=0 CONDITION=quick bash run_navigation_grpo_vast.sh"
    exit 1
  fi
}

validate_gpu_index "NAV_GPU" "${NAV_GPU}"
validate_gpu_index "TRAIN_GPU" "${TRAIN_GPU}"

if [[ "${NAV_GPU}" == "${TRAIN_GPU}" && "${ALLOW_SINGLE_GPU_TRAIN:-0}" != "1" ]]; then
  echo "ERROR: NAV_GPU and TRAIN_GPU are both ${NAV_GPU}."
  echo "Use a 2-GPU instance for training, or set ALLOW_SINGLE_GPU_TRAIN=1 for a small forced smoke test."
  echo "For single-GPU Unity validation, run: bash scripts/check_navigation_vast.sh"
  exit 1
fi

NAV_CUDA_VISIBLE_DEVICES="${NAV_CUDA_VISIBLE_DEVICES:-${NAV_GPU}}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-${TRAIN_GPU}}"

mkdir -p "${PROJECT_ROOT}/logs" "${RUN_ROOT}" "${HF_CACHE}"
cd "${PROJECT_ROOT}"

validate_history_window_value() {
  local value="$1"
  if ! [[ "${value}" =~ ^-?[0-9]+$ ]]; then
    echo "ERROR: NAV_HISTORY_WINDOW must be an integer >= -1; got '${value}'." >&2
    exit 1
  fi
  if [[ "${value}" -lt -1 ]]; then
    echo "ERROR: NAV_HISTORY_WINDOW must be >= -1; got '${value}'." >&2
    echo "Use -1 for all history, 0 for no history, or k>0 for the latest k completed turns." >&2
    exit 1
  fi
}

validate_thumbnail_scale_value() {
  local value="$1"
  if ! [[ "${value}" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]]; then
    echo "ERROR: NAV_THUMBNAIL_SCALE must be a positive number; got '${value}'." >&2
    exit 1
  fi
  if ! awk -v s="${value}" 'BEGIN { exit !(s > 0) }'; then
    echo "ERROR: NAV_THUMBNAIL_SCALE must be > 0; got '${value}'." >&2
    exit 1
  fi
}

validate_positive_integer_value() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${name} must be a positive integer; got '${value}'." >&2
    exit 1
  fi
}

validate_nonnegative_integer_value() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${name} must be a non-negative integer; got '${value}'." >&2
    exit 1
  fi
}

validate_positive_number_value() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]]; then
    echo "ERROR: ${name} must be a positive number; got '${value}'." >&2
    exit 1
  fi
  if ! awk -v s="${value}" 'BEGIN { exit !(s > 0) }'; then
    echo "ERROR: ${name} must be > 0; got '${value}'." >&2
    exit 1
  fi
}

window_experiment_name() {
  local window="$1"
  local scale="$2"
  local base
  local scale_slug
  if [[ "${window}" == "-1" ]]; then
    base="nav_grpo_window_all"
  else
    base="nav_grpo_window${window}"
  fi
  if awk -v s="${scale}" 'BEGIN { exit !(s == 1.0) }'; then
    echo "${base}"
  else
    scale_slug="${scale//./p}"
    echo "${base}_thumb${scale_slug}"
  fi
}

case "${CONDITION}" in
  quick)
    EXPERIMENT_NAME=nav_grpo_quick
    TRAIN_DATA=${SCRIPTDIR}/train_navigation_quick.yaml
    VAL_DATA=${SCRIPTDIR}/val_navigation_quick.yaml
    NAV_MAX_ENVS=64
    TRAINING_STEPS=5
    TRAIN_BATCH_SIZE=4
    ROLLOUT_N=1
    DATA_MAX_PROMPT=3000
    DATA_MAX_RESPONSE=5000
    ROLLOUT_PROMPT=6000
    ROLLOUT_RESPONSE=1024
    MAX_BATCHED_TOKENS=8000
    GPU_MEM_UTIL=0.70
    CONCAT_MULTI_TURN=True
    AGENT_LOOP_CFG=${PROJECT_ROOT}/vagen/configs/agent.yaml
    HISTORY_ARGS=()
    ROLLOUT_NUM_WORKERS=4
    ;;
  full_memory)
    EXPERIMENT_NAME=nav_grpo_full_memory
    TRAIN_DATA=${SCRIPTDIR}/train_navigation_window_exp.yaml
    VAL_DATA=${SCRIPTDIR}/val_navigation_window_exp.yaml
    NAV_MAX_ENVS=64
    TRAINING_STEPS=200
    TRAIN_BATCH_SIZE=30
    ROLLOUT_N=1
    DATA_MAX_PROMPT=3000
    DATA_MAX_RESPONSE=10000
    ROLLOUT_PROMPT=12000
    ROLLOUT_RESPONSE=1024
    MAX_BATCHED_TOKENS=14000
    GPU_MEM_UTIL=0.65
    CONCAT_MULTI_TURN=True
    AGENT_LOOP_CFG=${PROJECT_ROOT}/vagen/configs/agent.yaml
    HISTORY_ARGS=()
    ROLLOUT_NUM_WORKERS=8
    ;;
  no_memory)
    EXPERIMENT_NAME=nav_grpo_no_memory
    TRAIN_DATA=${SCRIPTDIR}/train_navigation_window_exp.yaml
    VAL_DATA=${SCRIPTDIR}/val_navigation_window_exp.yaml
    NAV_MAX_ENVS=64
    TRAINING_STEPS=200
    TRAIN_BATCH_SIZE=30
    ROLLOUT_N=1
    DATA_MAX_PROMPT=3000
    DATA_MAX_RESPONSE=5000
    ROLLOUT_PROMPT=4000
    ROLLOUT_RESPONSE=1024
    MAX_BATCHED_TOKENS=6000
    GPU_MEM_UTIL=0.70
    CONCAT_MULTI_TURN=False
    AGENT_LOOP_CFG=${PROJECT_ROOT}/vagen/configs/agent_no_concat.yaml
    HISTORY_ARGS=(trainer.history_window_size=0)
    ROLLOUT_NUM_WORKERS=8
    ;;
  window3)
    EXPERIMENT_NAME=nav_grpo_window3
    TRAIN_DATA=${SCRIPTDIR}/train_navigation_window_exp.yaml
    VAL_DATA=${SCRIPTDIR}/val_navigation_window_exp.yaml
    NAV_MAX_ENVS=64
    TRAINING_STEPS=200
    TRAIN_BATCH_SIZE=30
    ROLLOUT_N=1
    DATA_MAX_PROMPT=3000
    DATA_MAX_RESPONSE=7000
    ROLLOUT_PROMPT=7000
    ROLLOUT_RESPONSE=1024
    MAX_BATCHED_TOKENS=9000
    GPU_MEM_UTIL=0.68
    CONCAT_MULTI_TURN=False
    AGENT_LOOP_CFG=${PROJECT_ROOT}/vagen/configs/agent_no_concat.yaml
    HISTORY_ARGS=(trainer.history_window_size=3 trainer.thumbnail_scale=1.0)
    ROLLOUT_NUM_WORKERS=8
    ;;
  thumbnail)
    EXPERIMENT_NAME=nav_grpo_thumbnail
    TRAIN_DATA=${SCRIPTDIR}/train_navigation_window_exp.yaml
    VAL_DATA=${SCRIPTDIR}/val_navigation_window_exp.yaml
    NAV_MAX_ENVS=64
    TRAINING_STEPS=200
    TRAIN_BATCH_SIZE=30
    ROLLOUT_N=1
    DATA_MAX_PROMPT=3000
    DATA_MAX_RESPONSE=7000
    ROLLOUT_PROMPT=8000
    ROLLOUT_RESPONSE=1024
    MAX_BATCHED_TOKENS=10000
    GPU_MEM_UTIL=0.68
    CONCAT_MULTI_TURN=False
    AGENT_LOOP_CFG=${PROJECT_ROOT}/vagen/configs/agent_no_concat.yaml
    HISTORY_ARGS=(trainer.history_window_size=-1 trainer.thumbnail_scale=0.25)
    ROLLOUT_NUM_WORKERS=8
    ;;
  window3_thumb)
    EXPERIMENT_NAME=nav_grpo_window3_thumb
    TRAIN_DATA=${SCRIPTDIR}/train_navigation_window_exp.yaml
    VAL_DATA=${SCRIPTDIR}/val_navigation_window_exp.yaml
    NAV_MAX_ENVS=64
    TRAINING_STEPS=200
    TRAIN_BATCH_SIZE=30
    ROLLOUT_N=1
    DATA_MAX_PROMPT=3000
    DATA_MAX_RESPONSE=7000
    ROLLOUT_PROMPT=7000
    ROLLOUT_RESPONSE=1024
    MAX_BATCHED_TOKENS=9000
    GPU_MEM_UTIL=0.68
    CONCAT_MULTI_TURN=False
    AGENT_LOOP_CFG=${PROJECT_ROOT}/vagen/configs/agent_no_concat.yaml
    HISTORY_ARGS=(trainer.history_window_size=3 trainer.thumbnail_scale=0.25)
    ROLLOUT_NUM_WORKERS=8
    ;;
  window)
    NAV_HISTORY_WINDOW="${NAV_HISTORY_WINDOW:-3}"
    NAV_THUMBNAIL_SCALE="${NAV_THUMBNAIL_SCALE:-1.0}"
    validate_history_window_value "${NAV_HISTORY_WINDOW}"
    validate_thumbnail_scale_value "${NAV_THUMBNAIL_SCALE}"
    EXPERIMENT_NAME="${NAV_EXPERIMENT_NAME:-$(window_experiment_name "${NAV_HISTORY_WINDOW}" "${NAV_THUMBNAIL_SCALE}")}"
    TRAIN_DATA=${SCRIPTDIR}/train_navigation_window_exp.yaml
    VAL_DATA=${SCRIPTDIR}/val_navigation_window_exp.yaml
    NAV_MAX_ENVS=64
    TRAINING_STEPS=200
    TRAIN_BATCH_SIZE=30
    ROLLOUT_N=1
    DATA_MAX_PROMPT=3000
    DATA_MAX_RESPONSE=7000
    ROLLOUT_PROMPT=7000
    ROLLOUT_RESPONSE=1024
    MAX_BATCHED_TOKENS=9000
    GPU_MEM_UTIL=0.68
    CONCAT_MULTI_TURN=False
    AGENT_LOOP_CFG=${PROJECT_ROOT}/vagen/configs/agent_no_concat.yaml
    HISTORY_ARGS=(trainer.history_window_size="${NAV_HISTORY_WINDOW}" trainer.thumbnail_scale="${NAV_THUMBNAIL_SCALE}")
    ROLLOUT_NUM_WORKERS=8
    ;;
  *)
    echo "ERROR: Unknown CONDITION '${CONDITION}'."
    echo "Valid: quick | full_memory | no_memory | window3 | thumbnail | window3_thumb | window"
    exit 1
    ;;
esac

if [[ -n "${NAV_TRAINING_STEPS:-}" ]]; then
  validate_positive_integer_value "NAV_TRAINING_STEPS" "${NAV_TRAINING_STEPS}"
  TRAINING_STEPS="${NAV_TRAINING_STEPS}"
fi
if [[ -n "${NAV_TRAIN_BATCH_SIZE:-}" ]]; then
  validate_positive_integer_value "NAV_TRAIN_BATCH_SIZE" "${NAV_TRAIN_BATCH_SIZE}"
  TRAIN_BATCH_SIZE="${NAV_TRAIN_BATCH_SIZE}"
fi
if [[ -n "${NAV_ROLLOUT_NUM_WORKERS:-}" ]]; then
  validate_positive_integer_value "NAV_ROLLOUT_NUM_WORKERS" "${NAV_ROLLOUT_NUM_WORKERS}"
  ROLLOUT_NUM_WORKERS="${NAV_ROLLOUT_NUM_WORKERS}"
fi
if [[ -n "${NAV_MAX_ENVS_OVERRIDE:-}" ]]; then
  validate_positive_integer_value "NAV_MAX_ENVS_OVERRIDE" "${NAV_MAX_ENVS_OVERRIDE}"
  NAV_MAX_ENVS="${NAV_MAX_ENVS_OVERRIDE}"
fi
if [[ -n "${NAV_MAX_INFLIGHT_OVERRIDE:-}" ]]; then
  validate_positive_integer_value "NAV_MAX_INFLIGHT_OVERRIDE" "${NAV_MAX_INFLIGHT_OVERRIDE}"
  NAV_MAX_INFLIGHT="${NAV_MAX_INFLIGHT_OVERRIDE}"
else
  NAV_MAX_INFLIGHT="${NAV_MAX_INFLIGHT:-${NAV_MAX_ENVS}}"
fi
if [[ -n "${NAV_THREAD_POOL_SIZE_OVERRIDE:-}" ]]; then
  validate_positive_integer_value "NAV_THREAD_POOL_SIZE_OVERRIDE" "${NAV_THREAD_POOL_SIZE_OVERRIDE}"
  NAV_THREAD_POOL_SIZE="${NAV_THREAD_POOL_SIZE_OVERRIDE}"
else
  NAV_THREAD_POOL_SIZE="${NAV_THREAD_POOL_SIZE:-${NAV_MAX_ENVS}}"
fi

if command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")}"
elif [[ -d /usr/local/cuda ]]; then
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
else
  CUDA_HOME="${CUDA_HOME:-}"
fi

if [[ -n "${CUDA_HOME}" ]]; then
  export CUDA_HOME CUDA_PATH="${CUDA_HOME}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"
export HF_HOME="${HF_HOME:-${HF_CACHE}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUN_ROOT}/xdg_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export FLASHINFER_ENABLE_JIT="${FLASHINFER_ENABLE_JIT:-0}"
export FLASHINFER_JIT_WORKER_TIMEOUT="${FLASHINFER_JIT_WORKER_TIMEOUT:-60}"
export VAGEN_SGLANG_INIT_TIMEOUT="${VAGEN_SGLANG_INIT_TIMEOUT:-600}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[console,wandb]}"
NAV_LOG_IMAGE_ENABLE="${NAV_LOG_IMAGE_ENABLE:-false}"
NAV_VAL_BEFORE_TRAIN="${NAV_VAL_BEFORE_TRAIN:-True}"
NAV_TEST_FREQ="${NAV_TEST_FREQ:-10}"
NAV_SAVE_FREQ="${NAV_SAVE_FREQ:-20}"
NAV_LOG_VAL_GENERATIONS="${NAV_LOG_VAL_GENERATIONS:-10}"
NAV_LENIENT_ACTION_PARSE="${NAV_LENIENT_ACTION_PARSE:-false}"
NAV_EXAMPLE_COUNT="${NAV_EXAMPLE_COUNT:-}"
NAV_ENV_RETRIES="${NAV_ENV_RETRIES:-}"
NAV_ENV_TIMEOUT="${NAV_ENV_TIMEOUT:-}"
NAV_ENV_MAX_DELAY="${NAV_ENV_MAX_DELAY:-}"
NAV_ADMIT_TIMEOUT="${NAV_ADMIT_TIMEOUT:-5}"
NAV_SESSION_TIMEOUT="${NAV_SESSION_TIMEOUT:-3600}"
NAV_TRAIN_N_ENVS_OVERRIDE="${NAV_TRAIN_N_ENVS_OVERRIDE:-}"
NAV_VAL_N_ENVS_OVERRIDE="${NAV_VAL_N_ENVS_OVERRIDE:-}"
export WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/wandb}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${XDG_CACHE_HOME}" "${WANDB_DIR}"

if [[ -n "${NAV_ENV_RETRIES}" ]]; then
  validate_nonnegative_integer_value "NAV_ENV_RETRIES" "${NAV_ENV_RETRIES}"
fi
if [[ -n "${NAV_ENV_TIMEOUT}" ]]; then
  validate_positive_number_value "NAV_ENV_TIMEOUT" "${NAV_ENV_TIMEOUT}"
fi
if [[ -n "${NAV_ENV_MAX_DELAY}" ]]; then
  validate_positive_number_value "NAV_ENV_MAX_DELAY" "${NAV_ENV_MAX_DELAY}"
fi
validate_positive_number_value "NAV_ADMIT_TIMEOUT" "${NAV_ADMIT_TIMEOUT}"
validate_positive_number_value "NAV_SESSION_TIMEOUT" "${NAV_SESSION_TIMEOUT}"
if [[ -n "${NAV_TRAIN_N_ENVS_OVERRIDE}" ]]; then
  validate_positive_integer_value "NAV_TRAIN_N_ENVS_OVERRIDE" "${NAV_TRAIN_N_ENVS_OVERRIDE}"
fi
if [[ -n "${NAV_VAL_N_ENVS_OVERRIDE}" ]]; then
  validate_positive_integer_value "NAV_VAL_N_ENVS_OVERRIDE" "${NAV_VAL_N_ENVS_OVERRIDE}"
fi

if [[ "${NAV_LENIENT_ACTION_PARSE}" != "false" || -n "${NAV_EXAMPLE_COUNT}" || -n "${NAV_ENV_RETRIES}" || -n "${NAV_ENV_TIMEOUT}" || -n "${NAV_ENV_MAX_DELAY}" || -n "${NAV_TRAIN_N_ENVS_OVERRIDE}" || -n "${NAV_VAL_N_ENVS_OVERRIDE}" ]]; then
  NAV_CONFIG_OVERRIDE_DIR="${RUN_ROOT}/config_overrides/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${NAV_CONFIG_OVERRIDE_DIR}"
  python - "${TRAIN_DATA}" "${VAL_DATA}" "${NAV_CONFIG_OVERRIDE_DIR}" "${NAV_LENIENT_ACTION_PARSE}" "${NAV_EXAMPLE_COUNT}" "${NAV_ENV_RETRIES}" "${NAV_ENV_TIMEOUT}" "${NAV_ENV_MAX_DELAY}" "${NAV_TRAIN_N_ENVS_OVERRIDE}" "${NAV_VAL_N_ENVS_OVERRIDE}" <<'PY'
import os
import sys
import yaml

train_src, val_src, out_dir, lenient_raw, example_raw, retries_raw, timeout_raw, max_delay_raw, train_n_envs_raw, val_n_envs_raw = sys.argv[1:11]
truthy = {"1", "true", "yes", "on"}
falsy = {"0", "false", "no", "off", ""}
value = lenient_raw.strip().lower()
if value not in truthy | falsy:
    raise SystemExit(f"NAV_LENIENT_ACTION_PARSE must be boolean-like, got {lenient_raw!r}")
lenient = value in truthy
example_count = None if example_raw == "" else int(example_raw)
if example_count is not None and example_count < 0:
    raise SystemExit(f"NAV_EXAMPLE_COUNT must be >= 0, got {example_count}")
retries = None if retries_raw == "" else int(retries_raw)
timeout = None if timeout_raw == "" else float(timeout_raw)
max_delay = None if max_delay_raw == "" else float(max_delay_raw)
train_n_envs = None if train_n_envs_raw == "" else int(train_n_envs_raw)
val_n_envs = None if val_n_envs_raw == "" else int(val_n_envs_raw)

def patch_one(src, name, n_envs_override):
    with open(src) as f:
        cfg = yaml.safe_load(f)
    for env in cfg.get("envs", []):
        if n_envs_override is not None:
            env["n_envs"] = n_envs_override
        env_cfg = env.setdefault("config", {})
        env_cfg["lenient_action_parse"] = bool(lenient)
        if example_count is not None:
            env_cfg["example_count"] = example_count
        if retries is not None:
            env_cfg["retries"] = retries
        if timeout is not None:
            env_cfg["timeout"] = timeout
        if max_delay is not None:
            env_cfg["max_delay"] = max_delay
    dst = os.path.join(out_dir, name)
    with open(dst, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Wrote navigation config override: {dst}")
    return dst

print(patch_one(train_src, "train.yaml", train_n_envs))
print(patch_one(val_src, "val.yaml", val_n_envs))
PY
  TRAIN_DATA="${NAV_CONFIG_OVERRIDE_DIR}/train.yaml"
  VAL_DATA="${NAV_CONFIG_OVERRIDE_DIR}/val.yaml"
fi

if [[ -z "${VK_ICD_FILENAMES:-}" ]]; then
  shopt -s nullglob
  icd_files=(/usr/share/vulkan/icd.d/nvidia_icd*.json /etc/vulkan/icd.d/nvidia_icd*.json)
  shopt -u nullglob
  if (( ${#icd_files[@]} > 0 )); then
    export VK_ICD_FILENAMES="${icd_files[0]}"
  fi
fi

XVFB_PID=""
if [[ "${USE_XVFB:-0}" == "1" ]]; then
  if ! command -v Xvfb >/dev/null 2>&1; then
    echo "ERROR: USE_XVFB=1 was set, but Xvfb is not installed." >&2
    exit 1
  fi
  export DISPLAY="${DISPLAY:-:99}"
  Xvfb "${DISPLAY}" -screen 0 1280x1024x24 -nolisten tcp >/tmp/vagen-xvfb.log 2>&1 &
  XVFB_PID=$!
  sleep 1
fi

JOB_TMP="${JOB_TMP:-${DEFAULT_TMP_ROOT}/vg_$$}"
export TMPDIR="${TMPDIR:-${JOB_TMP}/t}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export RAY_TMPDIR="${RAY_TMPDIR:-${JOB_TMP}/r}"
SYNC_ROOT="${SYNC_ROOT:-${JOB_TMP}/sglang_sync}"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" "${SYNC_ROOT}"

if ! (test -d "${TMPDIR}" && test -w "${TMPDIR}" && : > "${TMPDIR}/.vagen_write_test" && rm -f "${TMPDIR}/.vagen_write_test"); then
  echo "ERROR: TMPDIR is not writable: ${TMPDIR}" >&2
  echo "Check disk space and permissions with: df -h /workspace /tmp && ls -ld '${TMPDIR}'" >&2
  exit 1
fi

VAGEN_SGLANG_WEIGHT_SYNC_METHOD="${VAGEN_SGLANG_WEIGHT_SYNC_METHOD:-disk}"
VAGEN_SGLANG_WEIGHT_SYNC_DIR="${VAGEN_SGLANG_WEIGHT_SYNC_DIR:-${SYNC_ROOT}}"
VAGEN_SGLANG_WEIGHT_SYNC_LOAD_FORMAT="${VAGEN_SGLANG_WEIGHT_SYNC_LOAD_FORMAT:-auto}"
VAGEN_SGLANG_WEIGHT_SYNC_FLUSH_CACHE="${VAGEN_SGLANG_WEIGHT_SYNC_FLUSH_CACHE:-true}"

RAY_RUNTIME_ENV_ARGS=(
  "+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_HOME='${CUDA_HOME}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_PATH='${CUDA_PATH:-${CUDA_HOME}}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.PATH='${PATH}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH='${LD_LIBRARY_PATH:-}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_VISIBLE_DEVICES='${TRAIN_CUDA_VISIBLE_DEVICES}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.TMPDIR='${TMPDIR}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.TMP='${TMP}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.TEMP='${TEMP}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME='${HF_HOME}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.HUGGINGFACE_HUB_CACHE='${HUGGINGFACE_HUB_CACHE}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_DISABLE_XET='${HF_HUB_DISABLE_XET}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.XDG_CACHE_HOME='${XDG_CACHE_HOME}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR='${WANDB_DIR}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF='${PYTORCH_CUDA_ALLOC_CONF}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='${PYTHONPATH}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_SGLANG_WEIGHT_SYNC_METHOD='${VAGEN_SGLANG_WEIGHT_SYNC_METHOD}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_SGLANG_WEIGHT_SYNC_DIR='${VAGEN_SGLANG_WEIGHT_SYNC_DIR}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_SGLANG_WEIGHT_SYNC_LOAD_FORMAT='${VAGEN_SGLANG_WEIGHT_SYNC_LOAD_FORMAT}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_SGLANG_WEIGHT_SYNC_FLUSH_CACHE='${VAGEN_SGLANG_WEIGHT_SYNC_FLUSH_CACHE}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.TORCHDYNAMO_DISABLE='${TORCHDYNAMO_DISABLE}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.FLASHINFER_ENABLE_JIT='${FLASHINFER_ENABLE_JIT}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.FLASHINFER_JIT_WORKER_TIMEOUT='${FLASHINFER_JIT_WORKER_TIMEOUT}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_SGLANG_INIT_TIMEOUT='${VAGEN_SGLANG_INIT_TIMEOUT}'"
)
if [[ -n "${WANDB_MODE:-}" ]]; then
  RAY_RUNTIME_ENV_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE='${WANDB_MODE}'")
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  RAY_RUNTIME_ENV_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY='${WANDB_API_KEY}'")
fi

RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
RAY_LOG_ARCHIVE_DIR="${RAY_LOG_ARCHIVE_DIR:-${PROJECT_ROOT}/logs/ray/vast_${CONDITION}_$(date +%Y%m%d_%H%M%S)}"

archive_ray_logs() {
  local ray_logs_dir
  ray_logs_dir="$(find "${RAY_TMPDIR}" -maxdepth 4 -type d -path '*/session_latest/logs' 2>/dev/null | head -n 1 || true)"
  if [[ -n "${ray_logs_dir}" && -d "${ray_logs_dir}" ]]; then
    mkdir -p "${RAY_LOG_ARCHIVE_DIR}"
    cp -a "${ray_logs_dir}/." "${RAY_LOG_ARCHIVE_DIR}/"
    echo "Archived Ray logs to ${RAY_LOG_ARCHIVE_DIR}"
  fi
}

cleanup() {
  archive_ray_logs
  if [[ -n "${NAV_SERVER_PID:-}" ]]; then
    echo "Stopping navigation server (PID ${NAV_SERVER_PID})..."
    kill "${NAV_SERVER_PID}" 2>/dev/null || true
    wait "${NAV_SERVER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${XVFB_PID}" ]]; then
    kill "${XVFB_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "========================================================"
echo "CONDITION:          ${CONDITION}"
echo "EXPERIMENT:         ${EXPERIMENT_NAME}"
echo "PROJECT_ROOT:       ${PROJECT_ROOT}"
echo "RUN_ROOT:           ${RUN_ROOT}"
echo "TRAIN_DATA:         ${TRAIN_DATA}"
echo "VAL_DATA:           ${VAL_DATA}"
echo "NAV_GPU:            ${NAV_GPU}"
echo "TRAIN_GPU:          ${TRAIN_GPU}"
echo "NAV_CUDA_VISIBLE:   ${NAV_CUDA_VISIBLE_DEVICES}"
echo "TRAIN_CUDA_VISIBLE: ${TRAIN_CUDA_VISIBLE_DEVICES}"
echo "TRAINING_STEPS:     ${TRAINING_STEPS}"
echo "TRAIN_BATCH_SIZE:   ${TRAIN_BATCH_SIZE}"
echo "ROLLOUT_WORKERS:    ${ROLLOUT_NUM_WORKERS}"
echo "NAV_MAX_ENVS:       ${NAV_MAX_ENVS}"
echo "NAV_MAX_INFLIGHT:   ${NAV_MAX_INFLIGHT}"
echo "NAV_THREAD_POOL:    ${NAV_THREAD_POOL_SIZE}"
echo "NAV_ADMIT_TIMEOUT:  ${NAV_ADMIT_TIMEOUT}"
echo "NAV_SESSION_TIMEOUT:${NAV_SESSION_TIMEOUT}"
echo "CUDA_HOME:          ${CUDA_HOME:-unset}"
echo "VK_ICD_FILENAMES:   ${VK_ICD_FILENAMES:-unset}"
echo "HF_HOME:            ${HF_HOME}"
echo "HF_HUB_DISABLE_XET: ${HF_HUB_DISABLE_XET}"
echo "TMPDIR:             ${TMPDIR}"
echo "TMP:                ${TMP}"
echo "TEMP:               ${TEMP}"
echo "RAY_TMPDIR:         ${RAY_TMPDIR}"
echo "TRAINER_LOGGER:     ${TRAINER_LOGGER}"
echo "NAV_LOG_IMAGE:      ${NAV_LOG_IMAGE_ENABLE}"
echo "VAL_BEFORE_TRAIN:   ${NAV_VAL_BEFORE_TRAIN}"
echo "TEST_FREQ:          ${NAV_TEST_FREQ}"
echo "SAVE_FREQ:          ${NAV_SAVE_FREQ}"
echo "LOG_VAL_GENS:       ${NAV_LOG_VAL_GENERATIONS}"
echo "LENIENT_ACTION:     ${NAV_LENIENT_ACTION_PARSE}"
echo "EXAMPLE_COUNT:      ${NAV_EXAMPLE_COUNT:-unchanged}"
echo "ENV_RETRIES:        ${NAV_ENV_RETRIES:-unchanged}"
echo "ENV_TIMEOUT:        ${NAV_ENV_TIMEOUT:-unchanged}"
echo "ENV_MAX_DELAY:      ${NAV_ENV_MAX_DELAY:-unchanged}"
echo "TRAIN_N_ENVS:       ${NAV_TRAIN_N_ENVS_OVERRIDE:-unchanged}"
echo "VAL_N_ENVS:         ${NAV_VAL_N_ENVS_OVERRIDE:-unchanged}"
echo "USE_XVFB:           ${USE_XVFB:-0}"
echo "CONCAT_MULTI_TURN:  ${CONCAT_MULTI_TURN}"
echo "HISTORY_ARGS:       ${HISTORY_ARGS[*]:-none}"
echo "========================================================"
nvidia-smi || true
echo
echo "Disk space:"
df -h "${RUN_ROOT}" "${TMPDIR}" "${HF_HOME}" /tmp 2>/dev/null | awk 'NR == 1 || !seen[$6]++'
python - <<'PY'
import os
import tempfile

print("Python tempdir:", tempfile.gettempdir())
with tempfile.NamedTemporaryFile(prefix="vagen_tmp_check_", delete=True) as f:
    f.write(b"ok")
    f.flush()
    print("Python temp write OK:", f.name)
for name in ("TMPDIR", "TMP", "TEMP"):
    print(f"{name}={os.environ.get(name)}")
PY
if command -v vulkaninfo >/dev/null 2>&1; then
  vulkaninfo --summary | sed -n '1,80p' || true
fi

python - <<'PY'
import sys
import torch
print("Python:", sys.executable)
print("CUDA available:", torch.cuda.is_available(), "| device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}:", torch.cuda.get_device_name(i))
PY

python - <<'PY'
import sys
import transformers

required = ["AutoModelForVision2Seq", "Qwen2_5_VLForConditionalGeneration"]
missing = [name for name in required if not hasattr(transformers, name)]
print("Transformers:", transformers.__version__)
if missing:
    raise SystemExit(
        "ERROR: transformers is missing "
        + ", ".join(missing)
        + ". Install a Qwen2.5-VL-capable version, e.g. `pip install -U transformers==4.57.1`."
    )
PY

if [[ "${PREDOWNLOAD_SCENES:-0}" == "1" ]]; then
  echo "Pre-downloading AI2-THOR scenes with CUDA_VISIBLE_DEVICES=${NAV_CUDA_VISIBLE_DEVICES}..."
  CUDA_VISIBLE_DEVICES="${NAV_CUDA_VISIBLE_DEVICES}" python -m vagen.envs.navigation.pre_download_scenes --gpu 0
fi

NAV_SERVER_LOG="${PROJECT_ROOT}/logs/nav_server_vast_${CONDITION}_$(date +%Y%m%d_%H%M%S).log"
echo
echo "Starting navigation server with CUDA_VISIBLE_DEVICES=${NAV_CUDA_VISIBLE_DEVICES} (AI2-THOR gpu 0, max_envs=${NAV_MAX_ENVS})..."
CUDA_VISIBLE_DEVICES="${NAV_CUDA_VISIBLE_DEVICES}" \
  python -m vagen.envs.navigation.serve \
    --devices="[0]" \
    --max_envs="${NAV_MAX_ENVS}" \
    --max_inflight="${NAV_MAX_INFLIGHT}" \
    --admit_timeout="${NAV_ADMIT_TIMEOUT}" \
    --thread_pool_size="${NAV_THREAD_POOL_SIZE}" \
    --session_timeout="${NAV_SESSION_TIMEOUT}" \
    --port="${NAV_SERVER_PORT}" \
  > "${NAV_SERVER_LOG}" 2>&1 &
NAV_SERVER_PID=$!
echo "Navigation server PID=${NAV_SERVER_PID} log=${NAV_SERVER_LOG}"

echo -n "Waiting for /health"
READY=0
for i in $(seq 1 60); do
  if curl -sf "http://${NAV_SERVER_HOST}:${NAV_SERVER_PORT}/health" >/dev/null 2>&1; then
    READY=1
    echo " ready after $((i * 5))s"
    break
  fi
  if ! kill -0 "${NAV_SERVER_PID}" 2>/dev/null; then
    echo
    echo "ERROR: Navigation server crashed. Last 80 lines:"
    tail -80 "${NAV_SERVER_LOG}" || true
    exit 1
  fi
  echo -n "."
  sleep 5
done

if [[ "${READY}" -eq 0 ]]; then
  echo
  echo "ERROR: Server did not become healthy within 300s. Last 80 lines:"
  tail -80 "${NAV_SERVER_LOG}" || true
  exit 1
fi

if [[ "${VERIFY_SERVER_RESET:-1}" == "1" ]]; then
  echo "Running one remote reset smoke test. This is the check that actually starts Unity."
  CUDA_VISIBLE_DEVICES="${NAV_CUDA_VISIBLE_DEVICES}" \
    python -m vagen.envs.navigation.benchmark \
      --base_url "http://${NAV_SERVER_HOST}:${NAV_SERVER_PORT}" \
      --num_rounds 1 \
      --num_clients 1 \
      --num_steps 0 \
      --timeout 300
fi

EXPERIMENT_DIR="${RUN_ROOT}/checkpoints/${EXPERIMENT_NAME}"
mkdir -p "${EXPERIMENT_DIR}"

echo
echo "Starting training with CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES}: ${EXPERIMENT_NAME}"
set +e
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" \
PYTHONUNBUFFERED=1 \
  python -m vagen.main_ppo \
    --config-path="${PROJECT_ROOT}/vagen/configs" \
    --config-name="vagen_multiturn" \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.dataloader_num_workers=0 \
    data.max_prompt_length="${DATA_MAX_PROMPT}" \
    data.max_response_length="${DATA_MAX_RESPONSE}" \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path="Qwen/Qwen2.5-VL-3B-Instruct" \
    ++actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.checkpoint.save_contents=['hf_model'] \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.prompt_length="${ROLLOUT_PROMPT}" \
    actor_rollout_ref.rollout.response_length="${ROLLOUT_RESPONSE}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.sampling_backend=pytorch \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_NUM_WORKERS}" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CFG}" \
    actor_rollout_ref.rollout.disable_log_stats=False \
    trainer.concat_multi_turn="${CONCAT_MULTI_TURN}" \
    "${HISTORY_ARGS[@]}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.critic_warmup=0 \
    critic.enable=False \
    trainer.val_before_train="${NAV_VAL_BEFORE_TRAIN}" \
    trainer.resume_mode=disable \
    "trainer.logger=${TRAINER_LOGGER}" \
    "trainer.log_image.enable=${NAV_LOG_IMAGE_ENABLE}" \
    trainer.total_training_steps="${TRAINING_STEPS}" \
    trainer.save_freq="${NAV_SAVE_FREQ}" \
    trainer.test_freq="${NAV_TEST_FREQ}" \
    trainer.log_val_generations="${NAV_LOG_VAL_GENERATIONS}" \
    trainer.project_name="nav_window_ablation" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${EXPERIMENT_DIR}" \
    trainer.validation_data_dir="${RUN_ROOT}/validation/${EXPERIMENT_NAME}" \
    trainer.rollout_data_dir="${RUN_ROOT}/rollout/${EXPERIMENT_NAME}" \
    +ray_kwargs.ray_init.include_dashboard=False \
    +ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}" \
    +ray_kwargs.ray_init.object_store_memory=4294967296 \
    "+ray_kwargs.ray_init._temp_dir='${RAY_TMPDIR}'" \
    "${RAY_RUNTIME_ENV_ARGS[@]}" \
    2>&1 | tee "${EXPERIMENT_DIR}/train.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

echo
echo "Training finished with exit=${TRAIN_EXIT}."
exit "${TRAIN_EXIT}"
