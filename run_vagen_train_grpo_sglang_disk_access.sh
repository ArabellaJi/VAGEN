#!/bin/bash
# ACCESS / Delta variant of run_vagen_train_grpo_sglang_disk.sh.
# Training modes and Hydra arguments are kept aligned with the Quest script.
#
# Usage: sbatch [--gpus-per-node=N] /u/wji1/VAGEN/run_vagen_train_grpo_sglang_disk_access.sh <MODE>
# Default partition below is Delta A100. To use H200, override at submit time:
#   sbatch --partition=gpuH200x8 /u/wji1/VAGEN/run_vagen_train_grpo_sglang_disk_access.sh concat
#   sbatch --partition=gpuH200x8 --gpus-per-node=4 /u/wji1/VAGEN/run_vagen_train_grpo_sglang_disk_access.sh 4gpu
#   concat       (default): 1 GPU, 8 traj/step
#   window1:     1 GPU, no-concat, 1-turn history window
#   strict1:     1 GPU, concat, harder 3-8 step maps
#   ppo:         1 GPU, concat PPO baseline
#   ppo_window1: 1 GPU, no-concat PPO baseline
#   ppo_strict1: 1 GPU, concat PPO on strict1 maps
#   ppo_strict1_smoke: fast 1-GPU PPO sanity check (10 steps, no val)
#   text:        1 GPU, text rendering diagnostic
#   vision_fmt:  1 GPU, vision + format_penalty=-0.1
#   vision_fix:  1 GPU, vision + render_scale=4 + format_penalty=-0.1 + filter
#   2gpu:        2 GPUs, 64 traj/step  -> sbatch --gpus-per-node=2 ... 2gpu
#   4gpu:        4 GPUs, 256 traj/step -> sbatch --gpus-per-node=4 ... 4gpu
#
# Note: Training parameters below are intentionally unchanged; only the
# cluster/runtime setup is adapted here. The default Slurm partition here is
# Delta's quad-A100 queue, but the same script can be submitted to gpuH200x8.
#SBATCH --job-name=vagen_grpo_sokoban_3b
#SBATCH --account=bfea-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/u/wji1/VAGEN/logs/%x_%j.out
#SBATCH --error=/u/wji1/VAGEN/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wenlanji2026@u.northwestern.edu

set -eo pipefail

MODE="${1:-concat}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="/u/wji1/VAGEN"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  # On Delta, sbatch may execute a staged copy under /var/spool/slurmd/... .
  # Prefer the original submit directory when it is the repo root.
  if [ -d "${SLURM_SUBMIT_DIR}/vagen/configs" ]; then
    DEFAULT_PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  fi
fi
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
ACCESS_ACCOUNT="${ACCESS_ACCOUNT:-bfea-delta-gpu}"
ACCESS_PROJECT_CODE="${ACCESS_PROJECT_CODE:-${ACCESS_ACCOUNT%%-delta-gpu*}}"
if [ "${ACCESS_PROJECT_CODE}" = "${ACCESS_ACCOUNT}" ]; then
  ACCESS_PROJECT_CODE="${ACCESS_ACCOUNT%%-*}"
fi
ACCESS_WORK_ROOT="${ACCESS_WORK_ROOT:-/work/hdd/${ACCESS_PROJECT_CODE}/${USER}}"
RUN_ROOT="${RUN_ROOT:-${ACCESS_WORK_ROOT}/vagen_runs}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vagen_noflash}"
MODEL_REPO_ID="Qwen/Qwen2.5-VL-3B-Instruct"
REF_MODEL_PATH="${REF_MODEL_PATH:-${HF_MODEL_LOCAL_PATH:-${MODEL_REPO_ID}}}"
HF_HOME_DEFAULT="${HF_HOME_DEFAULT:-${ACCESS_WORK_ROOT}/hf_cache}"
MAX_AGENT_NUM_WORKERS=4
N_GPUS_PER_NODE=1
GPU_MEMORY_UTIL=0.4
FILTER_ARGS=()
VAL_BEFORE_TRAIN=True
TOTAL_TRAINING_STEPS=400
SAVE_FREQ=20
TEST_FREQ=20
LOG_VAL_GENERATIONS=5
DEFAULT_A100_PARTITION="${DEFAULT_A100_PARTITION:-gpuA100x4}"
DEFAULT_H200_PARTITION="${DEFAULT_H200_PARTITION:-gpuH200x8}"

if [ "${REF_MODEL_PATH}" = "${MODEL_REPO_ID}" ]; then
  EARLY_HF_SNAPSHOT_ROOT="${HF_HOME_DEFAULT}/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots"
  if [ -d "${EARLY_HF_SNAPSHOT_ROOT}" ]; then
    EARLY_LOCAL_MODEL_SNAPSHOT="$(find "${EARLY_HF_SNAPSHOT_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
    if [ -n "${EARLY_LOCAL_MODEL_SNAPSHOT}" ]; then
      REF_MODEL_PATH="${EARLY_LOCAL_MODEL_SNAPSHOT}"
    fi
  fi
fi

case "${MODE}" in
  concat)
    EXPERIMENT_NAME=sokoban_grpo_sglang_disk_3b
    TRAIN_FILE=examples/train/sokoban/train_sokoban_vision.yaml
    VAL_FILE=examples/train/sokoban/val_sokoban_vision.yaml
    DATA_MAX_PROMPT=1024
    DATA_MAX_RESPONSE=4096
    ROLLOUT_PROMPT=4096
    ROLLOUT_RESPONSE=2560
    MAX_BATCHED_TOKENS=8192
    TRAIN_BATCH_SIZE=2
    PPO_MINI_BATCH_SIZE=2
    ROLLOUT_N=4
    VAL_BATCH_SIZE=32
    ACTOR_USE_KL_LOSS=False
    ACTOR_KL_LOSS_COEF=0.0
    AGENT_CONFIG=agent.yaml
    CONCAT_MULTI_TURN=True
    LOG_IMAGE_ENABLE=False
    ADV_ESTIMATOR=grpo
    ADV_EXTRA_ARGS=(algorithm.norm_adv_by_std_in_grpo=True)
    CRITIC_ARGS=(critic.enable=False)
    HISTORY_ARGS=()
    ;;
  window1)
    EXPERIMENT_NAME=sokoban_grpo_sglang_disk_3b_window1
    TRAIN_FILE=examples/train/sokoban/train_sokoban_vision.yaml
    VAL_FILE=examples/train/sokoban/val_sokoban_vision.yaml
    DATA_MAX_PROMPT=2048
    DATA_MAX_RESPONSE=512
    ROLLOUT_PROMPT=2048
    ROLLOUT_RESPONSE=512
    MAX_BATCHED_TOKENS=4096
    TRAIN_BATCH_SIZE=4
    PPO_MINI_BATCH_SIZE=4
    ROLLOUT_N=4
    VAL_BATCH_SIZE=32
    ACTOR_USE_KL_LOSS=False
    ACTOR_KL_LOSS_COEF=0.0
    AGENT_CONFIG=agent_no_concat.yaml
    CONCAT_MULTI_TURN=False
    LOG_IMAGE_ENABLE=False
    ADV_ESTIMATOR=grpo
    ADV_EXTRA_ARGS=(algorithm.norm_adv_by_std_in_grpo=True)
    CRITIC_ARGS=(critic.enable=False)
    HISTORY_ARGS=(trainer.history_window_size=1 trainer.thumbnail_scale=0.25)
    ;;
  strict1)
    EXPERIMENT_NAME=sokoban_grpo_sglang_disk_3b_strict1
    TRAIN_FILE=examples/train/sokoban/train_sokoban_vision_strict1.yaml
    VAL_FILE=examples/train/sokoban/val_sokoban_vision_strict1.yaml
    DATA_MAX_PROMPT=1024
    DATA_MAX_RESPONSE=4096
    ROLLOUT_PROMPT=4096
    ROLLOUT_RESPONSE=2560
    MAX_BATCHED_TOKENS=8192
    TRAIN_BATCH_SIZE=2
    PPO_MINI_BATCH_SIZE=2
    ROLLOUT_N=4
    VAL_BATCH_SIZE=32
    ACTOR_USE_KL_LOSS=False
    ACTOR_KL_LOSS_COEF=0.0
    AGENT_CONFIG=agent.yaml
    CONCAT_MULTI_TURN=True
    LOG_IMAGE_ENABLE=True
    ADV_ESTIMATOR=grpo
    ADV_EXTRA_ARGS=(algorithm.norm_adv_by_std_in_grpo=True)
    CRITIC_ARGS=(critic.enable=False)
    HISTORY_ARGS=()
    ;;
  ppo)
    EXPERIMENT_NAME=sokoban_ppo_sglang_disk_3b
    TRAIN_FILE=examples/train/sokoban/train_sokoban_vision.yaml
    VAL_FILE=examples/train/sokoban/val_sokoban_vision.yaml
    DATA_MAX_PROMPT=1024
    DATA_MAX_RESPONSE=4096
    ROLLOUT_PROMPT=4096
    ROLLOUT_RESPONSE=2560
    MAX_BATCHED_TOKENS=8192
    TRAIN_BATCH_SIZE=2
    PPO_MINI_BATCH_SIZE=2
    ROLLOUT_N=1
    VAL_BATCH_SIZE=32
    ACTOR_USE_KL_LOSS=False
    ACTOR_KL_LOSS_COEF=0.0
    AGENT_CONFIG=agent.yaml
    CONCAT_MULTI_TURN=True
    LOG_IMAGE_ENABLE=False
    ADV_ESTIMATOR=gae
    ADV_EXTRA_ARGS=()
    CRITIC_ARGS=(
      critic.enable=True
      critic.optim.lr=1e-5
      ++critic.model.override_config.attn_implementation=eager
      critic.model.use_remove_padding=False
      critic.model.path="${REF_MODEL_PATH}"
      critic.model.enable_gradient_checkpointing=True
      critic.ppo_micro_batch_size_per_gpu=1
      critic.model.fsdp_config.param_offload=True
      critic.model.fsdp_config.optimizer_offload=True
    )
    HISTORY_ARGS=()
    ;;
  ppo_window1)
    EXPERIMENT_NAME=sokoban_ppo_sglang_disk_3b_window1
    TRAIN_FILE=examples/train/sokoban/train_sokoban_vision.yaml
    VAL_FILE=examples/train/sokoban/val_sokoban_vision.yaml
    DATA_MAX_PROMPT=2048
    DATA_MAX_RESPONSE=512
    ROLLOUT_PROMPT=2048
    ROLLOUT_RESPONSE=512
    MAX_BATCHED_TOKENS=4096
    TRAIN_BATCH_SIZE=4
    PPO_MINI_BATCH_SIZE=4
    ROLLOUT_N=1
    VAL_BATCH_SIZE=32
    ACTOR_USE_KL_LOSS=False
    ACTOR_KL_LOSS_COEF=0.0
    AGENT_CONFIG=agent_no_concat.yaml
    CONCAT_MULTI_TURN=False
    LOG_IMAGE_ENABLE=False
    ADV_ESTIMATOR=no_concat_gae
    ADV_EXTRA_ARGS=()
    CRITIC_ARGS=(
      critic.enable=True
      critic.optim.lr=1e-5
      ++critic.model.override_config.attn_implementation=eager
      critic.model.use_remove_padding=False
      critic.model.path="${REF_MODEL_PATH}"
      critic.model.enable_gradient_checkpointing=True
      critic.ppo_micro_batch_size_per_gpu=1
      critic.model.fsdp_config.param_offload=True
      critic.model.fsdp_config.optimizer_offload=True
    )
    HISTORY_ARGS=(trainer.history_window_size=1 trainer.thumbnail_scale=0.25)
    ;;
  ppo_strict1)
    EXPERIMENT_NAME=sokoban_ppo_sglang_disk_3b_strict1
    TRAIN_FILE=examples/train/sokoban/train_sokoban_vision_strict1.yaml
    VAL_FILE=examples/train/sokoban/val_sokoban_vision_strict1.yaml
    DATA_MAX_PROMPT=1024
    DATA_MAX_RESPONSE=4096
    ROLLOUT_PROMPT=4096
    ROLLOUT_RESPONSE=2560
    MAX_BATCHED_TOKENS=8192
    TRAIN_BATCH_SIZE=2
    PPO_MINI_BATCH_SIZE=2
    ROLLOUT_N=1
    VAL_BATCH_SIZE=32
    ACTOR_USE_KL_LOSS=False
    ACTOR_KL_LOSS_COEF=0.0
    AGENT_CONFIG=agent.yaml
    CONCAT_MULTI_TURN=True
    LOG_IMAGE_ENABLE=True
    ADV_ESTIMATOR=gae
    ADV_EXTRA_ARGS=()
    CRITIC_ARGS=(
      critic.enable=True
      critic.optim.lr=1e-5
      ++critic.model.override_config.attn_implementation=eager
      critic.model.use_remove_padding=False
      critic.model.path="${REF_MODEL_PATH}"
      critic.model.enable_gradient_checkpointing=True
      critic.ppo_micro_batch_size_per_gpu=1
      critic.model.fsdp_config.param_offload=True
      critic.model.fsdp_config.optimizer_offload=True
    )
    HISTORY_ARGS=()
    ;;
  ppo_strict1_smoke)
    EXPERIMENT_NAME=sokoban_ppo_sglang_disk_3b_strict1_smoke
    TRAIN_FILE=examples/train/sokoban/train_sokoban_vision_strict1.yaml
    VAL_FILE=examples/train/sokoban/val_sokoban_vision_strict1.yaml
    DATA_MAX_PROMPT=2048
    DATA_MAX_RESPONSE=512
    ROLLOUT_PROMPT=2048
    ROLLOUT_RESPONSE=512
    MAX_BATCHED_TOKENS=4096
    TRAIN_BATCH_SIZE=1
    PPO_MINI_BATCH_SIZE=1
    ROLLOUT_N=1
    VAL_BATCH_SIZE=32
    ACTOR_USE_KL_LOSS=False
    ACTOR_KL_LOSS_COEF=0.0
    AGENT_CONFIG=agent_no_concat.yaml
    CONCAT_MULTI_TURN=False
    LOG_IMAGE_ENABLE=False
    ADV_ESTIMATOR=no_concat_gae
    ADV_EXTRA_ARGS=()
    CRITIC_ARGS=(
      critic.enable=True
      critic.optim.lr=1e-5
      ++critic.model.override_config.attn_implementation=eager
      critic.model.use_remove_padding=False
      critic.model.path="${REF_MODEL_PATH}"
      critic.model.enable_gradient_checkpointing=True
      critic.ppo_micro_batch_size_per_gpu=1
      critic.model.fsdp_config.param_offload=True
      critic.model.fsdp_config.optimizer_offload=True
    )
    HISTORY_ARGS=(trainer.history_window_size=1 trainer.thumbnail_scale=0.25)
    VAL_BEFORE_TRAIN=False
    TOTAL_TRAINING_STEPS=10
    SAVE_FREQ=0
    TEST_FREQ=0
    LOG_VAL_GENERATIONS=0
    ;;
  text)
    EXPERIMENT_NAME=sokoban_grpo_sglang_disk_3b_text
    TRAIN_FILE=examples/train/sokoban/train_sokoban_wm_text.yaml
    VAL_FILE=examples/train/sokoban/val_sokoban_wm_text.yaml
    DATA_MAX_PROMPT=1024
    DATA_MAX_RESPONSE=4096
    ROLLOUT_PROMPT=4096
    ROLLOUT_RESPONSE=2560
    MAX_BATCHED_TOKENS=8192
    TRAIN_BATCH_SIZE=2
    PPO_MINI_BATCH_SIZE=2
    ROLLOUT_N=4
    VAL_BATCH_SIZE=32
    ACTOR_USE_KL_LOSS=False
    ACTOR_KL_LOSS_COEF=0.0
    AGENT_CONFIG=agent.yaml
    CONCAT_MULTI_TURN=True
    LOG_IMAGE_ENABLE=False
    ADV_ESTIMATOR=grpo
    ADV_EXTRA_ARGS=(algorithm.norm_adv_by_std_in_grpo=True)
    CRITIC_ARGS=(critic.enable=False)
    HISTORY_ARGS=()
    ;;
  vision_fmt)
    EXPERIMENT_NAME=sokoban_grpo_sglang_disk_3b_vision_fmt
    TRAIN_FILE=examples/train/sokoban/train_sokoban_vision_fmt.yaml
    VAL_FILE=examples/train/sokoban/val_sokoban_vision.yaml
    DATA_MAX_PROMPT=1024
    DATA_MAX_RESPONSE=4096
    ROLLOUT_PROMPT=4096
    ROLLOUT_RESPONSE=2560
    MAX_BATCHED_TOKENS=8192
    TRAIN_BATCH_SIZE=2
    PPO_MINI_BATCH_SIZE=2
    ROLLOUT_N=4
    VAL_BATCH_SIZE=32
    ACTOR_USE_KL_LOSS=False
    ACTOR_KL_LOSS_COEF=0.0
    AGENT_CONFIG=agent.yaml
    CONCAT_MULTI_TURN=True
    LOG_IMAGE_ENABLE=False
    ADV_ESTIMATOR=grpo
    ADV_EXTRA_ARGS=(algorithm.norm_adv_by_std_in_grpo=True)
    CRITIC_ARGS=(critic.enable=False)
    HISTORY_ARGS=()
    ;;
  vision_fix)
    EXPERIMENT_NAME=sokoban_grpo_sglang_disk_3b_vision_fix
    TRAIN_FILE=examples/train/sokoban/train_sokoban_vision_fix.yaml
    VAL_FILE=examples/train/sokoban/val_sokoban_vision.yaml
    DATA_MAX_PROMPT=1024
    DATA_MAX_RESPONSE=4096
    ROLLOUT_PROMPT=4096
    ROLLOUT_RESPONSE=2560
    MAX_BATCHED_TOKENS=8192
    TRAIN_BATCH_SIZE=2
    PPO_MINI_BATCH_SIZE=2
    ROLLOUT_N=4
    VAL_BATCH_SIZE=32
    ACTOR_USE_KL_LOSS=False
    ACTOR_KL_LOSS_COEF=0.0
    AGENT_CONFIG=agent.yaml
    CONCAT_MULTI_TURN=True
    LOG_IMAGE_ENABLE=False
    ADV_ESTIMATOR=grpo
    ADV_EXTRA_ARGS=(algorithm.norm_adv_by_std_in_grpo=True)
    CRITIC_ARGS=(critic.enable=False)
    HISTORY_ARGS=()
    FILTER_ARGS=(filter.enable=True)
    ;;
  2gpu)
    EXPERIMENT_NAME=sokoban_grpo_sglang_disk_3b_2gpu
    TRAIN_FILE=examples/train/sokoban/train_sokoban_vision.yaml
    VAL_FILE=examples/train/sokoban/val_sokoban_vision.yaml
    DATA_MAX_PROMPT=1024
    DATA_MAX_RESPONSE=4096
    ROLLOUT_PROMPT=4096
    ROLLOUT_RESPONSE=2560
    MAX_BATCHED_TOKENS=10000
    TRAIN_BATCH_SIZE=8
    PPO_MINI_BATCH_SIZE=8
    ROLLOUT_N=8
    VAL_BATCH_SIZE=32
    ACTOR_USE_KL_LOSS=False
    ACTOR_KL_LOSS_COEF=0.0
    AGENT_CONFIG=agent.yaml
    CONCAT_MULTI_TURN=True
    LOG_IMAGE_ENABLE=False
    ADV_ESTIMATOR=grpo
    ADV_EXTRA_ARGS=(algorithm.norm_adv_by_std_in_grpo=True)
    CRITIC_ARGS=(critic.enable=False)
    HISTORY_ARGS=()
    N_GPUS_PER_NODE=2
    GPU_MEMORY_UTIL=0.5
    ;;
  4gpu)
    EXPERIMENT_NAME=sokoban_grpo_sglang_disk_3b_4gpu
    TRAIN_FILE=examples/train/sokoban/train_sokoban_vision.yaml
    VAL_FILE=examples/train/sokoban/val_sokoban_vision.yaml
    DATA_MAX_PROMPT=1024
    DATA_MAX_RESPONSE=4096
    ROLLOUT_PROMPT=4096
    ROLLOUT_RESPONSE=2560
    MAX_BATCHED_TOKENS=10000
    TRAIN_BATCH_SIZE=32
    PPO_MINI_BATCH_SIZE=32
    ROLLOUT_N=8
    VAL_BATCH_SIZE=32
    ACTOR_USE_KL_LOSS=False
    ACTOR_KL_LOSS_COEF=0.0
    AGENT_CONFIG=agent.yaml
    CONCAT_MULTI_TURN=True
    LOG_IMAGE_ENABLE=False
    ADV_ESTIMATOR=grpo
    ADV_EXTRA_ARGS=(algorithm.norm_adv_by_std_in_grpo=True)
    CRITIC_ARGS=(critic.enable=False)
    HISTORY_ARGS=()
    N_GPUS_PER_NODE=4
    GPU_MEMORY_UTIL=0.6
    ;;
  *)
    echo "Unknown MODE: ${MODE}. Use 'concat', 'window1', 'strict1', 'ppo', 'ppo_window1', 'ppo_strict1', 'ppo_strict1_smoke', 'text', 'vision_fmt', 'vision_fix', '2gpu', or '4gpu'." >&2
    exit 1
    ;;
esac

ROLLOUT_PROMPT_COUNT=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
AGENT_NUM_WORKERS=${ROLLOUT_PROMPT_COUNT}
if [ "${AGENT_NUM_WORKERS}" -gt "${MAX_AGENT_NUM_WORKERS}" ]; then
  AGENT_NUM_WORKERS=${MAX_AGENT_NUM_WORKERS}
fi
while [ "${AGENT_NUM_WORKERS}" -gt 1 ] && [ $((ROLLOUT_PROMPT_COUNT % AGENT_NUM_WORKERS)) -ne 0 ]; do
  AGENT_NUM_WORKERS=$((AGENT_NUM_WORKERS - 1))
done

if [ ! -d "${PROJECT_ROOT}/vagen/configs" ]; then
  echo "PROJECT_ROOT does not look like the VAGEN repo root: ${PROJECT_ROOT}" >&2
  echo "SCRIPT_DIR=${SCRIPT_DIR}" >&2
  echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}" >&2
  echo "Set PROJECT_ROOT explicitly or submit with sbatch from the repo root." >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${RUN_ROOT}"
mkdir -p "${HF_HOME_DEFAULT}"
cd "${PROJECT_ROOT}"

load_first_available_module() {
  local candidate
  for candidate in "$@"; do
    if module load "${candidate}" >/dev/null 2>&1; then
      echo "Loaded module: ${candidate}"
      return 0
    fi
  done
  return 1
}

safe_module_reset() {
  local reset_status=0

  if ! command -v module >/dev/null 2>&1; then
    echo "module command is unavailable before environment setup; skipping module reset." >&2
    return 0
  fi

  module reset || reset_status=$?
  if [ "${reset_status}" -ne 0 ]; then
    echo "WARNING: module reset exited with status ${reset_status}; continuing with explicit module loads." >&2
  fi

  return 0
}

detect_access_gpu_class() {
  local partition="${1:-}"
  local gpu_name="${2:-}"

  case "${partition}" in
    gpuH200*) echo "h200"; return ;;
    gpuH100*) echo "h100"; return ;;
    gpuA100*) echo "a100"; return ;;
  esac

  case "${gpu_name}" in
    *H200*) echo "h200"; return ;;
    *H100*) echo "h100"; return ;;
    *A100*) echo "a100"; return ;;
  esac

  echo "unknown"
}

safe_module_reset

if ! command -v conda >/dev/null 2>&1; then
  load_first_available_module anaconda3_gpu pytorch-conda/2.8 || true
fi

if ! command -v nvcc >/dev/null 2>&1; then
  load_first_available_module cudatoolkit/25.3_12.8 cudatoolkit cuda/12.8 cuda/11.8.0 || true
fi

if [ -f ~/.bashrc ]; then
  source ~/.bashrc
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" >/dev/null 2>&1 || true
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is unavailable after module setup."
  echo "Loaded modules:"
  module list
  exit 1
fi

conda activate "${CONDA_ENV_NAME}"

set -u

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if [ "${N_GPUS_PER_NODE}" -le 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
  else
    DEFAULT_CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((N_GPUS_PER_NODE - 1)))"
    export CUDA_VISIBLE_DEVICES="${DEFAULT_CUDA_VISIBLE_DEVICES}"
  fi
else
  export CUDA_VISIBLE_DEVICES
fi

if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
else
  echo "No nvcc found after module load."
  echo "Loaded modules:"
  module list
  exit 1
fi

export CUDA_PATH="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}"

unset CC CXX CUDAHOSTCXX CMAKE_CUDA_HOST_COMPILER CUDA_NVCC_EXECUTABLE
GCC_BIN="$(command -v gcc)"
GXX_BIN="$(command -v g++)"
export CC="${GCC_BIN}"
export CXX="${GXX_BIN}"
export CUDAHOSTCXX="${GXX_BIN}"
export CMAKE_CUDA_HOST_COMPILER="${GXX_BIN}"

SLURM_ACTIVE_PARTITION="${SLURM_JOB_PARTITION:-${SLURM_PARTITION:-unknown}}"
DETECTED_GPU_NAME=""
if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
fi

ACCESS_GPU_CLASS="$(detect_access_gpu_class "${SLURM_ACTIVE_PARTITION}" "${DETECTED_GPU_NAME}")"
case "${ACCESS_GPU_CLASS}" in
  h200|h100)
    DEFAULT_TORCH_CUDA_ARCH_LIST=9.0
    ;;
  a100)
    DEFAULT_TORCH_CUDA_ARCH_LIST=8.0
    ;;
  *)
    DEFAULT_TORCH_CUDA_ARCH_LIST=8.0
    ;;
esac

unset PYTHONNOUSERSITE
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${DEFAULT_TORCH_CUDA_ARCH_LIST}}"
export HF_HOME="${HF_HOME:-${HF_HOME_DEFAULT}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

HF_TOKEN_VALUE="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
HF_TOKEN_PRESENT=false
if [ -n "${HF_TOKEN_VALUE}" ]; then
  export HF_TOKEN="${HF_TOKEN_VALUE}"
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN_VALUE}"
  HF_TOKEN_PRESENT=true
fi

if [ "${REF_MODEL_PATH}" = "${MODEL_REPO_ID}" ]; then
  HF_SNAPSHOT_ROOT="${HF_HOME}/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots"
  if [ -d "${HF_SNAPSHOT_ROOT}" ]; then
    LOCAL_MODEL_SNAPSHOT="$(find "${HF_SNAPSHOT_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
    if [ -n "${LOCAL_MODEL_SNAPSHOT}" ]; then
      REF_MODEL_PATH="${LOCAL_MODEL_SNAPSHOT}"
    fi
  fi
fi

export TORCHDYNAMO_DISABLE=1
export FLASHINFER_JIT_WORKER_TIMEOUT=60
export FLASHINFER_ENABLE_JIT=0
VAGEN_FORCE_EAGER_ATTN=1
export VAGEN_SGLANG_INIT_TIMEOUT=600

echo "MODE: ${MODE}"
echo "EXPERIMENT_NAME: ${EXPERIMENT_NAME}"
echo "MODEL_REPO_ID: ${MODEL_REPO_ID}"
echo "REF_MODEL_PATH: ${REF_MODEL_PATH}"
echo "TRAIN_FILE: ${TRAIN_FILE}"
echo "VAL_FILE: ${VAL_FILE}"
echo "TRAIN_BATCH_SIZE: ${TRAIN_BATCH_SIZE}"
echo "PPO_MINI_BATCH_SIZE: ${PPO_MINI_BATCH_SIZE}"
echo "ROLLOUT_N: ${ROLLOUT_N}"
echo "ROLLOUT_PROMPT_COUNT: ${ROLLOUT_PROMPT_COUNT}"
echo "AGENT_NUM_WORKERS: ${AGENT_NUM_WORKERS}"
echo "VAL_BATCH_SIZE: ${VAL_BATCH_SIZE}"
echo "ADV_ESTIMATOR: ${ADV_ESTIMATOR}"
echo "N_GPUS_PER_NODE: ${N_GPUS_PER_NODE}"
echo "GPU_MEMORY_UTIL: ${GPU_MEMORY_UTIL}"
echo "ACTOR_USE_KL_LOSS: ${ACTOR_USE_KL_LOSS}"
echo "ACTOR_KL_LOSS_COEF: ${ACTOR_KL_LOSS_COEF}"
echo "LOG_IMAGE_ENABLE: ${LOG_IMAGE_ENABLE}"
echo "VAL_BEFORE_TRAIN: ${VAL_BEFORE_TRAIN}"
echo "TOTAL_TRAINING_STEPS: ${TOTAL_TRAINING_STEPS}"
echo "SAVE_FREQ: ${SAVE_FREQ}"
echo "TEST_FREQ: ${TEST_FREQ}"
echo "LOG_VAL_GENERATIONS: ${LOG_VAL_GENERATIONS}"
echo "ACCESS_ACCOUNT: ${ACCESS_ACCOUNT}"
echo "ACCESS_PROJECT_CODE: ${ACCESS_PROJECT_CODE}"
echo "ACCESS_WORK_ROOT: ${ACCESS_WORK_ROOT}"
echo "DEFAULT_A100_PARTITION: ${DEFAULT_A100_PARTITION}"
echo "DEFAULT_H200_PARTITION: ${DEFAULT_H200_PARTITION}"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-unset}"
echo "SLURM_ACTIVE_PARTITION: ${SLURM_ACTIVE_PARTITION}"
echo "DETECTED_GPU_NAME: ${DETECTED_GPU_NAME:-unknown}"
echo "ACCESS_GPU_CLASS: ${ACCESS_GPU_CLASS}"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "RUN_ROOT: ${RUN_ROOT}"
echo "CONDA_ENV_NAME: ${CONDA_ENV_NAME}"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}"
echo "CONDA_PREFIX=${CONDA_PREFIX:-unset}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "CUDA_PATH=${CUDA_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "HF_HOME=${HF_HOME}"
echo "HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "HF_TOKEN_PRESENT=${HF_TOKEN_PRESENT}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "CC=${CC}"
echo "CXX=${CXX}"
echo "CUDAHOSTCXX=${CUDAHOSTCXX}"
echo "CMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}"

if [ "${ACCESS_GPU_CLASS}" = "unknown" ]; then
  echo "WARNING: could not infer GPU class from partition/device name; defaulting TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}" >&2
fi

case "${SLURM_ACTIVE_PARTITION}" in
  gpuH200*)
    echo "Running on Delta H200 partition. Training params are unchanged; only hardware and arch selection differ from A100."
    ;;
  gpuA100*)
    echo "Running on Delta A100 partition. Submit with --partition=${DEFAULT_H200_PARTITION} if you want the H200 version."
    ;;
esac

which python
command -v gcc || true
command -v g++ || true
gcc --version || true
g++ --version || true
command -v nvcc
nvcc --version
nvidia-smi || true

python - <<'PY'
import sys
print("runtime python:", sys.executable)
PY

python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

export JOB_TMP="${SLURM_TMPDIR:-/tmp/j${SLURM_JOB_ID}}"
export TMPDIR="${JOB_TMP}/t"
export RAY_TMPDIR="${JOB_TMP}/r"
SYNC_ROOT="${JOB_TMP}/sglang_sync"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" "${SYNC_ROOT}"

VAGEN_SGLANG_WEIGHT_SYNC_METHOD=disk
VAGEN_SGLANG_WEIGHT_SYNC_DIR="${SYNC_ROOT}"
VAGEN_SGLANG_WEIGHT_SYNC_LOAD_FORMAT=auto
VAGEN_SGLANG_WEIGHT_SYNC_FLUSH_CACHE=true

RAY_LOG_ARCHIVE_DIR="${PROJECT_ROOT}/logs/ray/${SLURM_JOB_ID}"
SYNC_ARCHIVE_DIR="${PROJECT_ROOT}/logs/sglang_sync/${SLURM_JOB_ID}"
RAY_NUM_CPUS="${SLURM_CPUS_PER_TASK:-4}"

archive_ray_logs() {
  local ray_logs_dir=""
  if [ -d "${RAY_TMPDIR}/session_latest/logs" ]; then
    ray_logs_dir="${RAY_TMPDIR}/session_latest/logs"
  else
    ray_logs_dir="$(find "${RAY_TMPDIR}" -maxdepth 4 -type d -path '*/session_latest/logs' 2>/dev/null | head -n 1 || true)"
  fi

  if [ -n "${ray_logs_dir}" ] && [ -d "${ray_logs_dir}" ]; then
    mkdir -p "${RAY_LOG_ARCHIVE_DIR}"
    cp -a "${ray_logs_dir}/." "${RAY_LOG_ARCHIVE_DIR}/"
    echo "Archived Ray logs to ${RAY_LOG_ARCHIVE_DIR}"
  else
    echo "No Ray logs found under ${RAY_TMPDIR}"
  fi

  if [ -d "${SYNC_ROOT}" ] && [ -n "$(ls -A "${SYNC_ROOT}" 2>/dev/null)" ]; then
    mkdir -p "${SYNC_ARCHIVE_DIR}"
    find "${SYNC_ROOT}" -maxdepth 4 \( -name "*.txt" -o -name "*.json" -o -name "*.safetensors.index.json" \) \
      -exec cp --parents {} "${SYNC_ARCHIVE_DIR}/" \; 2>/dev/null || true
    echo "Archived sglang sync metadata to ${SYNC_ARCHIVE_DIR}"
  else
    echo "No sglang sync files found under ${SYNC_ROOT}"
  fi
}

trap archive_ray_logs EXIT

PY=$(which python)
echo "Python: ${PY}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "RAY_TMPDIR: ${RAY_TMPDIR}"
echo "RAY_LOG_ARCHIVE_DIR: ${RAY_LOG_ARCHIVE_DIR}"
echo "RAY_NUM_CPUS: ${RAY_NUM_CPUS}"
echo "SYNC_ROOT: ${SYNC_ROOT}"
echo "VAGEN_SGLANG_WEIGHT_SYNC_METHOD: ${VAGEN_SGLANG_WEIGHT_SYNC_METHOD}"
echo "VAGEN_SGLANG_WEIGHT_SYNC_DIR: ${VAGEN_SGLANG_WEIGHT_SYNC_DIR}"
echo "VAGEN_SGLANG_WEIGHT_SYNC_LOAD_FORMAT: ${VAGEN_SGLANG_WEIGHT_SYNC_LOAD_FORMAT}"
echo "TORCHDYNAMO_DISABLE: ${TORCHDYNAMO_DISABLE}"
echo "FLASHINFER_JIT_WORKER_TIMEOUT: ${FLASHINFER_JIT_WORKER_TIMEOUT}"
echo "FLASHINFER_ENABLE_JIT: ${FLASHINFER_ENABLE_JIT}"
echo "VAGEN_FORCE_EAGER_ATTN: ${VAGEN_FORCE_EAGER_ATTN}"
echo "VAGEN_SGLANG_INIT_TIMEOUT: ${VAGEN_SGLANG_INIT_TIMEOUT}"

PYTHONUNBUFFERED=1 "${PY}" -m vagen.main_ppo \
  --config-path="${PWD}/vagen/configs" \
  --config-name="vagen_multiturn" \
  data.train_files="${PWD}/${TRAIN_FILE}" \
  data.val_files="${PWD}/${VAL_FILE}" \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.val_batch_size=${VAL_BATCH_SIZE} \
  data.dataloader_num_workers=0 \
  data.max_prompt_length=${DATA_MAX_PROMPT} \
  data.max_response_length=${DATA_MAX_RESPONSE} \
  algorithm.adv_estimator=${ADV_ESTIMATOR} \
  "${ADV_EXTRA_ARGS[@]}" \
  algorithm.kl_ctrl.kl_coef=0.0 \
  actor_rollout_ref.model.path="${REF_MODEL_PATH}" \
  ++actor_rollout_ref.model.override_config.attn_implementation=eager \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=${ACTOR_USE_KL_LOSS} \
  actor_rollout_ref.actor.kl_loss_coef=${ACTOR_KL_LOSS_COEF} \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.checkpoint.save_contents=['hf_model'] \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.prompt_length=${ROLLOUT_PROMPT} \
  actor_rollout_ref.rollout.response_length=${ROLLOUT_RESPONSE} \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_BATCHED_TOKENS} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTIL} \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.sampling_backend=pytorch \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS} \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${PWD}/vagen/configs/${AGENT_CONFIG}" \
  actor_rollout_ref.rollout.disable_log_stats=False \
  trainer.concat_multi_turn=${CONCAT_MULTI_TURN} \
  "${HISTORY_ARGS[@]}" \
  trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
  trainer.nnodes=1 \
  +ray_kwargs.ray_init.include_dashboard=False \
  +ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS} \
  +ray_kwargs.ray_init.object_store_memory=4294967296 \
  "+ray_kwargs.ray_init._temp_dir='${RAY_TMPDIR}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_HOME='${CUDA_HOME}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_PATH='${CUDA_PATH}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.PATH='${PATH}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH='${LD_LIBRARY_PATH}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME='${HF_HOME}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_CACHE='${HF_HUB_CACHE}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF='${PYTORCH_CUDA_ALLOC_CONF}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH='${PYTHONPATH}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.CC='${CC}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.CXX='${CXX}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.CUDAHOSTCXX='${CUDAHOSTCXX}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.CMAKE_CUDA_HOST_COMPILER='${CMAKE_CUDA_HOST_COMPILER}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_SGLANG_WEIGHT_SYNC_METHOD='${VAGEN_SGLANG_WEIGHT_SYNC_METHOD}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_SGLANG_WEIGHT_SYNC_DIR='${VAGEN_SGLANG_WEIGHT_SYNC_DIR}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_SGLANG_WEIGHT_SYNC_LOAD_FORMAT='${VAGEN_SGLANG_WEIGHT_SYNC_LOAD_FORMAT}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_SGLANG_WEIGHT_SYNC_FLUSH_CACHE='${VAGEN_SGLANG_WEIGHT_SYNC_FLUSH_CACHE}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.TORCHDYNAMO_DISABLE='${TORCHDYNAMO_DISABLE}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.FLASHINFER_JIT_WORKER_TIMEOUT='${FLASHINFER_JIT_WORKER_TIMEOUT}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.FLASHINFER_ENABLE_JIT='${FLASHINFER_ENABLE_JIT}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_FORCE_EAGER_ATTN='${VAGEN_FORCE_EAGER_ATTN}'" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_SGLANG_INIT_TIMEOUT='${VAGEN_SGLANG_INIT_TIMEOUT}'" \
  trainer.critic_warmup=0 \
  "${CRITIC_ARGS[@]}" \
  "${FILTER_ARGS[@]}" \
  'trainer.logger=[console,wandb]' \
  trainer.log_image.enable=${LOG_IMAGE_ENABLE} \
  trainer.resume_mode=disable \
  trainer.val_before_train=${VAL_BEFORE_TRAIN} \
  trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.test_freq=${TEST_FREQ} \
  trainer.project_name="vagen_sokoban" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${RUN_ROOT}/checkpoints/${EXPERIMENT_NAME}" \
  trainer.validation_data_dir="${RUN_ROOT}/validation/${EXPERIMENT_NAME}" \
  trainer.rollout_data_dir="${RUN_ROOT}/rollout/${EXPERIMENT_NAME}" \
  trainer.log_val_generations=${LOG_VAL_GENERATIONS}
