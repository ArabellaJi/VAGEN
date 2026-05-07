#!/bin/bash
# Quest (Northwestern) launcher for Crafter GRPO training.
# Mirrors run_vagen_train_grpo_sglang_disk.sh; adapted for Crafter.
#
# Usage:
#   sbatch --gres=gpu:h100:4 examples/train/crafter/submit_grpo_qwen25vl3b.sh full         # 4 GPU, 100 steps, ~24h
#   sbatch --gres=gpu:h100:2 --time=12:00:00 examples/train/crafter/submit_grpo_qwen25vl3b.sh 2gpu        # 2 GPU, 100 steps, ~12h (8 groups x 8 rollouts = 64/step)
#   sbatch --gres=gpu:h100:2 --time=14:00:00 examples/train/crafter/submit_grpo_qwen25vl3b.sh 2gpu_mem    # 2 GPU, 100 steps, ~14h (history=3, hires=1, thumbnail=0.25)
#   sbatch --gres=gpu:h100:2 --time=16:00:00 examples/train/crafter/submit_grpo_qwen25vl3b.sh 2gpu_hires3 # 2 GPU, 100 steps, ~16h (history=3, all hires)
#   sbatch --gres=gpu:h100:1 examples/train/crafter/submit_grpo_qwen25vl3b.sh 1gpu        # 1 GPU, 100 steps, ~6h  (no history)
#   sbatch --gres=gpu:h100:1 examples/train/crafter/submit_grpo_qwen25vl3b.sh 1gpu_mem    # 1 GPU, 100 steps, ~7h  (history=3, hires=1, thumbnail=0.25)
#   sbatch --gres=gpu:h100:1 examples/train/crafter/submit_grpo_qwen25vl3b.sh 1gpu_hires3 # 1 GPU, 100 steps, ~8h  (history=3, all hires)
#   sbatch --gres=gpu:h100:1 examples/train/crafter/submit_grpo_qwen25vl3b.sh smoke             # 1 GPU, 3 steps, pipeline check (no history)
#   sbatch --gres=gpu:h100:1 examples/train/crafter/submit_grpo_qwen25vl3b.sh smoke_mem         # 1 GPU, 3 steps, pipeline check (history=3, hires=1)
#   sbatch --gres=gpu:h100:1 examples/train/crafter/submit_grpo_qwen25vl3b.sh smoke_hires3      # 1 GPU, 3 steps, pipeline check (history=3, all hires)
#   sbatch --gres=gpu:h100:2 --time=01:00:00 examples/train/crafter/submit_grpo_qwen25vl3b.sh smoke_2gpu_mem    # 2 GPU, 3 steps, memory check for 2gpu_mem config
#   sbatch --gres=gpu:h100:2 --time=01:00:00 examples/train/crafter/submit_grpo_qwen25vl3b.sh smoke_2gpu_hires3 # 2 GPU, 3 steps, memory check for 2gpu_hires3 config
#
# Memory config: history_window_size=0 (no-concat, each turn independent)
#   Prompt per turn: sys (~650) + obs image (~950) ≈ 1700 tokens  → ROLLOUT_PROMPT=3000
#   Response per turn: 512 tokens                                  → ROLLOUT_RESPONSE=512
#   Data buffer: 15 turns × 512 = 7680                            → DATA_MAX_RESPONSE=8000
#
#SBATCH --job-name=crafter_grpo
#SBATCH --account=p33224
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/home/eiu4164/projects/VAGEN/logs/%x_%j.out
#SBATCH --error=/home/eiu4164/projects/VAGEN/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wenlanji2026@u.northwestern.edu

set -eo pipefail

MODE="${1:-full}"

PROJECT_ROOT=/home/eiu4164/projects/VAGEN
RUN_ROOT=/projects/p33224/vagen_runs
MODEL_REPO_ID="Qwen/Qwen2.5-VL-3B-Instruct"
REF_MODEL_PATH="${REF_MODEL_PATH:-${HF_MODEL_LOCAL_PATH:-${MODEL_REPO_ID}}}"
HF_HOME_DEFAULT=/projects/p33224/hf_cache

# ── Mode selection ─────────────────────────────────────────────────────────────
case "${MODE}" in
  smoke)
    # Pipeline sanity check: 1 GPU, tiny batch, 3 steps, no val, no history, ~30 min.
    # Submit with: sbatch --gres=gpu:h100:1 --time=01:00:00 ... smoke
    EXPERIMENT_NAME=crafter_grpo_3b_smoke
    TRAIN_FILE=examples/train/crafter/train_crafter_vision.yaml
    VAL_FILE=examples/train/crafter/val_crafter_vision.yaml
    DATA_MAX_PROMPT=3000
    DATA_MAX_RESPONSE=8000
    ROLLOUT_PROMPT=3000
    ROLLOUT_RESPONSE=512
    MAX_BATCHED_TOKENS=6000
    TRAIN_BATCH_SIZE=2
    PPO_MINI_BATCH_SIZE=2
    ROLLOUT_N=4
    VAL_BATCH_SIZE=4
    N_GPUS_PER_NODE=1
    GPU_MEMORY_UTIL=0.6
    AGENT_CONFIG=agent_no_concat.yaml
    CONCAT_MULTI_TURN=False
    HISTORY_WINDOW_SIZE=0
    HIRES_WINDOW_SIZE=0
    THUMBNAIL_SCALE=1.0
    VAL_BEFORE_TRAIN=False
    TOTAL_TRAINING_STEPS=3
    SAVE_FREQ=0
    TEST_FREQ=0
    LOG_VAL_GENERATIONS=0
    ;;
  smoke_mem)
    # Pipeline check: history=3, hires=1, thumbnail=0.25. ~45 min.
    # Submit with: sbatch --gres=gpu:h100:1 --time=01:00:00 ... smoke_mem
    EXPERIMENT_NAME=crafter_grpo_3b_smoke_mem
    TRAIN_FILE=examples/train/crafter/train_crafter_vision.yaml
    VAL_FILE=examples/train/crafter/val_crafter_vision.yaml
    DATA_MAX_PROMPT=3000
    DATA_MAX_RESPONSE=8000
    ROLLOUT_PROMPT=3000
    ROLLOUT_RESPONSE=512
    MAX_BATCHED_TOKENS=6000
    TRAIN_BATCH_SIZE=2
    PPO_MINI_BATCH_SIZE=2
    ROLLOUT_N=4
    VAL_BATCH_SIZE=4
    N_GPUS_PER_NODE=1
    GPU_MEMORY_UTIL=0.6
    AGENT_CONFIG=agent_no_concat.yaml
    CONCAT_MULTI_TURN=False
    HISTORY_WINDOW_SIZE=3
    HIRES_WINDOW_SIZE=1
    THUMBNAIL_SCALE=0.25
    VAL_BEFORE_TRAIN=False
    TOTAL_TRAINING_STEPS=3
    SAVE_FREQ=0
    TEST_FREQ=0
    LOG_VAL_GENERATIONS=0
    ;;
  smoke_hires3)
    # Pipeline check: history=3, all hires. ~45 min.
    # Submit with: sbatch --gres=gpu:h100:1 --time=01:00:00 ... smoke_hires3
    EXPERIMENT_NAME=crafter_grpo_3b_smoke_hires3
    TRAIN_FILE=examples/train/crafter/train_crafter_vision.yaml
    VAL_FILE=examples/train/crafter/val_crafter_vision.yaml
    DATA_MAX_PROMPT=3000
    DATA_MAX_RESPONSE=8000
    ROLLOUT_PROMPT=3000
    ROLLOUT_RESPONSE=512
    MAX_BATCHED_TOKENS=6000
    TRAIN_BATCH_SIZE=2
    PPO_MINI_BATCH_SIZE=2
    ROLLOUT_N=4
    VAL_BATCH_SIZE=4
    N_GPUS_PER_NODE=1
    GPU_MEMORY_UTIL=0.6
    AGENT_CONFIG=agent_no_concat.yaml
    CONCAT_MULTI_TURN=False
    HISTORY_WINDOW_SIZE=3
    HIRES_WINDOW_SIZE=3
    THUMBNAIL_SCALE=1.0
    VAL_BEFORE_TRAIN=False
    TOTAL_TRAINING_STEPS=3
    SAVE_FREQ=0
    TEST_FREQ=0
    LOG_VAL_GENERATIONS=0
    ;;
  smoke_2gpu_mem)
    # Memory check for 2gpu_mem: exact same config, 3 steps only. ~20 min.
    # Submit with: sbatch --gres=gpu:h100:2 --time=01:00:00 ... smoke_2gpu_mem
    EXPERIMENT_NAME=crafter_grpo_3b_smoke_2gpu_mem
    TRAIN_FILE=examples/train/crafter/train_crafter_vision.yaml
    VAL_FILE=examples/train/crafter/val_crafter_vision.yaml
    DATA_MAX_PROMPT=3000
    DATA_MAX_RESPONSE=4000
    ROLLOUT_PROMPT=3000
    ROLLOUT_RESPONSE=256
    MAX_BATCHED_TOKENS=10000
    TRAIN_BATCH_SIZE=4
    PPO_MINI_BATCH_SIZE=4
    ROLLOUT_N=8
    VAL_BATCH_SIZE=16
    N_GPUS_PER_NODE=2
    GPU_MEMORY_UTIL=0.6
    AGENT_CONFIG=agent_no_concat.yaml
    CONCAT_MULTI_TURN=False
    HISTORY_WINDOW_SIZE=3
    HIRES_WINDOW_SIZE=1
    THUMBNAIL_SCALE=0.25
    VAL_BEFORE_TRAIN=False
    TOTAL_TRAINING_STEPS=3
    SAVE_FREQ=0
    TEST_FREQ=0
    LOG_VAL_GENERATIONS=0
    KL_COEF=0.01
    USE_KL_LOSS=True
    KL_LOSS_COEF=0.01
    ENTROPY_COEFF=0.005
    FILTER_ENABLE=True
    FILTER_TOP_P=0.8
    RAY_OBJECT_STORE_MEMORY=8589934592   # 8 GB
    ;;
  smoke_2gpu_hires3)
    # Memory check for 2gpu_hires3: exact same config, 3 steps only. ~20 min.
    # Submit with: sbatch --gres=gpu:h100:2 --time=01:00:00 ... smoke_2gpu_hires3
    EXPERIMENT_NAME=crafter_grpo_3b_smoke_2gpu_hires3
    TRAIN_FILE=examples/train/crafter/train_crafter_vision.yaml
    VAL_FILE=examples/train/crafter/val_crafter_vision.yaml
    DATA_MAX_PROMPT=3000
    DATA_MAX_RESPONSE=4000
    ROLLOUT_PROMPT=3000
    ROLLOUT_RESPONSE=256
    MAX_BATCHED_TOKENS=8000
    TRAIN_BATCH_SIZE=2
    PPO_MINI_BATCH_SIZE=2
    ROLLOUT_N=8
    VAL_BATCH_SIZE=16
    N_GPUS_PER_NODE=2
    GPU_MEMORY_UTIL=0.6
    AGENT_CONFIG=agent_no_concat.yaml
    CONCAT_MULTI_TURN=False
    HISTORY_WINDOW_SIZE=3
    HIRES_WINDOW_SIZE=3
    THUMBNAIL_SCALE=1.0
    VAL_BEFORE_TRAIN=False
    TOTAL_TRAINING_STEPS=3
    SAVE_FREQ=0
    TEST_FREQ=0
    LOG_VAL_GENERATIONS=0
    KL_COEF=0.01
    USE_KL_LOSS=True
    KL_LOSS_COEF=0.01
    ENTROPY_COEFF=0.005
    FILTER_ENABLE=True
    FILTER_TOP_P=0.8
    RAY_OBJECT_STORE_MEMORY=17179869184  # 16 GB
    ;;
  2gpu)
    # Two-GPU training, no history, 100 steps, ~12h.
    # 8 groups x 8 rollouts = 64 rollouts/step (8x more GRPO signal than 1gpu).
    # Includes KL regularization, entropy bonus, and variance filter to prevent collapse.
    # Submit with: sbatch --gres=gpu:h100:2 --time=8:00:00 ... 2gpu
    EXPERIMENT_NAME=crafter_grpo_3b_2gpu
    TRAIN_FILE=examples/train/crafter/train_crafter_vision.yaml
    VAL_FILE=examples/train/crafter/val_crafter_vision.yaml
    DATA_MAX_PROMPT=3000
    DATA_MAX_RESPONSE=4000
    ROLLOUT_PROMPT=3000
    ROLLOUT_RESPONSE=256
    MAX_BATCHED_TOKENS=16000
    TRAIN_BATCH_SIZE=8
    PPO_MINI_BATCH_SIZE=8
    ROLLOUT_N=8
    VAL_BATCH_SIZE=16
    N_GPUS_PER_NODE=2
    GPU_MEMORY_UTIL=0.6
    AGENT_CONFIG=agent_no_concat.yaml
    CONCAT_MULTI_TURN=False
    HISTORY_WINDOW_SIZE=0
    HIRES_WINDOW_SIZE=0
    THUMBNAIL_SCALE=1.0
    VAL_BEFORE_TRAIN=True
    TOTAL_TRAINING_STEPS=100
    SAVE_FREQ=20
    TEST_FREQ=20
    LOG_VAL_GENERATIONS=5
    KL_COEF=0.01
    USE_KL_LOSS=True
    KL_LOSS_COEF=0.01
    ENTROPY_COEFF=0.005
    FILTER_ENABLE=True
    FILTER_TOP_P=0.8
    ;;
  2gpu_mem)
    # Two-GPU training, history=3, hires=1, thumbnail=0.25, 100 steps, ~14h.
    # TRAIN_BATCH_SIZE=4 (not 8) because hires+history image tensors are large.
    # Submit with: sbatch --gres=gpu:h100:2 --time=14:00:00 ... 2gpu_mem
    EXPERIMENT_NAME=crafter_grpo_3b_2gpu_mem
    TRAIN_FILE=examples/train/crafter/train_crafter_vision.yaml
    VAL_FILE=examples/train/crafter/val_crafter_vision.yaml
    DATA_MAX_PROMPT=3000
    DATA_MAX_RESPONSE=4000
    ROLLOUT_PROMPT=3000
    ROLLOUT_RESPONSE=256
    MAX_BATCHED_TOKENS=10000
    TRAIN_BATCH_SIZE=4
    PPO_MINI_BATCH_SIZE=4
    ROLLOUT_N=8
    VAL_BATCH_SIZE=16
    N_GPUS_PER_NODE=2
    GPU_MEMORY_UTIL=0.6
    AGENT_CONFIG=agent_no_concat.yaml
    CONCAT_MULTI_TURN=False
    HISTORY_WINDOW_SIZE=3
    HIRES_WINDOW_SIZE=1
    THUMBNAIL_SCALE=0.25
    VAL_BEFORE_TRAIN=True
    TOTAL_TRAINING_STEPS=100
    SAVE_FREQ=20
    TEST_FREQ=20
    LOG_VAL_GENERATIONS=5
    KL_COEF=0.01
    USE_KL_LOSS=True
    KL_LOSS_COEF=0.01
    ENTROPY_COEFF=0.005
    FILTER_ENABLE=True
    FILTER_TOP_P=0.8
    RAY_OBJECT_STORE_MEMORY=8589934592   # 8 GB: larger for hires image tensors
    ;;
  2gpu_hires3)
    # Two-GPU training, history=3, all hires, 100 steps, ~16h.
    # TRAIN_BATCH_SIZE=2 (not 8): full-res images × history=3 are too large for bigger batches.
    # Submit with: sbatch --gres=gpu:h100:2 --time=16:00:00 ... 2gpu_hires3
    EXPERIMENT_NAME=crafter_grpo_3b_2gpu_hires3
    TRAIN_FILE=examples/train/crafter/train_crafter_vision.yaml
    VAL_FILE=examples/train/crafter/val_crafter_vision.yaml
    DATA_MAX_PROMPT=3000
    DATA_MAX_RESPONSE=4000
    ROLLOUT_PROMPT=3000
    ROLLOUT_RESPONSE=256
    MAX_BATCHED_TOKENS=8000
    TRAIN_BATCH_SIZE=4
    PPO_MINI_BATCH_SIZE=4
    ROLLOUT_N=8
    VAL_BATCH_SIZE=16
    N_GPUS_PER_NODE=2
    GPU_MEMORY_UTIL=0.6
    AGENT_CONFIG=agent_no_concat.yaml
    CONCAT_MULTI_TURN=False
    HISTORY_WINDOW_SIZE=3
    HIRES_WINDOW_SIZE=3
    THUMBNAIL_SCALE=1.0
    VAL_BEFORE_TRAIN=True
    TOTAL_TRAINING_STEPS=100
    SAVE_FREQ=20
    TEST_FREQ=20
    LOG_VAL_GENERATIONS=5
    KL_COEF=0.01
    USE_KL_LOSS=True
    KL_LOSS_COEF=0.01
    ENTROPY_COEFF=0.005
    FILTER_ENABLE=True
    FILTER_TOP_P=0.8
    RAY_OBJECT_STORE_MEMORY=17179869184  # 16 GB: full-res history images are very large
    ;;
  1gpu)
    # Single-GPU training, no history, 100 steps, ~6h.
    # Submit with: sbatch --gres=gpu:h100:1 --time=08:00:00 ... 1gpu
    EXPERIMENT_NAME=crafter_grpo_3b_1gpu
    TRAIN_FILE=examples/train/crafter/train_crafter_vision.yaml
    VAL_FILE=examples/train/crafter/val_crafter_vision.yaml
    DATA_MAX_PROMPT=3000
    DATA_MAX_RESPONSE=8000
    ROLLOUT_PROMPT=3000
    ROLLOUT_RESPONSE=512
    MAX_BATCHED_TOKENS=6000
    TRAIN_BATCH_SIZE=2
    PPO_MINI_BATCH_SIZE=2
    ROLLOUT_N=4
    VAL_BATCH_SIZE=8
    N_GPUS_PER_NODE=1
    GPU_MEMORY_UTIL=0.5
    AGENT_CONFIG=agent_no_concat.yaml
    CONCAT_MULTI_TURN=False
    HISTORY_WINDOW_SIZE=0
    HIRES_WINDOW_SIZE=0
    THUMBNAIL_SCALE=1.0
    VAL_BEFORE_TRAIN=True
    TOTAL_TRAINING_STEPS=100
    SAVE_FREQ=20
    TEST_FREQ=20
    LOG_VAL_GENERATIONS=5
    ;;
  1gpu_mem)
    # 1 GPU, history=3, hires=1, thumbnail=0.25, 100 steps, ~7h.
    # Submit with: sbatch --gres=gpu:h100:1 --time=09:00:00 ... 1gpu_mem
    EXPERIMENT_NAME=crafter_grpo_3b_1gpu_mem
    TRAIN_FILE=examples/train/crafter/train_crafter_vision.yaml
    VAL_FILE=examples/train/crafter/val_crafter_vision.yaml
    DATA_MAX_PROMPT=3000
    DATA_MAX_RESPONSE=8000
    ROLLOUT_PROMPT=3000
    ROLLOUT_RESPONSE=256
    MAX_BATCHED_TOKENS=6000
    TRAIN_BATCH_SIZE=2
    PPO_MINI_BATCH_SIZE=2
    ROLLOUT_N=8
    VAL_BATCH_SIZE=8
    N_GPUS_PER_NODE=1
    GPU_MEMORY_UTIL=0.5
    AGENT_CONFIG=agent_no_concat.yaml
    CONCAT_MULTI_TURN=False
    HISTORY_WINDOW_SIZE=3
    HIRES_WINDOW_SIZE=1
    THUMBNAIL_SCALE=0.25
    VAL_BEFORE_TRAIN=True
    TOTAL_TRAINING_STEPS=100
    SAVE_FREQ=20
    TEST_FREQ=20
    LOG_VAL_GENERATIONS=5
    KL_COEF=0.01
    USE_KL_LOSS=True
    KL_LOSS_COEF=0.01
    ENTROPY_COEFF=0.005
    FILTER_ENABLE=True
    FILTER_TOP_P=0.8
    ;;
  1gpu_hires3)
    # 1 GPU, history=3, all hires, 100 steps, ~8h.
    # Submit with: sbatch --gres=gpu:h100:1 --time=10:00:00 ... 1gpu_hires3
    EXPERIMENT_NAME=crafter_grpo_3b_1gpu_hires3
    TRAIN_FILE=examples/train/crafter/train_crafter_vision.yaml
    VAL_FILE=examples/train/crafter/val_crafter_vision.yaml
    DATA_MAX_PROMPT=3000
    DATA_MAX_RESPONSE=8000
    ROLLOUT_PROMPT=3000
    ROLLOUT_RESPONSE=512
    MAX_BATCHED_TOKENS=6000
    TRAIN_BATCH_SIZE=2
    PPO_MINI_BATCH_SIZE=2
    ROLLOUT_N=4
    VAL_BATCH_SIZE=8
    N_GPUS_PER_NODE=1
    GPU_MEMORY_UTIL=0.5
    AGENT_CONFIG=agent_no_concat.yaml
    CONCAT_MULTI_TURN=False
    HISTORY_WINDOW_SIZE=3
    HIRES_WINDOW_SIZE=3
    THUMBNAIL_SCALE=1.0
    VAL_BEFORE_TRAIN=True
    TOTAL_TRAINING_STEPS=100
    SAVE_FREQ=20
    TEST_FREQ=20
    LOG_VAL_GENERATIONS=5
    KL_COEF=0.01
    USE_KL_LOSS=True
    KL_LOSS_COEF=0.01
    ENTROPY_COEFF=0.005
    FILTER_ENABLE=True
    FILTER_TOP_P=0.8
    ;;
  full)
    # Full training: 4 GPU, no history, 100 steps, ~24 hours.
    # Submit with: sbatch --gres=gpu:h100:4 --time=24:00:00 ... full
    EXPERIMENT_NAME=crafter_grpo_3b
    TRAIN_FILE=examples/train/crafter/train_crafter_vision.yaml
    VAL_FILE=examples/train/crafter/val_crafter_vision.yaml
    DATA_MAX_PROMPT=3000
    DATA_MAX_RESPONSE=8000
    ROLLOUT_PROMPT=3000
    ROLLOUT_RESPONSE=512
    MAX_BATCHED_TOKENS=10000
    TRAIN_BATCH_SIZE=32
    PPO_MINI_BATCH_SIZE=32
    ROLLOUT_N=8
    VAL_BATCH_SIZE=32
    N_GPUS_PER_NODE=4
    GPU_MEMORY_UTIL=0.6
    AGENT_CONFIG=agent_no_concat.yaml
    CONCAT_MULTI_TURN=False
    HISTORY_WINDOW_SIZE=0
    HIRES_WINDOW_SIZE=0
    THUMBNAIL_SCALE=1.0
    VAL_BEFORE_TRAIN=True
    TOTAL_TRAINING_STEPS=100
    SAVE_FREQ=20
    TEST_FREQ=20
    LOG_VAL_GENERATIONS=5
    ;;
  *)
    echo "Unknown MODE: ${MODE}. Use 'smoke', 'smoke_mem', 'smoke_hires3', 'smoke_2gpu_mem', 'smoke_2gpu_hires3', '2gpu', '2gpu_mem', '2gpu_hires3', '1gpu', '1gpu_mem', '1gpu_hires3', or 'full'." >&2; exit 1
    ;;
esac
# Regularization defaults — override in case statements for specific modes (e.g. 2gpu)
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-4294967296}"  # 4 GB default
KL_COEF="${KL_COEF:-0.0}"
USE_KL_LOSS="${USE_KL_LOSS:-False}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.0}"
FILTER_ENABLE="${FILTER_ENABLE:-False}"
FILTER_TOP_P="${FILTER_TOP_P:-0.8}"
CONCAT_MULTI_TURN=False
ADV_ESTIMATOR=grpo
MAX_AGENT_NUM_WORKERS=$((N_GPUS_PER_NODE * 4))
VAGEN_SGLANG_INIT_TIMEOUT=1800
RAY_NUM_CPUS=$((N_GPUS_PER_NODE * 8))

# Resolve local HF snapshot if already cached
if [ "${REF_MODEL_PATH}" = "${MODEL_REPO_ID}" ]; then
  EARLY_HF_SNAPSHOT_ROOT="${HF_HOME_DEFAULT}/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots"
  if [ -d "${EARLY_HF_SNAPSHOT_ROOT}" ]; then
    EARLY_LOCAL="$(find "${EARLY_HF_SNAPSHOT_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
    [ -n "${EARLY_LOCAL}" ] && REF_MODEL_PATH="${EARLY_LOCAL}"
  fi
fi

# AGENT_NUM_WORKERS must evenly divide ROLLOUT_PROMPT_COUNT
ROLLOUT_PROMPT_COUNT=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
AGENT_NUM_WORKERS=${ROLLOUT_PROMPT_COUNT}
[ "${AGENT_NUM_WORKERS}" -gt "${MAX_AGENT_NUM_WORKERS}" ] && AGENT_NUM_WORKERS=${MAX_AGENT_NUM_WORKERS}
while [ "${AGENT_NUM_WORKERS}" -gt 1 ] && [ $((ROLLOUT_PROMPT_COUNT % AGENT_NUM_WORKERS)) -ne 0 ]; do
  AGENT_NUM_WORKERS=$((AGENT_NUM_WORKERS - 1))
done

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${RUN_ROOT}"
cd "${PROJECT_ROOT}"

# ── Modules ────────────────────────────────────────────────────────────────────
module purge
module load python-miniconda3/4.10.3
module load gcc/11.2.0
module load cuda/12.6.2-gcc-12.4.0

source ~/.bashrc
conda activate vagen_noflash

set -u

# Build CUDA_VISIBLE_DEVICES dynamically if not set externally
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  _cvd=""; for _i in $(seq 0 $((N_GPUS_PER_NODE - 1))); do _cvd="${_cvd:+${_cvd},}${_i}"; done
  export CUDA_VISIBLE_DEVICES="${_cvd}"
fi

if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
else
  echo "No nvcc found after module load." >&2; module list; exit 1
fi

export CUDA_PATH="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}"

unset CC CXX CUDAHOSTCXX CMAKE_CUDA_HOST_COMPILER CUDA_NVCC_EXECUTABLE
GCC_BIN="$(command -v gcc)"; GXX_BIN="$(command -v g++)"
export CC="${GCC_BIN}" CXX="${GXX_BIN}" CUDAHOSTCXX="${GXX_BIN}" CMAKE_CUDA_HOST_COMPILER="${GXX_BIN}"

unset PYTHONNOUSERSITE
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export TORCH_CUDA_ARCH_LIST="9.0"
export HF_HOME=/projects/p33224/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export TORCHDYNAMO_DISABLE=1
export FLASHINFER_JIT_WORKER_TIMEOUT=60
export FLASHINFER_ENABLE_JIT=0
VAGEN_FORCE_EAGER_ATTN=1
export VAGEN_SGLANG_INIT_TIMEOUT

# Resolve local snapshot again (after HF_HOME is set)
if [ "${REF_MODEL_PATH}" = "${MODEL_REPO_ID}" ]; then
  HF_SNAPSHOT_ROOT="${HF_HOME}/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots"
  if [ -d "${HF_SNAPSHOT_ROOT}" ]; then
    LOCAL="$(find "${HF_SNAPSHOT_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
    [ -n "${LOCAL}" ] && REF_MODEL_PATH="${LOCAL}"
  fi
fi

# ── Ray / SGLang temp dirs ──────────────────────────────────────────────────────
export JOB_TMP="/tmp/j${SLURM_JOB_ID}"
export TMPDIR="${JOB_TMP}/t"
export RAY_TMPDIR="${JOB_TMP}/r"
SYNC_ROOT="${JOB_TMP}/sglang_sync"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" "${SYNC_ROOT}"

VAGEN_SGLANG_WEIGHT_SYNC_METHOD=disk
VAGEN_SGLANG_WEIGHT_SYNC_DIR="${SYNC_ROOT}"
VAGEN_SGLANG_WEIGHT_SYNC_LOAD_FORMAT=auto
VAGEN_SGLANG_WEIGHT_SYNC_FLUSH_CACHE=true

RAY_LOG_ARCHIVE_DIR="${PROJECT_ROOT}/logs/ray/${SLURM_JOB_ID}"

archive_ray_logs() {
  local d
  d="$(find "${RAY_TMPDIR}" -maxdepth 4 -type d -path '*/session_latest/logs' 2>/dev/null | head -n 1 || true)"
  if [ -n "${d}" ] && [ -d "${d}" ]; then
    mkdir -p "${RAY_LOG_ARCHIVE_DIR}"; cp -a "${d}/." "${RAY_LOG_ARCHIVE_DIR}/"
    echo "Archived Ray logs to ${RAY_LOG_ARCHIVE_DIR}"
  fi
}
trap archive_ray_logs EXIT

# ── Info dump ───────────────────────────────────────────────────────────────────
echo "EXPERIMENT_NAME: ${EXPERIMENT_NAME}"
echo "REF_MODEL_PATH:  ${REF_MODEL_PATH}"
echo "N_GPUS:          ${N_GPUS_PER_NODE}  ROLLOUT_N=${ROLLOUT_N}  TRAIN_BATCH=${TRAIN_BATCH_SIZE}"
echo "AGENT_NUM_WORKERS: ${AGENT_NUM_WORKERS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
nvcc --version; nvidia-smi || true
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
PY

# ── Training ────────────────────────────────────────────────────────────────────
PYTHONUNBUFFERED=1 python -m vagen.main_ppo \
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
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.kl_ctrl.kl_coef=${KL_COEF} \
  actor_rollout_ref.model.path="${REF_MODEL_PATH}" \
  ++actor_rollout_ref.model.override_config.attn_implementation=eager \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
  actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.checkpoint.save_contents=['hf_model'] \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.use_torch_compile=False \
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
  trainer.history_window_size=${HISTORY_WINDOW_SIZE} \
  trainer.hires_window_size=${HIRES_WINDOW_SIZE} \
  trainer.thumbnail_scale=${THUMBNAIL_SCALE} \
  trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
  trainer.nnodes=1 \
  critic.enable=False \
  +ray_kwargs.ray_init.include_dashboard=False \
  +ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS} \
  +ray_kwargs.ray_init.object_store_memory=${RAY_OBJECT_STORE_MEMORY} \
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
  'trainer.logger=[console,wandb]' \
  trainer.log_image.enable=False \
  trainer.resume_mode=disable \
  trainer.val_before_train=${VAL_BEFORE_TRAIN} \
  trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.test_freq=${TEST_FREQ} \
  trainer.project_name="vagen_crafter" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${RUN_ROOT}/checkpoints/${EXPERIMENT_NAME}" \
  trainer.validation_data_dir="${RUN_ROOT}/validation/${EXPERIMENT_NAME}" \
  trainer.rollout_data_dir="${RUN_ROOT}/rollout/${EXPERIMENT_NAME}" \
  trainer.log_val_generations=${LOG_VAL_GENERATIONS} \
  filter.enable=${FILTER_ENABLE} \
  filter.name=reward_variance_top_p \
  "filter.filter_kwargs.top_p=${FILTER_TOP_P}"
