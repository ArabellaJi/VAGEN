#!/bin/bash
# Usage: sbatch [--gres=gpu:h100:N] run_memory_bins_train_grpo.sh <MODE>
#
#   smoke:         quick sanity check — 8 envs, 30 steps, no val
#   no_memory:     1 GPU, no history — agent sees current obs only (baseline)
#   window3_thumb: 1 GPU, 3-turn window + thumbnail compression (proposed method)
#   window3:       1 GPU, 3-turn window, full-res history (ablation: does thumbnail hurt?)
#   full_thumb:    1 GPU, unlimited history with thumbnail compression
#   full_memory:   1 GPU, unlimited history at full resolution (WARNING: see note below)
#   4gpu:          4 GPUs, window3_thumb scaled up — 256 traj/step
#
# Token budget reference (Qwen2.5-VL, 14px patch, 2x2 merge):
#   384x384 full image  →  ~182 visual tokens   (cell_size=48, 8×48=384 — same as sokoban)
#   96x96  thumbnail    →  ~9   visual tokens   (thumbnail_scale=0.25)
#   system prompt       →  ~400 tokens
#   per-turn obs text   →  ~50  tokens
#   response cap        →  512  tokens (response_length_per_turn in yaml)
#
# Why 384 not 512: 512×512 → 325 tokens caused system-RAM OOM on Quest (128G) during
# full-scale rollout (8 traj × 60 turns × image tensors ≈ 12GB, combined with FSDP
# optimizer states). 384×384 → 182 tokens is the proven sokoban working configuration.
#
# Per-turn context budget at steady state (worst-case):
#   no_memory:     sys(400) + obs_text(50) + img(182)                        ≈  632  → 1024
#   window3_thumb: sys(400) + 3×(text+thumb+resp)(~571) + current(232)       ≈ 2345  → 4096
#   window3:       sys(400) + 3×(text+img+resp)(744)   + current(232)        ≈ 2864  → 4096
#   full_thumb:    grows with episode; response tokens (512/turn) dominate;
#                  truncation expected beyond ~13 turns (future fix: strip <think> from old turns)
#   full_memory:   sys(400) + (T-1)×744 + current(232); turn 10 ≈ 7340,
#                  turn 11 ≈ 8084 → TRUNCATED at ROLLOUT_PROMPT=8192 beyond ~11 turns.

#SBATCH --job-name=vagen_grpo_membins_3b
#SBATCH --account=p33224
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=/home/eiu4164/projects/VAGEN/logs/%x_%j.out
#SBATCH --error=/home/eiu4164/projects/VAGEN/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wenlanji2026@u.northwestern.edu

set -eo pipefail

MODE="${1:-window3_thumb}"

PROJECT_ROOT=/home/eiu4164/projects/VAGEN
RUN_ROOT=/projects/p33224/vagen_runs
MODEL_REPO_ID="Qwen/Qwen2.5-VL-3B-Instruct"
REF_MODEL_PATH="${REF_MODEL_PATH:-${HF_MODEL_LOCAL_PATH:-${MODEL_REPO_ID}}}"
HF_HOME_DEFAULT=/projects/p33224/hf_cache
MAX_AGENT_NUM_WORKERS=4
N_GPUS_PER_NODE=1
GPU_MEMORY_UTIL=0.4
FILTER_ARGS=()
EXTRA_ARGS=()
VAL_BEFORE_TRAIN=True
TOTAL_TRAINING_STEPS=400
SAVE_FREQ=20
TEST_FREQ=20
LOG_VAL_GENERATIONS=5

# Resolve local HF snapshot early so all downstream settings see the same path.
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
  smoke)
    # Full-pipeline smoke test: 8 envs, 30 max_steps, window=1 + thumbnail.
    # Validates every 5 training steps so W&B shows action_is_valid, success rate,
    # and logged validation generations — same code paths as real experiments.
    # Expected runtime: ~15–20 min (val before train + 2 mid-run vals + final val).
    EXPERIMENT_NAME=membins_grpo_smoke
    TRAIN_FILE=examples/train/memory_bins/train_memory_bins_smoke.yaml
    VAL_FILE=examples/train/memory_bins/val_memory_bins_smoke.yaml
    DATA_MAX_PROMPT=2048
    DATA_MAX_RESPONSE=512
    ROLLOUT_PROMPT=2048
    ROLLOUT_RESPONSE=512
    MAX_BATCHED_TOKENS=4096
    TRAIN_BATCH_SIZE=1
    PPO_MINI_BATCH_SIZE=1
    ROLLOUT_N=1
    VAL_BATCH_SIZE=8
    ACTOR_USE_KL_LOSS=False
    ACTOR_KL_LOSS_COEF=0.0
    AGENT_CONFIG=agent_no_concat.yaml
    CONCAT_MULTI_TURN=False
    LOG_IMAGE_ENABLE=True
    ADV_ESTIMATOR=grpo
    ADV_EXTRA_ARGS=(algorithm.norm_adv_by_std_in_grpo=True)
    CRITIC_ARGS=(critic.enable=False)
    # window=1 to confirm thumbnail path runs; not the real ablation
    HISTORY_ARGS=(trainer.history_window_size=1 trainer.thumbnail_scale=0.25)
    VAL_BEFORE_TRAIN=True
    TOTAL_TRAINING_STEPS=10
    SAVE_FREQ=0
    TEST_FREQ=5
    LOG_VAL_GENERATIONS=4
    ;;

  no_memory)
    # Baseline: agent sees only the current observation — no conversation history.
    # Any score above chance must come from single-step visual reasoning alone.
    # Compare against window3_thumb to measure the value of memory.
    #
    # Context: sys(400) + obs_text(50) + img(325) ≈ 775 tokens per turn.
    EXPERIMENT_NAME=membins_grpo_no_memory
    TRAIN_FILE=examples/train/memory_bins/train_memory_bins.yaml
    VAL_FILE=examples/train/memory_bins/val_memory_bins.yaml
    DATA_MAX_PROMPT=1024
    DATA_MAX_RESPONSE=512
    ROLLOUT_PROMPT=1024
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
    HISTORY_ARGS=()
    ;;

  window3_thumb)
    # Proposed method: 3-turn sliding window with thumbnail compression on the
    # 2 older turns (or however the trainer implements window+thumbnail).
    #
    # Context: sys(400) + 3 history turns×(obs_text(50)+thumb(20)+response(512))
    #          + current_obs(375) ≈ 2521 tokens per turn.
    # ROLLOUT_PROMPT=4096 gives ~60% headroom for longer think responses.
    EXPERIMENT_NAME=membins_grpo_window3_thumb
    TRAIN_FILE=examples/train/memory_bins/train_memory_bins.yaml
    VAL_FILE=examples/train/memory_bins/val_memory_bins.yaml
    DATA_MAX_PROMPT=4096
    DATA_MAX_RESPONSE=512
    ROLLOUT_PROMPT=4096
    ROLLOUT_RESPONSE=512
    MAX_BATCHED_TOKENS=8192
    TRAIN_BATCH_SIZE=2
    PPO_MINI_BATCH_SIZE=2
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
    HISTORY_ARGS=(trainer.history_window_size=3 trainer.thumbnail_scale=0.25)
    ;;

  window3)
    # Ablation: 3-turn sliding window at full resolution (no thumbnail compression).
    # Isolates whether the thumbnail is the limiting factor vs window3_thumb.
    # If window3 ≈ window3_thumb → thumbnail is lossless for this task (solid-color bins).
    # If window3 > window3_thumb → thumbnail hurts and we need higher resolution history.
    #
    # Context: sys(400) + 3×(obs_text(50)+img(325)+response(512)) + current(375) ≈ 3436 tokens.
    # ROLLOUT_PROMPT=4096 leaves ~660 tokens of headroom — should run cleanly.
    EXPERIMENT_NAME=membins_grpo_window3
    TRAIN_FILE=examples/train/memory_bins/train_memory_bins.yaml
    VAL_FILE=examples/train/memory_bins/val_memory_bins.yaml
    DATA_MAX_PROMPT=4096
    DATA_MAX_RESPONSE=512
    ROLLOUT_PROMPT=4096
    ROLLOUT_RESPONSE=512
    MAX_BATCHED_TOKENS=8192
    TRAIN_BATCH_SIZE=2
    PPO_MINI_BATCH_SIZE=2
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
    HISTORY_ARGS=(trainer.history_window_size=3 trainer.thumbnail_scale=1.0)
    ;;

  full_thumb)
    # Ablation: unlimited history with thumbnail compression.
    # Shows whether a longer (but compressed) memory helps over a 3-turn window.
    #
    # Warning: response tokens (512 each) dominate even thumbnail turns.
    # At turn 60: 59×(50+20+512)=59×582≈34k tokens — far beyond any budget.
    # ROLLOUT_PROMPT=8192 accommodates roughly 12–14 thumbnail turns before
    # truncation; useful as an upper bound but expect frequent prompt truncation.
    # Future work: strip <think> from old turns to cut response cost to ~25 tokens.
    EXPERIMENT_NAME=membins_grpo_full_thumb
    TRAIN_FILE=examples/train/memory_bins/train_memory_bins.yaml
    VAL_FILE=examples/train/memory_bins/val_memory_bins.yaml
    # DATA_MAX_PROMPT < ROLLOUT_PROMPT intentionally:
    # Rollout sees up to 8192 tokens (~13 thumbnail turns) — full long-context behavior.
    # Training truncates to 4096 tokens (~6 turns) to avoid GPU OOM during backward:
    # 8192-token attention matrix = (16, 8192, 8192) × 2 bytes ≈ 4.2 GB peak per layer.
    # At 4096 tokens the peak drops to ~1 GB, fitting comfortably within the H100.
    DATA_MAX_PROMPT=4096
    DATA_MAX_RESPONSE=512
    ROLLOUT_PROMPT=8192
    ROLLOUT_RESPONSE=512
    MAX_BATCHED_TOKENS=16384
    TRAIN_BATCH_SIZE=2
    PPO_MINI_BATCH_SIZE=2
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
    HISTORY_ARGS=(trainer.history_window_size=-1 trainer.thumbnail_scale=0.25)
    # GPU_MEMORY_UTIL stays at the default 0.4 (not 0.5).
    # free_cache_engine=True empties the SGLang KV cache between rollout and training
    # but does NOT release the pre-allocated GPU memory pool. At 0.5 the pool holds
    # ~39.6 GB, leaving only ~2.75 GB free for loss.backward() → OOM.
    # At 0.4 the pool is ~31.7 GB, freeing ~8 GB for the backward pass.
    ;;

  full_memory)
    # WARNING: This mode will be heavily truncated for episodes longer than ~9 turns.
    #
    # Token budget: sys(400) + (T-1)×(obs_text(50)+img(325)+response(512)) + current(375)
    #   Turn 5:  sys + 4×887 + 375 = 4323  (already over 4096, needs 8192)
    #   Turn 9:  sys + 8×887 + 375 = 7871  (fits in 8192 with margin)
    #   Turn 10: sys + 9×887 + 375 = 8758  (truncated at ROLLOUT_PROMPT=8192)
    # With max_turns=60, VAGEN will truncate the oldest turns to fit the budget,
    # so the model effectively sees at most ~9 turns of full-res context.
    #
    # Useful interpretations:
    #   1. Upper bound: "as much full-res history as fits in 8192 tokens" (~9 turns).
    #   2. For a clean full-history comparison, use the smoke yamls (max_turns=30)
    #      and accept that the model still sees at most ~9 turns before truncation.
    #   3. Becomes truly clean only after <think> stripping is implemented
    #      (drops response cost from 512 → ~25 tokens, fitting ~32 turns in 8192).
    EXPERIMENT_NAME=membins_grpo_full_memory
    TRAIN_FILE=examples/train/memory_bins/train_memory_bins.yaml
    # Validation uses smoke yaml (8 envs) instead of the full 200-env val set.
    # Full-res unlimited history: 200 envs × 60 turns × ~8 full-res images ≈ 170 GB
    # of image data in Ray object store → system RAM OOM even at 256 G.
    # 8 smoke envs × 30 turns × ~8 images ≈ 3 GB — feasible.
    # Val metrics will have higher variance but the mode runs to completion.
    VAL_FILE=examples/train/memory_bins/val_memory_bins_smoke.yaml
    # Same DATA_MAX_PROMPT=4096 cap as full_thumb: 8192-token backward OOMs on GPU.
    DATA_MAX_PROMPT=4096
    DATA_MAX_RESPONSE=512
    ROLLOUT_PROMPT=8192
    ROLLOUT_RESPONSE=512
    MAX_BATCHED_TOKENS=16384
    TRAIN_BATCH_SIZE=2
    PPO_MINI_BATCH_SIZE=2
    ROLLOUT_N=4
    VAL_BATCH_SIZE=8
    ACTOR_USE_KL_LOSS=False
    ACTOR_KL_LOSS_COEF=0.0
    AGENT_CONFIG=agent_no_concat.yaml
    CONCAT_MULTI_TURN=False
    LOG_IMAGE_ENABLE=False
    ADV_ESTIMATOR=grpo
    ADV_EXTRA_ARGS=(algorithm.norm_adv_by_std_in_grpo=True)
    CRITIC_ARGS=(critic.enable=False)
    HISTORY_ARGS=(trainer.history_window_size=-1 trainer.thumbnail_scale=1.0)
    # Same GPU_MEMORY_UTIL=0.4 rationale as full_thumb above.
    ;;

  4gpu)
    # Scale window3_thumb to 4 H100s for 256 traj/step.
    # Submit: sbatch --time=24:00:00 --gres=gpu:h100:4 run_memory_bins_train_grpo.sh 4gpu
    EXPERIMENT_NAME=membins_grpo_window3_thumb_4gpu
    TRAIN_FILE=examples/train/memory_bins/train_memory_bins.yaml
    VAL_FILE=examples/train/memory_bins/val_memory_bins.yaml
    DATA_MAX_PROMPT=4096
    DATA_MAX_RESPONSE=512
    ROLLOUT_PROMPT=4096
    ROLLOUT_RESPONSE=512
    MAX_BATCHED_TOKENS=10000
    TRAIN_BATCH_SIZE=32
    PPO_MINI_BATCH_SIZE=32
    ROLLOUT_N=8
    VAL_BATCH_SIZE=32
    ACTOR_USE_KL_LOSS=False
    ACTOR_KL_LOSS_COEF=0.0
    AGENT_CONFIG=agent_no_concat.yaml
    CONCAT_MULTI_TURN=False
    LOG_IMAGE_ENABLE=False
    ADV_ESTIMATOR=grpo
    ADV_EXTRA_ARGS=(algorithm.norm_adv_by_std_in_grpo=True)
    CRITIC_ARGS=(critic.enable=False)
    HISTORY_ARGS=(trainer.history_window_size=3 trainer.thumbnail_scale=0.25)
    EXTRA_ARGS=(
      actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
      actor_rollout_ref.ref.use_torch_compile=False
    )
    N_GPUS_PER_NODE=4
    GPU_MEMORY_UTIL=0.6
    VAGEN_SGLANG_INIT_TIMEOUT=1800
    RAY_NUM_CPUS=32
    ;;

  *)
    echo "Unknown MODE: ${MODE}. Use 'smoke', 'no_memory', 'window3_thumb', 'window3', 'full_thumb', 'full_memory', or '4gpu'." >&2
    exit 1
    ;;
esac

# Compute AGENT_NUM_WORKERS: largest divisor of ROLLOUT_PROMPT_COUNT that fits MAX_AGENT_NUM_WORKERS.
ROLLOUT_PROMPT_COUNT=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
AGENT_NUM_WORKERS=${ROLLOUT_PROMPT_COUNT}
if [ "${AGENT_NUM_WORKERS}" -gt "${MAX_AGENT_NUM_WORKERS}" ]; then
  AGENT_NUM_WORKERS=${MAX_AGENT_NUM_WORKERS}
fi
while [ "${AGENT_NUM_WORKERS}" -gt 1 ] && [ $((ROLLOUT_PROMPT_COUNT % AGENT_NUM_WORKERS)) -ne 0 ]; do
  AGENT_NUM_WORKERS=$((AGENT_NUM_WORKERS - 1))
done

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${RUN_ROOT}"
cd "${PROJECT_ROOT}"

module purge
module load python-miniconda3/4.10.3
module load gcc/11.2.0
module load cuda/12.6.2-gcc-12.4.0

source ~/.bashrc
conda activate vagen_noflash

set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
else
  echo "No nvcc found after module load."
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

unset PYTHONNOUSERSITE
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export TORCH_CUDA_ARCH_LIST="9.0"
export HF_HOME=/projects/p33224/hf_cache
export HF_HUB_CACHE="${HF_HOME}/hub"
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

# Prefer local HF snapshot so SGLang doesn't need Hub metadata checks.
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

VAGEN_SGLANG_INIT_TIMEOUT="${VAGEN_SGLANG_INIT_TIMEOUT:-600}"
export VAGEN_SGLANG_INIT_TIMEOUT

echo "MODE:                  ${MODE}"
echo "EXPERIMENT_NAME:       ${EXPERIMENT_NAME}"
echo "MODEL_REPO_ID:         ${MODEL_REPO_ID}"
echo "REF_MODEL_PATH:        ${REF_MODEL_PATH}"
echo "TRAIN_FILE:            ${TRAIN_FILE}"
echo "VAL_FILE:              ${VAL_FILE}"
echo "TRAIN_BATCH_SIZE:      ${TRAIN_BATCH_SIZE}"
echo "PPO_MINI_BATCH_SIZE:   ${PPO_MINI_BATCH_SIZE}"
echo "ROLLOUT_N:             ${ROLLOUT_N}"
echo "ROLLOUT_PROMPT_COUNT:  ${ROLLOUT_PROMPT_COUNT}"
echo "AGENT_NUM_WORKERS:     ${AGENT_NUM_WORKERS}"
echo "VAL_BATCH_SIZE:        ${VAL_BATCH_SIZE}"
echo "ADV_ESTIMATOR:         ${ADV_ESTIMATOR}"
echo "N_GPUS_PER_NODE:       ${N_GPUS_PER_NODE}"
echo "GPU_MEMORY_UTIL:       ${GPU_MEMORY_UTIL}"
echo "HISTORY_ARGS:          ${HISTORY_ARGS[*]:-}"
echo "ACTOR_USE_KL_LOSS:     ${ACTOR_USE_KL_LOSS}"
echo "ACTOR_KL_LOSS_COEF:    ${ACTOR_KL_LOSS_COEF}"
echo "LOG_IMAGE_ENABLE:      ${LOG_IMAGE_ENABLE}"
echo "VAL_BEFORE_TRAIN:      ${VAL_BEFORE_TRAIN}"
echo "TOTAL_TRAINING_STEPS:  ${TOTAL_TRAINING_STEPS}"
echo "SAVE_FREQ:             ${SAVE_FREQ}"
echo "TEST_FREQ:             ${TEST_FREQ}"
echo "LOG_VAL_GENERATIONS:   ${LOG_VAL_GENERATIONS}"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}"
echo "CONDA_PREFIX=${CONDA_PREFIX:-unset}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "HF_HOME=${HF_HOME}"
echo "HF_TOKEN_PRESENT=${HF_TOKEN_PRESENT}"
echo "PYTHONPATH=${PYTHONPATH}"
which python
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
SYNC_ARCHIVE_DIR="${PROJECT_ROOT}/logs/sglang_sync/${SLURM_JOB_ID}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-4}}"

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
  fi
}

trap archive_ray_logs EXIT

PY=$(which python)
echo "Python: ${PY}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "RAY_TMPDIR: ${RAY_TMPDIR}"
echo "RAY_NUM_CPUS: ${RAY_NUM_CPUS}"
echo "SYNC_ROOT: ${SYNC_ROOT}"
echo "VAGEN_SGLANG_WEIGHT_SYNC_METHOD: ${VAGEN_SGLANG_WEIGHT_SYNC_METHOD}"
echo "TORCHDYNAMO_DISABLE: ${TORCHDYNAMO_DISABLE}"
echo "FLASHINFER_ENABLE_JIT: ${FLASHINFER_ENABLE_JIT}"
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
  "${EXTRA_ARGS[@]}" \
  'trainer.logger=[console,wandb]' \
  trainer.log_image.enable=${LOG_IMAGE_ENABLE} \
  trainer.resume_mode=disable \
  trainer.val_before_train=${VAL_BEFORE_TRAIN} \
  trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.test_freq=${TEST_FREQ} \
  trainer.project_name="vagen_memory_bins" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${RUN_ROOT}/checkpoints/${EXPERIMENT_NAME}" \
  trainer.validation_data_dir="${RUN_ROOT}/validation/${EXPERIMENT_NAME}" \
  trainer.rollout_data_dir="${RUN_ROOT}/rollout/${EXPERIMENT_NAME}" \
  trainer.log_val_generations=${LOG_VAL_GENERATIONS}
