#!/bin/bash
# run_navigation_grpo.sh
#
# Navigation GRPO training on Quest (Northwestern) — long-horizon memory ablation.
#
# ── One-time pre-requisites (run once before first job) ──────────────────────
#   pip install ai2thor
#   sudo apt-get install -y libvulkan1 vulkan-tools
#   conda activate vagen_noflash
#   python -m vagen.envs.navigation.pre_download_scenes   # ~10 min, downloads all scenes
#
# ── Submission ────────────────────────────────────────────────────────────────
#   sbatch --export=CONDITION=full_memory  run_navigation_grpo.sh   # full trajectory baseline
#   sbatch --export=CONDITION=no_memory    run_navigation_grpo.sh   # no history
#   sbatch --export=CONDITION=window3      run_navigation_grpo.sh   # 3-turn sliding window
#   sbatch --export=CONDITION=thumbnail    run_navigation_grpo.sh   # full history, 25% images
#   sbatch --export=CONDITION=window3_thumb run_navigation_grpo.sh  # proposed method
#   sbatch --export=CONDITION=quick        run_navigation_grpo.sh   # smoke-test (5 steps)
#
# Two GPUs are requested:
#   GPU 0 — AI2-THOR navigation server  (Unity rendering)
#   GPU 1 — GRPO training + SGLang rollout
#
#SBATCH --job-name=nav_grpo
#SBATCH --account=p33224
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/home/eiu4164/projects/VAGEN/logs/%x_%j.out
#SBATCH --error=/home/eiu4164/projects/VAGEN/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wenlanji2026@u.northwestern.edu

set -eo pipefail

# ── Experiment condition ──────────────────────────────────────────────────────
CONDITION="${CONDITION:-full_memory}"

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT=/home/eiu4164/projects/VAGEN
RUN_ROOT=/projects/p33224/vagen_runs
HF_CACHE=/projects/p33224/hf_cache
SCRIPTDIR="${PROJECT_ROOT}/examples/train/navigation"

# GPU assignment: nav server on GPU 0, training on GPU 1
NAV_GPU=0
TRAIN_GPU=1
NAV_SERVER_PORT=8000

mkdir -p "${PROJECT_ROOT}/logs" "${RUN_ROOT}"
cd "${PROJECT_ROOT}"

# ── Condition-specific parameters ─────────────────────────────────────────────
case "${CONDITION}" in
  quick)
    # Smoke-test: verify the full pipeline works. 5 training steps, small batch.
    EXPERIMENT_NAME=nav_grpo_quick
    TRAIN_DATA=${SCRIPTDIR}/train_navigation_quick.yaml       # n_envs=30, max_turns=5
    VAL_DATA=${SCRIPTDIR}/val_navigation_quick.yaml           # n_envs=10
    NAV_MAX_ENVS=16
    TRAINING_STEPS=5
    TRAIN_BATCH_SIZE=8
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
    ;;
  full_memory)
    # Full trajectory in context — baseline for comparison.
    EXPERIMENT_NAME=nav_grpo_full_memory
    TRAIN_DATA=${SCRIPTDIR}/train_navigation_window_exp.yaml  # created below
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
    ;;
  no_memory)
    # Only the current frame — no history at all.
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
    ;;
  window3)
    # 3-turn sliding window, full-resolution images — memory without compression.
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
    ;;
  thumbnail)
    # Full history, old images compressed to 25% resolution.
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
    ;;
  window3_thumb)
    # Proposed method: 3 recent turns at full resolution + all older as 25% thumbnails.
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
    ;;
  *)
    echo "ERROR: Unknown CONDITION '${CONDITION}'."
    echo "  Valid: quick | full_memory | no_memory | window3 | thumbnail | window3_thumb"
    exit 1
    ;;
esac

# ── Module loads ──────────────────────────────────────────────────────────────
module purge
module load python-miniconda3/4.10.3
module load gcc/11.2.0
module load cuda/12.6.2-gcc-12.4.0

source ~/.bashrc
conda activate vagen_noflash

set -u

# ── CUDA / compiler environment ───────────────────────────────────────────────
if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
else
  echo "ERROR: nvcc not found after module load."; module list; exit 1
fi

export CUDA_PATH="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PROJECT_ROOT}"

# Force GCC so FlashInfer JIT does not inherit a C++17-incomplete Clang.
unset CC CXX CUDAHOSTCXX CMAKE_CUDA_HOST_COMPILER CUDA_NVCC_EXECUTABLE
GCC_BIN="$(command -v gcc)"
GXX_BIN="$(command -v g++)"
export CC="${GCC_BIN}" CXX="${GXX_BIN}" CUDAHOSTCXX="${GXX_BIN}"
export CMAKE_CUDA_HOST_COMPILER="${GXX_BIN}"

# ── Runtime environment variables ─────────────────────────────────────────────
unset PYTHONNOUSERSITE
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export TORCH_CUDA_ARCH_LIST="9.0"   # H100
export HF_HOME="${HF_CACHE}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export TORCHDYNAMO_DISABLE=1          # stop Dynamo from compiling
export FLASHINFER_ENABLE_JIT=0        # skip FlashInfer JIT
export FLASHINFER_JIT_WORKER_TIMEOUT=60
export VAGEN_SGLANG_INIT_TIMEOUT=600
export WANDB_DIR="${RUN_ROOT}/wandb"
mkdir -p "${WANDB_DIR}"

# ── Temp / SGLang sync dirs ───────────────────────────────────────────────────
export JOB_TMP="/tmp/j${SLURM_JOB_ID}"
export TMPDIR="${JOB_TMP}/t"
export RAY_TMPDIR="${JOB_TMP}/r"
SYNC_ROOT="${JOB_TMP}/sglang_sync"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" "${SYNC_ROOT}"

VAGEN_SGLANG_WEIGHT_SYNC_METHOD=disk
VAGEN_SGLANG_WEIGHT_SYNC_DIR="${SYNC_ROOT}"
VAGEN_SGLANG_WEIGHT_SYNC_LOAD_FORMAT=auto
VAGEN_SGLANG_WEIGHT_SYNC_FLUSH_CACHE=true

RAY_NUM_CPUS=16
RAY_LOG_ARCHIVE_DIR="${PROJECT_ROOT}/logs/ray/${SLURM_JOB_ID}"

# ── Cleanup / archive ─────────────────────────────────────────────────────────
archive_ray_logs() {
  local ray_logs_dir
  ray_logs_dir="$(find "${RAY_TMPDIR}" -maxdepth 4 -type d -path '*/session_latest/logs' 2>/dev/null | head -n 1 || true)"
  if [[ -n "${ray_logs_dir}" && -d "${ray_logs_dir}" ]]; then
    mkdir -p "${RAY_LOG_ARCHIVE_DIR}"
    cp -a "${ray_logs_dir}/." "${RAY_LOG_ARCHIVE_DIR}/"
    echo "Archived Ray logs → ${RAY_LOG_ARCHIVE_DIR}"
  fi
}

cleanup() {
  archive_ray_logs
  if [[ -n "${NAV_SERVER_PID:-}" ]]; then
    echo "[$(date '+%H:%M:%S')] Stopping navigation server (PID ${NAV_SERVER_PID})..."
    kill "${NAV_SERVER_PID}" 2>/dev/null || true
    wait "${NAV_SERVER_PID}" 2>/dev/null || true
    echo "[$(date '+%H:%M:%S')] Navigation server stopped."
  fi
}
trap cleanup EXIT

# ── Diagnostics ───────────────────────────────────────────────────────────────
echo "========================================================"
echo "CONDITION:          ${CONDITION}"
echo "EXPERIMENT:         ${EXPERIMENT_NAME}"
echo "TRAIN_DATA:         ${TRAIN_DATA}"
echo "CONCAT_MULTI_TURN:  ${CONCAT_MULTI_TURN}"
echo "HISTORY_ARGS:       ${HISTORY_ARGS[*]:-none}"
echo "ROLLOUT_PROMPT:     ${ROLLOUT_PROMPT}"
echo "ROLLOUT_RESPONSE:   ${ROLLOUT_RESPONSE}"
echo "SLURM_JOB_ID:       ${SLURM_JOB_ID}"
echo "CUDA_HOME:          ${CUDA_HOME}"
echo "CC=${CC}  CXX=${CXX}"
echo "========================================================"
nvcc --version
nvidia-smi || true

python - <<'PY'
import torch, sys
print("Python:", sys.executable)
print("CUDA available:", torch.cuda.is_available(), "| device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}:", torch.cuda.get_device_name(i))
PY

# ── Step 1: Start navigation server on GPU 0 ──────────────────────────────────
NAV_SERVER_LOG="${PROJECT_ROOT}/logs/nav_server_${SLURM_JOB_ID}.log"
echo ""
echo "[$(date '+%H:%M:%S')] Starting navigation server on GPU ${NAV_GPU} (max_envs=${NAV_MAX_ENVS})..."

CUDA_VISIBLE_DEVICES="${NAV_GPU}" \
  python -m vagen.envs.navigation.serve \
    --devices="[0]" \
    --max_envs="${NAV_MAX_ENVS}" \
    --thread_pool_size="${NAV_MAX_ENVS}" \
    --port="${NAV_SERVER_PORT}" \
  > "${NAV_SERVER_LOG}" 2>&1 &
NAV_SERVER_PID=$!
echo "[$(date '+%H:%M:%S')] Navigation server PID=${NAV_SERVER_PID}  log=${NAV_SERVER_LOG}"

# Poll until healthy (up to 5 min; first env creation takes ~2s, first scene download may take longer)
echo -n "[$(date '+%H:%M:%S')] Waiting for server"
READY=0
for i in $(seq 1 60); do
  if curl -sf "http://localhost:${NAV_SERVER_PORT}/health" > /dev/null 2>&1; then
    READY=1; echo " ready after $((i*5))s"; break
  fi
  if ! kill -0 "${NAV_SERVER_PID}" 2>/dev/null; then
    echo ""
    echo "ERROR: Navigation server crashed. Last 30 lines of ${NAV_SERVER_LOG}:"
    tail -30 "${NAV_SERVER_LOG}" || true
    exit 1
  fi
  echo -n "."
  sleep 5
done

if [[ "${READY}" -eq 0 ]]; then
  echo ""
  echo "ERROR: Server did not become healthy within 300s. Last 30 lines:"
  tail -30 "${NAV_SERVER_LOG}" || true
  exit 1
fi

# ── Step 2: Run GRPO training on GPU 1 ───────────────────────────────────────
EXPERIMENT_DIR="${RUN_ROOT}/checkpoints/${EXPERIMENT_NAME}"
mkdir -p "${EXPERIMENT_DIR}"

echo ""
echo "[$(date '+%H:%M:%S')] Starting training: ${EXPERIMENT_NAME}"

CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" \
PYTHONUNBUFFERED=1 \
  python -m vagen.main_ppo \
    --config-path="${PROJECT_ROOT}/vagen/configs" \
    --config-name="vagen_multiturn" \
    \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.dataloader_num_workers=0 \
    data.max_prompt_length="${DATA_MAX_PROMPT}" \
    data.max_response_length="${DATA_MAX_RESPONSE}" \
    \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    \
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
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
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
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CFG}" \
    actor_rollout_ref.rollout.disable_log_stats=False \
    \
    trainer.concat_multi_turn="${CONCAT_MULTI_TURN}" \
    "${HISTORY_ARGS[@]}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.critic_warmup=0 \
    critic.enable=False \
    trainer.val_before_train=True \
    trainer.resume_mode=disable \
    'trainer.logger=[console,wandb]' \
    'trainer.log_image.enable=True' \
    trainer.total_training_steps="${TRAINING_STEPS}" \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.log_val_generations=10 \
    trainer.project_name="nav_window_ablation" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${EXPERIMENT_DIR}" \
    trainer.validation_data_dir="${RUN_ROOT}/validation/${EXPERIMENT_NAME}" \
    trainer.rollout_data_dir="${RUN_ROOT}/rollout/${EXPERIMENT_NAME}" \
    \
    +ray_kwargs.ray_init.include_dashboard=False \
    +ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}" \
    +ray_kwargs.ray_init.object_store_memory=4294967296 \
    "+ray_kwargs.ray_init._temp_dir='${RAY_TMPDIR}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_HOME='${CUDA_HOME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_PATH='${CUDA_PATH}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.PATH='${PATH}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH='${LD_LIBRARY_PATH}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_VISIBLE_DEVICES='${TRAIN_GPU}'" \
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
    "+ray_kwargs.ray_init.runtime_env.env_vars.FLASHINFER_ENABLE_JIT='${FLASHINFER_ENABLE_JIT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.FLASHINFER_JIT_WORKER_TIMEOUT='${FLASHINFER_JIT_WORKER_TIMEOUT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_SGLANG_INIT_TIMEOUT='${VAGEN_SGLANG_INIT_TIMEOUT}'" \
    2>&1 | tee "${EXPERIMENT_DIR}/train.log"

TRAIN_EXIT=${PIPESTATUS[0]}
echo ""
echo "[$(date '+%H:%M:%S')] Training finished (exit=${TRAIN_EXIT})."
exit "${TRAIN_EXIT}"
