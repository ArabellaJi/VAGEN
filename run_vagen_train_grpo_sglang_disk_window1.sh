#!/bin/bash
#SBATCH --job-name=vagen_grpo_sokoban_3b_w1
#SBATCH --account=p33224
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/home/eiu4164/projects/VAGEN/logs/%x_%j.out
#SBATCH --error=/home/eiu4164/projects/VAGEN/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wenlanji2026@u.northwestern.edu

set -eo pipefail

PROJECT_ROOT=/home/eiu4164/projects/VAGEN
RUN_ROOT=/projects/p33224/vagen_runs
# window1: no-concat mode, model sees only 1 previous turn as history context
EXPERIMENT_NAME=sokoban_grpo_sglang_disk_3b_window1

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

unset PYTHONNOUSERSITE
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export TORCH_CUDA_ARCH_LIST="9.0"
export HF_HOME=/projects/p33224/hf_cache
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

export TORCHDYNAMO_DISABLE=1
export FLASHINFER_JIT_WORKER_TIMEOUT=60
export FLASHINFER_ENABLE_JIT=0
export VAGEN_SGLANG_INIT_TIMEOUT=600

echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}"
echo "CONDA_PREFIX=${CONDA_PREFIX:-unset}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "CUDA_PATH=${CUDA_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "CC=${CC}"
echo "CXX=${CXX}"
echo "CUDAHOSTCXX=${CUDAHOSTCXX}"
echo "CMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}"
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
RAY_NUM_CPUS=8

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
echo "VAGEN_SGLANG_INIT_TIMEOUT: ${VAGEN_SGLANG_INIT_TIMEOUT}"

PYTHONUNBUFFERED=1 "${PY}" -m vagen.main_ppo \
  --config-path="${PWD}/vagen/configs" \
  --config-name="vagen_multiturn" \
  data.train_files="${PWD}/examples/train/sokoban/train_sokoban_vision.yaml" \
  data.val_files="${PWD}/examples/train/sokoban/val_sokoban_vision.yaml" \
  data.train_batch_size=4 \
  data.dataloader_num_workers=0 \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
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
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.checkpoint.save_contents=['hf_model'] \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.prompt_length=2048 \
  actor_rollout_ref.rollout.response_length=512 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  +actor_rollout_ref.rollout.engine_kwargs.sglang.sampling_backend=pytorch \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.agent.num_workers=4 \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${PWD}/vagen/configs/agent_no_concat.yaml" \
  actor_rollout_ref.rollout.disable_log_stats=False \
  trainer.concat_multi_turn=False \
  trainer.history_window_size=1 \
  trainer.thumbnail_scale=0.25 \
  trainer.n_gpus_per_node=1 \
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
  "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_SGLANG_INIT_TIMEOUT='${VAGEN_SGLANG_INIT_TIMEOUT}'" \
  trainer.critic_warmup=0 \
  critic.enable=False \
  'trainer.logger=[console]' \
  trainer.val_before_train=True \
  trainer.total_training_steps=200 \
  trainer.save_freq=20 \
  trainer.test_freq=20 \
  trainer.project_name="vagen" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${RUN_ROOT}/checkpoints/${EXPERIMENT_NAME}" \
  trainer.validation_data_dir="${RUN_ROOT}/validation/${EXPERIMENT_NAME}" \
  trainer.rollout_data_dir="${RUN_ROOT}/rollout/${EXPERIMENT_NAME}" \
  trainer.log_val_generations=5
