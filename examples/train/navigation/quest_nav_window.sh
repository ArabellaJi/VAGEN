#!/bin/bash
# ============================================================
# QUEST SLURM script: Navigation window/memory ablation
#
# Usage:
#   # Full memory (baseline, uses concat mode):
#   sbatch --export=CONDITION=full_memory quest_nav_window.sh
#
#   # No memory (window=0):
#   sbatch --export=CONDITION=no_memory quest_nav_window.sh
#
#   # Sliding window (k=3):
#   sbatch --export=CONDITION=window3 quest_nav_window.sh
#
#   # Thumbnail (full history, images compressed to 25%):
#   sbatch --export=CONDITION=thumbnail quest_nav_window.sh
#
# Or edit CONDITION directly below and submit normally:
#   sbatch quest_nav_window.sh
# ============================================================

#SBATCH --job-name=nav_window_test
#SBATCH --account=p33224
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:2          # GPU 0: nav server; GPU 1: training
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/home/eiu4164/projects/VAGEN/logs/%x_%j.out
#SBATCH --error=/home/eiu4164/projects/VAGEN/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wenlanji2026@u.northwestern.edu

set -eo pipefail

# ── Experiment condition ──────────────────────────────────────────────────────
# Options: full_memory | no_memory | window3 | thumbnail
CONDITION="${CONDITION:-no_memory}"

# ── Paths ─────────────────────────────────────────────────────────────────────
VAGEN_ROOT="/home/eiu4164/projects/VAGEN"
SCRIPTDIR="${VAGEN_ROOT}/examples/train/navigation"
LOG_DIR="${VAGEN_ROOT}/logs"
RUN_ROOT="/projects/p33224/vagen_runs"
HF_CACHE="/projects/p33224/hf_cache"

mkdir -p "${LOG_DIR}" "${RUN_ROOT}"

# ── Condition-specific settings ───────────────────────────────────────────────
case "${CONDITION}" in
  full_memory)
    # Concat mode: full history, standard training
    CONCAT_MULTI_TURN=True
    HISTORY_WINDOW_SIZE=-1      # unused in concat mode
    THUMBNAIL_SCALE=1.0         # unused in concat mode
    ADV_ESTIMATOR=gae
    AGENT_LOOP_CFG="${VAGEN_ROOT}/vagen/configs/agent.yaml"
    ;;
  no_memory)
    # No-concat mode, window=0: model sees only current frame
    CONCAT_MULTI_TURN=False
    HISTORY_WINDOW_SIZE=0
    THUMBNAIL_SCALE=1.0
    ADV_ESTIMATOR=no_concat_gae
    AGENT_LOOP_CFG="${VAGEN_ROOT}/vagen/configs/agent_no_concat.yaml"
    ;;
  window3)
    # No-concat mode, sliding window of 3 turns
    CONCAT_MULTI_TURN=False
    HISTORY_WINDOW_SIZE=3
    THUMBNAIL_SCALE=1.0
    ADV_ESTIMATOR=no_concat_gae
    AGENT_LOOP_CFG="${VAGEN_ROOT}/vagen/configs/agent_no_concat.yaml"
    ;;
  thumbnail)
    # No-concat mode, full history but historical images compressed to 25%
    CONCAT_MULTI_TURN=False
    HISTORY_WINDOW_SIZE=-1
    THUMBNAIL_SCALE=0.25
    ADV_ESTIMATOR=no_concat_gae
    AGENT_LOOP_CFG="${VAGEN_ROOT}/vagen/configs/agent_no_concat.yaml"
    ;;
  *)
    echo "ERROR: Unknown CONDITION '${CONDITION}'. Use: full_memory | no_memory | window3 | thumbnail"
    exit 1
    ;;
esac

EXPERIMENT_NAME="nav_${CONDITION}"
EXPERIMENT_DIR="${RUN_ROOT}/checkpoints/${EXPERIMENT_NAME}"
mkdir -p "${EXPERIMENT_DIR}"

# ── Environment setup ─────────────────────────────────────────────────────────
module load python-miniconda3/4.10.3
source ~/.bashrc
conda activate vagen

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_CACHE}"
export OMP_NUM_THREADS=8
export WANDB_DIR="${RUN_ROOT}/wandb"
mkdir -p "${WANDB_DIR}"

export JOB_TMP="/tmp/j${SLURM_JOB_ID}"
export TMPDIR="${JOB_TMP}/t"
export RAY_TMPDIR="${JOB_TMP}/r"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"

# ── Logging ───────────────────────────────────────────────────────────────────
PY=$(which python)
echo "Python:             ${PY}"
echo "SLURM_JOB_ID:       ${SLURM_JOB_ID}"
echo "CONDITION:          ${CONDITION}"
echo "concat_multi_turn:  ${CONCAT_MULTI_TURN}"
echo "history_window_size:${HISTORY_WINDOW_SIZE}"
echo "thumbnail_scale:    ${THUMBNAIL_SCALE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-auto}"

cd "${VAGEN_ROOT}"

# ── Step 1: Start Navigation server on GPU 0 ─────────────────────────────────
echo "[$(date)] Starting Navigation server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m vagen.envs.navigation.serve \
    --devices="[0]" \
    --max_envs=64 \
    --thread_pool_size=64 \
    --port=8000 \
    > "${LOG_DIR}/nav_server_${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
echo "Navigation server PID: ${SERVER_PID}"

# Wait until server is ready (poll /health endpoint)
echo "[$(date)] Waiting for navigation server to be ready..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "[$(date)] Server ready after ${i} x 5s = $((i*5))s"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: Navigation server crashed. Check ${LOG_DIR}/nav_server_${SLURM_JOB_ID}.log"
        exit 1
    fi
    sleep 5
done

# ── Step 2: Run PPO training on GPU 1 ────────────────────────────────────────
echo "[$(date)] Starting training: ${EXPERIMENT_NAME}"

CUDA_VISIBLE_DEVICES=1 python -m vagen.main_ppo \
    --config-path="${VAGEN_ROOT}/vagen/configs" \
    --config-name='vagen_multiturn' \
    \
    data.train_files="${SCRIPTDIR}/train_navigation_quick.yaml" \
    data.val_files="${SCRIPTDIR}/val_navigation_quick.yaml" \
    data.train_batch_size=30 \
    data.max_prompt_length=4000 \
    data.max_response_length=3000 \
    \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.kl_ctrl.kl_coef=0.0 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CFG}" \
    actor_rollout_ref.rollout.disable_log_stats=False \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    critic.enable=True \
    critic.optim.lr=1e-5 \
    critic.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.concat_multi_turn=${CONCAT_MULTI_TURN} \
    trainer.history_window_size=${HISTORY_WINDOW_SIZE} \
    trainer.thumbnail_scale=${THUMBNAIL_SCALE} \
    trainer.val_before_train=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="nav_window_ablation" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${EXPERIMENT_DIR}" \
    trainer.validation_data_dir="${EXPERIMENT_DIR}/validation" \
    trainer.rollout_data_dir="${EXPERIMENT_DIR}/rollout_data" \
    trainer.log_val_generations=10 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_training_steps=30 \
    2>&1 | tee "${EXPERIMENT_DIR}/train.log"

TRAIN_EXIT=$?

# ── Cleanup ───────────────────────────────────────────────────────────────────
echo "[$(date)] Training finished (exit=${TRAIN_EXIT}). Stopping navigation server..."
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true

echo "[$(date)] Done. Logs: ${EXPERIMENT_DIR}/train.log"
exit ${TRAIN_EXIT}
