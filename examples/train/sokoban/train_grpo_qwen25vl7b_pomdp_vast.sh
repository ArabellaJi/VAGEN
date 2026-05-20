#!/bin/bash
# POMDP Sokoban GRPO training — Vast.ai smoke test (100 steps, 7B)
#
# Conditions (set via MEMORY env var before running):
#   MEMORY=full      trainer.history_window_size=-1  thumbnail_scale=1.0
#   MEMORY=thumb05   trainer.history_window_size=-1  thumbnail_scale=0.5
#   MEMORY=none      trainer.history_window_size=0
#
# GPU count is auto-detected via N_GPUS env var (default 1).
# Disk-based weight sync is used so SGLang and FSDP weights never coexist in GPU memory.
#
# Usage (single GPU):
#   MEMORY=full MODEL_PATH=/root/models/Qwen2.5-VL-7B-Instruct bash train_grpo_qwen25vl7b_pomdp_vast.sh
#
# Usage (4-GPU, override any param):
#   N_GPUS=4 MEMORY=full MODEL_PATH=... bash train_grpo_qwen25vl7b_pomdp_vast.sh \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.5

set -x

MEMORY="${MEMORY:-full}"
MODEL_PATH="${MODEL_PATH:-/root/models/Qwen2.5-VL-7B-Instruct}"
N_GPUS="${N_GPUS:-1}"

PROJECT_NAME="vagen_pomdp"
EXPERIMENT_NAME="sokoban_grpo_7b_pomdp_${MEMORY}"

BASEDIR=$(pwd)
SCRIPTDIR=$(dirname "$0")
EXPERIMENT_DIR=${BASEDIR}/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}
SAVE_CHECKPOINT_DIR=${EXPERIMENT_DIR}/verl_checkpoints
DATASET_TRAIN=${SCRIPTDIR}/train_sokoban_pomdp.yaml
DATASET_VAL=${SCRIPTDIR}/val_sokoban_pomdp.yaml
agent_loop_config_path=${BASEDIR}/vagen/configs/agent.yaml

mkdir -p ${EXPERIMENT_DIR}

# Memory strategy parameters
case "${MEMORY}" in
  full)
    HISTORY_WINDOW=-1
    THUMBNAIL=1.0
    ;;
  thumb05)
    HISTORY_WINDOW=-1
    THUMBNAIL=0.5
    ;;
  none)
    HISTORY_WINDOW=0
    THUMBNAIL=1.0
    ;;
  *)
    echo "Unknown MEMORY='${MEMORY}'. Use: full | thumb05 | none"
    exit 1
    ;;
esac

# Disk-based weight sync: FSDP writes to /tmp, SGLang reloads from there.
# This avoids having both models in GPU memory simultaneously (which causes OOM on 1 GPU).
SYNC_ROOT="/tmp/vagen_sglang_sync_$$"
mkdir -p "${SYNC_ROOT}"
trap "rm -rf ${SYNC_ROOT}" EXIT

# Disable torch.compile and FlashInfer JIT to prevent deadlocks during SGLang init.
export TORCHDYNAMO_DISABLE=1
export FLASHINFER_ENABLE_JIT=0
export FLASHINFER_JIT_WORKER_TIMEOUT=60
export VAGEN_SGLANG_INIT_TIMEOUT=600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== POMDP Training Smoke Test ==="
echo "  MEMORY strategy : ${MEMORY}"
echo "  history_window  : ${HISTORY_WINDOW}"
echo "  thumbnail_scale : ${THUMBNAIL}"
echo "  model           : ${MODEL_PATH}"
echo "  n_gpus          : ${N_GPUS}"
echo "  sync_root       : ${SYNC_ROOT}"
echo "=================================="

PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
    --config-path=${BASEDIR}/vagen/configs \
    --config-name='vagen_multiturn' \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=${DATASET_VAL} \
    data.train_batch_size=32 \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    "+actor_rollout_ref.rollout.engine_kwargs.sglang.sampling_backend=pytorch" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=${agent_loop_config_path} \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    trainer.history_window_size=${HISTORY_WINDOW} \
    trainer.thumbnail_scale=${THUMBNAIL} \
    trainer.concat_multi_turn=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=25 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR} \
    trainer.validation_data_dir=${EXPERIMENT_DIR}/validation \
    trainer.rollout_data_dir=${EXPERIMENT_DIR}/rollout_data \
    trainer.log_val_generations=32 \
    trainer.total_training_steps=100 \
    data.max_prompt_length=4000 \
    data.max_response_length=6000 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_SGLANG_WEIGHT_SYNC_METHOD=disk" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_SGLANG_WEIGHT_SYNC_DIR='${SYNC_ROOT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_SGLANG_WEIGHT_SYNC_LOAD_FORMAT=auto" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_SGLANG_WEIGHT_SYNC_FLUSH_CACHE=true" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.TORCHDYNAMO_DISABLE=1" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.FLASHINFER_ENABLE_JIT=0" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    "$@" \
    2>&1 | tee ${EXPERIMENT_DIR}/${EXPERIMENT_NAME}.log
