#!/bin/bash
# MEM1 POMDP Sokoban GRPO training — Vast.ai smoke test (100 steps, 7B text)
#
# MEM1: compact memory state instead of full conversation history.
# Each inference step uses only [system + memory + current_obs].
# Full trajectory is retained in prompt_ids for GRPO loss computation.
#
# GPU count is auto-detected via N_GPUS env var (default 1).
# Disk-based weight sync is used so SGLang and FSDP weights never coexist in GPU memory.
#
# Usage (single GPU):
#   MODEL_PATH=/root/models/Qwen2.5-7B-Instruct bash train_grpo_qwen25_7b_mem1_pomdp_vast.sh
#
# Usage (4-GPU, override any param):
#   N_GPUS=4 MODEL_PATH=... bash train_grpo_qwen25_7b_mem1_pomdp_vast.sh \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.5

set -x

MODEL_PATH="${MODEL_PATH:-/root/models/Qwen2.5-7B-Instruct}"
N_GPUS="${N_GPUS:-1}"

PROJECT_NAME="vagen_pomdp"
EXPERIMENT_NAME="sokoban_grpo_7b_mem1_pomdp"

BASEDIR=$(pwd)
SCRIPTDIR=$(dirname "$0")
EXPERIMENT_DIR=${BASEDIR}/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}
SAVE_CHECKPOINT_DIR=${EXPERIMENT_DIR}/verl_checkpoints
DATASET_TRAIN=${SCRIPTDIR}/train_sokoban_mem1_pomdp.yaml
DATASET_VAL=${SCRIPTDIR}/val_sokoban_mem1_pomdp.yaml
agent_loop_config_path=${BASEDIR}/vagen/configs/agent.yaml

mkdir -p ${EXPERIMENT_DIR}

# ---------------------------------------------------------------------------
# Patch verl's fsdp_checkpoint_manager to move state_dict to CPU before
# save_pretrained.  HF's save_pretrained clones every CUDA tensor, which OOMs
# when SGLang + FSDP already occupy ~79 GiB of an 80 GiB A100.  CPU tensors
# clone in RAM (221 GiB free), so this is safe and always fast enough.
# The patch is idempotent: a second run is a no-op.
# ---------------------------------------------------------------------------
_CKPT_MGR="${BASEDIR}/verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py"
if [ -f "${_CKPT_MGR}" ] && ! grep -q "_vagen_cpu_patched" "${_CKPT_MGR}"; then
    python3 - "${_CKPT_MGR}" <<'PYPATCH'
import sys
path = sys.argv[1]
with open(path) as f:
    code = f.read()
old = '                save_model.save_pretrained(hf_local_path, state_dict=state_dict)'
new = (
    '                # vagen: move to CPU to avoid CUDA OOM on A100 80GB (_vagen_cpu_patched)\n'
    '                state_dict = {\n'
    '                    k: v.cpu() if hasattr(v, "cpu") and getattr(v, "is_cuda", False) else v\n'
    '                    for k, v in state_dict.items()\n'
    '                }\n'
    '                save_model.save_pretrained(hf_local_path, state_dict=state_dict)'
)
if old not in code:
    print(f"[vagen] WARNING: patch pattern not found in {path!r} — skipping", flush=True)
    sys.exit(0)
with open(path, "w") as f:
    f.write(code.replace(old, new, 1))
print(f"[vagen] Patched {path}", flush=True)
PYPATCH
elif [ -f "${_CKPT_MGR}" ]; then
    echo "[vagen] fsdp_checkpoint_manager already patched — skipping"
else
    echo "[vagen] WARNING: ${_CKPT_MGR} not found — patch skipped"
fi

# Kill any leftover SGLang/Ray processes from previous crashed runs.
ray stop --force 2>/dev/null || true
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
pkill -9 -f "sglang._srt" 2>/dev/null || true
sleep 3

# Disk-based weight sync: FSDP writes here, SGLang reloads from here.
# Use the experiment directory (main filesystem) rather than /tmp (often a small tmpfs)
# to avoid "No space left on device" when writing the 15 GiB model weights.
SYNC_ROOT="${EXPERIMENT_DIR}/sglang_sync"
mkdir -p "${SYNC_ROOT}"
trap "rm -rf ${SYNC_ROOT}" EXIT

# Disable torch.compile and FlashInfer JIT to prevent deadlocks during SGLang init.
export TORCHDYNAMO_DISABLE=1
export FLASHINFER_ENABLE_JIT=0
export FLASHINFER_JIT_WORKER_TIMEOUT=60
export VAGEN_SGLANG_INIT_TIMEOUT=600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== MEM1 POMDP Training Smoke Test ==="
echo "  model    : ${MODEL_PATH}"
echo "  n_gpus   : ${N_GPUS}"
echo "  sync_root: ${SYNC_ROOT}"
echo "======================================="

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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=${agent_loop_config_path} \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    trainer.history_window_size=-1 \
    trainer.concat_multi_turn=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=50 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR} \
    trainer.validation_data_dir=${EXPERIMENT_DIR}/validation \
    trainer.rollout_data_dir=null \
    trainer.log_val_generations=0 \
    trainer.total_training_steps=100 \
    data.max_prompt_length=4000 \
    data.max_response_length=6000 \
    critic.enable=False \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_SGLANG_WEIGHT_SYNC_METHOD='disk'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_SGLANG_WEIGHT_SYNC_DIR='${SYNC_ROOT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_SGLANG_WEIGHT_SYNC_LOAD_FORMAT='auto'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VAGEN_SGLANG_WEIGHT_SYNC_FLUSH_CACHE='true'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.TORCHDYNAMO_DISABLE='1'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.FLASHINFER_ENABLE_JIT='0'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'" \
    +ray_kwargs.ray_init.object_store_memory=4294967296 \
    "$@" \
    2>&1 | tee ${EXPERIMENT_DIR}/${EXPERIMENT_NAME}.log
