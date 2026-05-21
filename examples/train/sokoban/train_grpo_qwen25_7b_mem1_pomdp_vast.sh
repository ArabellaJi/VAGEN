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

# ---------------------------------------------------------------------------
# Patch fsdp_workers.py: skip load_fsdp_model_to_gpu in compute_log_prob.
# With param_offload=True FSDP already does JIT per-layer loading during every
# forward pass.  The explicit pre-load clones the entire 17 GiB model to GPU at
# once, consuming ~26 GiB and causing OOM when SGLang already holds ~55 GiB on
# an A100 80 GiB.  Removing the pre-load cuts FSDP peak to ~4 GiB (JIT).
# Idempotent: second run is a no-op.
# ---------------------------------------------------------------------------
_FSDP_WRK="${BASEDIR}/verl/verl/workers/fsdp_workers.py"
if [ -f "${_FSDP_WRK}" ] && ! grep -q "_vagen_all_patched" "${_FSDP_WRK}"; then
    python3 - "${_FSDP_WRK}" <<'PYPATCH'
import sys, re
path = sys.argv[1]
with open(path) as f:
    lines = f.readlines()
# Functions where load_fsdp_model_to_gpu is required (called before SGLang starts)
KEEP_FUNCS = {'save_checkpoint', '_build_rollout'}
def nearest_func(lines, idx):
    for j in range(idx - 1, max(0, idx - 200), -1):
        m = re.match(r'\s{0,8}def (\w+)', lines[j])
        if m:
            return m.group(1)
    return None
new_lines = []
patched = []
skipped = []
for i, line in enumerate(lines):
    if ('load_fsdp_model_to_gpu(self.actor_module_fsdp)' in line
            and not line.lstrip().startswith('#')):
        fname = nearest_func(lines, i)
        if fname in KEEP_FUNCS:
            skipped.append((i + 1, fname))
            new_lines.append(line)
        else:
            indent = ' ' * (len(line) - len(line.lstrip()))
            new_lines.append(indent + '# vagen: skip GPU pre-load; param_offload JIT is sufficient (_vagen_all_patched)\n')
            new_lines.append(indent + '# ' + line.lstrip())
            new_lines.append(indent + 'pass\n')
            patched.append(i + 1)
    else:
        new_lines.append(line)
if skipped:
    print(f"[vagen] Kept load_fsdp_model_to_gpu in {skipped} (required)", flush=True)
if not patched:
    print(f"[vagen] WARNING: no unpatched load_fsdp_model_to_gpu(self.actor_module_fsdp) found in {path!r}", flush=True)
else:
    with open(path, 'w') as f:
        f.writelines(new_lines)
    import py_compile
    py_compile.compile(path)
    print(f"[vagen] Patched lines {patched}, kept lines {[l for l,_ in skipped]} in {path}. Syntax OK.", flush=True)
PYPATCH
elif [ -f "${_FSDP_WRK}" ]; then
    echo "[vagen] fsdp_workers already fully patched — skipping"
else
    echo "[vagen] WARNING: ${_FSDP_WRK} not found — patch skipped"
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
    actor_rollout_ref.actor.checkpoint.save_contents=['hf_model'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.mode=async \
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
