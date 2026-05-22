# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

VAGEN is a multi-turn reinforcement learning framework for training Vision-Language Model (VLM) agents, published at NeurIPS 2025. It frames agentic tasks as POMDPs and trains models with PPO/GRPO using the veRL backend for distributed training and SGLang for inference.

The repo has two layers:
- **`vagen/`** — VAGEN's own code (environments, agent loops, trainer, evaluation)
- **`verl/`** — veRL submodule (FSDP workers, rollout engine, PPO core algorithms)

## Installation

```bash
conda create -n vagen python=3.12 -y && conda activate vagen
git submodule update --init --recursive
cd verl && USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh && pip install --no-deps -e . && cd ..
pip install -e .
pip install "trl==0.26.2" "uvicorn<0.41"
```

## Running Training

All training is launched via shell scripts in `examples/train/<env>/`. The entry point is always:

```bash
cd VAGEN
python3 -m vagen.main_ppo \
    --config-path=vagen/configs \
    --config-name=vagen_multiturn \
    data.train_files=... data.val_files=... ...
```

The config system is Hydra: `vagen/configs/vagen_multiturn.yaml` merges with veRL's `ppo_trainer.yaml`. All shell script parameters override Hydra defaults via dotpath syntax.

```bash
# Standard Sokoban GRPO (4 GPUs)
bash examples/train/sokoban/train_grpo_qwen25vl3b.sh

# Single-GPU smoke test with custom overrides
MODEL_PATH=/path/to/model N_GPUS=1 bash examples/train/sokoban/train_grpo_qwen25_7b_mem1_pomdp_vast.sh \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4
```

## Running Evaluation

```bash
# Sokoban evaluation
bash examples/evaluate/sokoban/run_eval.sh

# FrozenLake with sglang backend
bash examples/evaluate/frozenlake/sglang/eval_qwen25_vl_3b.sh
```

Or directly:
```bash
python3 -m vagen.evaluate.run_eval --config examples/evaluate/sokoban/config.yaml
```

## Architecture

### Training Data Flow

```
vagen/main_ppo.py          # Entry point: Ray init, creates RayPPOTrainer
vagen/ray_trainer.py       # Outer training loop: rollout → filter → update
  ↓ rollout
verl/workers/fsdp_workers.py (ActorRolloutWorker)
  ↓ agent loop
vagen/agent_loop/gym_agent_loop.py  # Per-trajectory async loop: call SGLang, step env, repeat
  ↓ env
vagen/envs/<env>/<env>_env.py       # GymImageEnv subclass
  ↓ update
verl/workers/actor/dp_actor.py      # FSDP actor: backward + optimizer step
```

### Environment Interface

All environments inherit from `GymImageEnv` (→ `GymBaseEnv`) and implement three async methods:
- `reset(seed)` → `(obs, info)`
- `step(action_str)` → `(obs, reward, done, info)`
- `system_prompt()` → `str`

Observations return `{"obs_str": "...", "multi_modal_input": {"<image>": [PIL.Image, ...]}}` for vision, or just `{"obs_str": "..."}` for text.

To add a new environment: implement the class, register it in `vagen/configs/env_registry.yaml`, create `train.yaml`/`val.yaml` dataset configs, and write a training script.

### Two Training Paradigms

**Concat mode** (`trainer.concat_multi_turn=True`, default): entire trajectory concatenated as one sequence. Context: `sys + obs_0 + res_0 + obs_1 + res_1 + ...`

**No-concat mode** (`trainer.concat_multi_turn=False`): each turn is an independent training instance. Requires `algorithm.adv_estimator=no_concat_gae`.

### Memory / History Control

- `trainer.history_window_size`: turns visible during rollout (0=no memory, k=sliding k-turn window, -1=full history)
- `trainer.hires_window_size`: most recent k turns at full image resolution; older turns use `thumbnail_scale`

### Remote Environments

Heavy environments (ManiSkill, Navigation) run as separate HTTP services. The client `vagen/envs_remote/GymImageEnvClient` connects to a `GymService` (FastAPI) running in another process. Use `RemoteEnv` in `env_registry.yaml`.

## Key Config Parameters

Main config: `vagen/configs/vagen_multiturn.yaml`. Critical overrides for single-GPU runs:

| Parameter | Purpose |
|---|---|
| `actor_rollout_ref.rollout.gpu_memory_utilization` | SGLang KV cache budget (0.4–0.6 typical) |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | Offload FSDP params to CPU during update |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | Offload Adam states to CPU |
| `actor_rollout_ref.rollout.free_cache_engine` | Release SGLang KV cache after rollout |
| `trainer.history_window_size` | Agent memory depth during rollout |
| `data.max_prompt_length` / `data.max_response_length` | Token budget per turn |
| `filter.enable` | Enable reward-variance top-p trajectory filtering (recommended for GRPO) |

## Known Issues

- **B200/RTX 6000 Pro**: add `+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer +actor_rollout_ref.rollout.engine_kwargs.sglang.mm_attention_backend=triton_attn`
- **SGLang workers die silently**: pin `uvicorn<0.41`
- **Single A100 80GB with param_offload=True + optimizer_offload=True**: Adam device mismatch (grad on CUDA, exp_avg on CPU). Fix: add `actor_rollout_ref.actor.fsdp_config.optimizer_offload=False` to keep Adam states on CUDA with the grads.
- **Disk weight sync fills disk**: if `VAGEN_SGLANG_WEIGHT_SYNC_METHOD=disk` env var is set, each step writes ~14 GiB. Remove that env var to use in-memory IPC instead.
