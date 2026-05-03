#!/bin/bash
# ACCESS / NCSA Delta wrapper for VAGEN Navigation GRPO.
#
# This script adapts the non-Slurm navigation launcher to Delta. It handles
# Slurm resources, Delta modules, conda activation, work/cache directories,
# and the mapping from Slurm-allocated GPUs to the two processes:
#
#   GPU A: AI2-THOR navigation server / Unity CloudRendering
#   GPU B: GRPO training + SGLang rollout
#
# Usage:
#   sbatch run_navigation_grpo_access.sh
#   sbatch --export=ALL,CONDITION=quick run_navigation_grpo_access.sh
#   sbatch --partition=gpuH200x8 --gpus-per-node=2 --export=ALL,CONDITION=window3_thumb run_navigation_grpo_access.sh
#
# Useful overrides:
#   ACCESS_ACCOUNT=bfea-delta-gpu
#   PROJECT_ROOT=/u/wji1/VAGEN
#   ACCESS_WORK_ROOT=/work/hdd/bfea/$USER
#   RUN_ROOT=/work/hdd/bfea/$USER/vagen_runs
#   CONDA_ENV_NAME=vagen_noflash
#   CONDITION=quick
#   PREDOWNLOAD_SCENES=1
#
# For a one-GPU smoke test only:
#   sbatch --gpus-per-node=1 --export=ALL,CONDITION=quick,ALLOW_SINGLE_GPU_TRAIN=1 run_navigation_grpo_access.sh

#SBATCH --job-name=nav_grpo
#SBATCH --account=bfea-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=/u/wji1/VAGEN/logs/%x_%j.out
#SBATCH --error=/u/wji1/VAGEN/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wenlanji2026@u.northwestern.edu

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_PROJECT_ROOT="/u/wji1/VAGEN"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/vagen/configs" ]]; then
  DEFAULT_PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -d "${SCRIPT_DIR}/vagen/configs" ]]; then
  DEFAULT_PROJECT_ROOT="${SCRIPT_DIR}"
fi
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"

ACCESS_ACCOUNT="${ACCESS_ACCOUNT:-${SLURM_JOB_ACCOUNT:-bfea-delta-gpu}}"
ACCESS_PROJECT_CODE="${ACCESS_PROJECT_CODE:-${ACCESS_ACCOUNT%%-delta-gpu*}}"
if [[ "${ACCESS_PROJECT_CODE}" == "${ACCESS_ACCOUNT}" ]]; then
  ACCESS_PROJECT_CODE="${ACCESS_ACCOUNT%%-*}"
fi
ACCESS_WORK_ROOT="${ACCESS_WORK_ROOT:-/work/hdd/${ACCESS_PROJECT_CODE}/${USER}}"
RUN_ROOT="${RUN_ROOT:-${ACCESS_WORK_ROOT}/vagen_runs}"
HF_CACHE="${HF_CACHE:-${ACCESS_WORK_ROOT}/hf_cache}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vagen_noflash}"

if [[ ! -d "${PROJECT_ROOT}/vagen/configs" ]]; then
  echo "ERROR: PROJECT_ROOT does not look like the VAGEN repo root: ${PROJECT_ROOT}" >&2
  echo "Set PROJECT_ROOT explicitly or submit from the repo root." >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs" "${RUN_ROOT}" "${HF_CACHE}"
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

if command -v module >/dev/null 2>&1; then
  module reset || true
  if ! command -v conda >/dev/null 2>&1; then
    load_first_available_module anaconda3_gpu pytorch-conda/2.8 || true
  fi
  if ! command -v nvcc >/dev/null 2>&1; then
    load_first_available_module cudatoolkit/25.3_12.8 cudatoolkit cuda/12.8 cuda/11.8.0 || true
  fi
else
  echo "WARNING: module command is unavailable; assuming conda/cuda are already configured." >&2
fi

if [[ -f ~/.bashrc ]]; then
  source ~/.bashrc
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" >/dev/null 2>&1 || true
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is unavailable after module setup." >&2
  module list 2>&1 || true
  exit 1
fi
conda activate "${CONDA_ENV_NAME}"

if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
elif [[ -d /usr/local/cuda ]]; then
  export CUDA_HOME="/usr/local/cuda"
else
  echo "ERROR: nvcc/CUDA_HOME not found after module setup." >&2
  module list 2>&1 || true
  exit 1
fi
export CUDA_PATH="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

detect_gpu_class() {
  local partition="${1:-}"
  local gpu_name="${2:-}"

  case "${partition}" in
    gpuH200*) echo "h200"; return ;;
    gpuH100*) echo "h100"; return ;;
    gpuA100*) echo "a100"; return ;;
    gpuA40*) echo "a40"; return ;;
  esac

  case "${gpu_name}" in
    *H200*) echo "h200"; return ;;
    *H100*) echo "h100"; return ;;
    *A100*) echo "a100"; return ;;
    *A40*) echo "a40"; return ;;
  esac

  echo "unknown"
}

DETECTED_GPU_NAME=""
if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
fi
GPU_CLASS="$(detect_gpu_class "${SLURM_JOB_PARTITION:-}" "${DETECTED_GPU_NAME}")"
case "${GPU_CLASS}" in
  h200|h100) DEFAULT_TORCH_CUDA_ARCH_LIST=9.0 ;;
  a100) DEFAULT_TORCH_CUDA_ARCH_LIST=8.0 ;;
  a40) DEFAULT_TORCH_CUDA_ARCH_LIST=8.6 ;;
  *) DEFAULT_TORCH_CUDA_ARCH_LIST="8.0;9.0" ;;
esac

export PROJECT_ROOT
export RUN_ROOT
export HF_CACHE
export HF_HOME="${HF_HOME:-${HF_CACHE}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUN_ROOT}/xdg_cache}"
export WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/wandb}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${DEFAULT_TORCH_CUDA_ARCH_LIST}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export FLASHINFER_ENABLE_JIT="${FLASHINFER_ENABLE_JIT:-0}"
export FLASHINFER_JIT_WORKER_TIMEOUT="${FLASHINFER_JIT_WORKER_TIMEOUT:-60}"
export VAGEN_SGLANG_INIT_TIMEOUT="${VAGEN_SGLANG_INIT_TIMEOUT:-600}"
mkdir -p "${HUGGINGFACE_HUB_CACHE}" "${XDG_CACHE_HOME}" "${WANDB_DIR}"

JOB_TMP_DEFAULT="${SLURM_TMPDIR:-/tmp/j${SLURM_JOB_ID:-$$}}"
export JOB_TMP="${JOB_TMP:-${JOB_TMP_DEFAULT}}"
export VAGEN_TMP_ROOT="${VAGEN_TMP_ROOT:-${JOB_TMP}}"
export TMPDIR="${TMPDIR:-${JOB_TMP}/t}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export RAY_TMPDIR="${RAY_TMPDIR:-${JOB_TMP}/r}"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"

if [[ -z "${VK_ICD_FILENAMES:-}" ]]; then
  shopt -s nullglob
  icd_files=(/usr/share/vulkan/icd.d/nvidia_icd*.json /etc/vulkan/icd.d/nvidia_icd*.json)
  shopt -u nullglob
  if (( ${#icd_files[@]} > 0 )); then
    export VK_ICD_FILENAMES="${icd_files[0]}"
  fi
fi

ALLOCATED_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "${ALLOCATED_CUDA_DEVICES}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    ALLOCATED_CUDA_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  else
    ALLOCATED_CUDA_DEVICES="0,1"
  fi
fi
IFS=',' read -r -a GPU_IDS <<< "${ALLOCATED_CUDA_DEVICES}"

if (( ${#GPU_IDS[@]} < 2 )) && [[ "${ALLOW_SINGLE_GPU_TRAIN:-0}" != "1" ]]; then
  echo "ERROR: navigation training expects 2 GPUs: one for AI2-THOR, one for training." >&2
  echo "Slurm/CUDA_VISIBLE_DEVICES gave: ${ALLOCATED_CUDA_DEVICES}" >&2
  echo "Use --gpus-per-node=2, or set ALLOW_SINGLE_GPU_TRAIN=1 for a tiny smoke test." >&2
  exit 1
fi

export NAV_CUDA_VISIBLE_DEVICES="${NAV_CUDA_VISIBLE_DEVICES:-${GPU_IDS[0]}}"
if [[ -z "${TRAIN_CUDA_VISIBLE_DEVICES:-}" ]]; then
  if (( ${#GPU_IDS[@]} >= 2 )); then
    export TRAIN_CUDA_VISIBLE_DEVICES="${GPU_IDS[1]}"
  else
    export TRAIN_CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}"
  fi
fi

export NAV_GPU="${NAV_GPU:-0}"
if [[ -z "${TRAIN_GPU:-}" ]]; then
  if (( ${#GPU_IDS[@]} >= 2 )); then
    export TRAIN_GPU=1
  else
    export TRAIN_GPU=0
  fi
fi

echo "========================================================"
echo "ACCESS navigation wrapper"
echo "PROJECT_ROOT:       ${PROJECT_ROOT}"
echo "RUN_ROOT:           ${RUN_ROOT}"
echo "HF_HOME:            ${HF_HOME}"
echo "ACCESS_ACCOUNT:     ${ACCESS_ACCOUNT}"
echo "SLURM_PARTITION:    ${SLURM_JOB_PARTITION:-unset}"
echo "DETECTED_GPU_NAME:  ${DETECTED_GPU_NAME:-unknown}"
echo "GPU_CLASS:          ${GPU_CLASS}"
echo "CUDA_HOME:          ${CUDA_HOME}"
echo "CUDA_VISIBLE_DEVICES from Slurm: ${ALLOCATED_CUDA_DEVICES}"
echo "NAV_GPU logical:    ${NAV_GPU}"
echo "TRAIN_GPU logical:  ${TRAIN_GPU}"
echo "NAV_CUDA_VISIBLE:   ${NAV_CUDA_VISIBLE_DEVICES}"
echo "TRAIN_CUDA_VISIBLE: ${TRAIN_CUDA_VISIBLE_DEVICES}"
echo "CONDA_ENV_NAME:     ${CONDA_ENV_NAME}"
echo "CONDITION:          ${CONDITION:-quick}"
echo "TMPDIR:             ${TMPDIR}"
echo "VK_ICD_FILENAMES:   ${VK_ICD_FILENAMES:-unset}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
echo "========================================================"
module list 2>&1 || true
which python
nvidia-smi || true

exec bash "${PROJECT_ROOT}/run_navigation_grpo_vast.sh"
