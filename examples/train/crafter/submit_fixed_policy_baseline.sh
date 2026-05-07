#!/bin/bash
# Fixed-policy Crafter baseline on validation seeds.
#
# Usage:
#   sbatch -A p32139 --time=01:00:00 examples/train/crafter/submit_fixed_policy_baseline.sh do
#   sbatch -A p32139 --time=01:00:00 examples/train/crafter/submit_fixed_policy_baseline.sh do,move_left

#SBATCH --job-name=crafter_fixed_policy
#SBATCH --partition=gengpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/eiu4164/projects/VAGEN/logs/%x_%j.out
#SBATCH --error=/home/eiu4164/projects/VAGEN/logs/%x_%j.err

set -eo pipefail

POLICY_ACTIONS="${1:-do}"
MAX_ENVS="${2:-}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/eiu4164/projects/VAGEN}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/logs/crafter_baselines}"

mkdir -p "${OUT_DIR}" "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

module purge
module load python-miniconda3/4.10.3
source ~/.bashrc
conda activate vagen_noflash

STAMP="$(date +%Y%m%d_%H%M%S)"
SAFE_ACTIONS="${POLICY_ACTIONS//,/__}"
JSONL="${OUT_DIR}/fixed_${SAFE_ACTIONS}_${SLURM_JOB_ID:-local}_${STAMP}.jsonl"

CMD=(
  python scripts/eval_crafter_fixed_policy.py
  --val-yaml examples/train/crafter/val_crafter_vision.yaml
  --actions "${POLICY_ACTIONS}"
  --concurrency 8
  --render-mode text
  --jsonl "${JSONL}"
)

if [ -n "${MAX_ENVS}" ]; then
  CMD+=(--max-envs "${MAX_ENVS}")
fi

echo "Running fixed policy baseline: ${POLICY_ACTIONS}"
echo "Writing per-episode jsonl: ${JSONL}"
"${CMD[@]}"
