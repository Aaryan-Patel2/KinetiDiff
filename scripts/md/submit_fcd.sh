#!/bin/bash
# FCD computation on existing guided molecules (CPU-only, ~5 min)
#SBATCH -p mit_quicktest
#SBATCH -J kd_fcd
#SBATCH -o logs/md/fcd_%j.log
#SBATCH -e logs/md/fcd_%j.err
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:15:00

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
PYTHON="/home/aaru0302/.conda/envs/tsm_gcdm/bin/python"
mkdir -p "${REPO_ROOT}/logs/md"
echo "Node: $(hostname) | Job: ${SLURM_JOB_ID}"
cd "${REPO_ROOT}"
${PYTHON} scripts/compute_fcd.py
