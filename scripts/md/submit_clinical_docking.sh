#!/bin/bash
# Clinical benchmark docking (saracatinib, zilurgisertib, LDN-193189 vs ACVR1)
# CPU-only Vina — fits in quicktest partition (≈10 min for 4 compounds × 2 receptors)
#SBATCH -p mit_quicktest
#SBATCH -J kd_clinical_dock
#SBATCH -o logs/md/clinical_docking_%j.log
#SBATCH -e logs/md/clinical_docking_%j.err
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 00:15:00

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
# Use kinetidiff-md env: has meeko, vina, rdkit already installed
PYTHON="/home/aaru0302/.conda/envs/kinetidiff-md/bin/python"
mkdir -p "${REPO_ROOT}/logs/md" "${REPO_ROOT}/results/clinical_benchmark"
echo "Node: $(hostname) | Job: ${SLURM_JOB_ID}"
cd "${REPO_ROOT}"
${PYTHON} scripts/clinical_benchmark_docking.py
