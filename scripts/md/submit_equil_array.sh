#!/bin/bash
# =============================================================================
# submit_equil_array.sh — Equilibration job array for all 6 systems
# Managed by: engagebot-hpc (partition selection, resource flags)
# =============================================================================
# 6 systems (3 leads × 2 receptors), ~2–3 h each on RTX Pro 6000.
# RTX Pro 6000 Blackwell (SM 12.0): CUDA 11.8 PTX fails — must use OpenCL.
# SLURM packs 2 tasks per 2-GPU node (or up to 8 per 8-GPU node) automatically.
#
# Usage:
#   sbatch scripts/md/submit_equil_array.sh
# =============================================================================
#SBATCH -p mit_preemptable
#SBATCH -J kd_equil
#SBATCH -o logs/md/equil_%A_%a.log
#SBATCH -e logs/md/equil_%A_%a.err
#SBATCH -a 0-5                    # 6 tasks: 3 leads × 2 receptors
#SBATCH --gres=gpu:rtx_pro_6000:1 # 1× RTX Pro 6000 per task; SLURM packs 2/node on 2-GPU nodes
#SBATCH -c 8                      # 8 CPUs per task
#SBATCH --mem=32G
#SBATCH -t 06:00:00               # Equilibration ≈2-3h
#SBATCH --requeue                 # requeue on preemption

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load miniforge
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kinetidiff-md

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
CONFIG="${KD_MD_CONFIG:-${REPO_ROOT}/configs/simulation.yaml}"
SCRATCH="${HOME}/orcd/scratch"
LOG_DIR="${REPO_ROOT}/logs/md"

mkdir -p "${LOG_DIR}"

export ORCD_SCRATCH="${SCRATCH}"
export OPENMM_DEFAULT_PLATFORM="OpenCL"  # RTX Pro 6000 SM 12.0 incompatible with CUDA 11.8 PTX
# SLURM sets CUDA_VISIBLE_DEVICES to isolate the GPU assigned to this task;
# NVIDIA OpenCL ICD respects this mask, so no explicit device index needed.

echo "=============================================="
echo "SLURM_JOB_ID:        ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "SLURM_NODELIST:      ${SLURM_NODELIST}"
echo "Repo root:           ${REPO_ROOT}"
echo "Config:              ${CONFIG}"
echo "Scratch:             ${SCRATCH}"
echo "=============================================="

# ── Run equilibration for this task's system ──────────────────────────────────
# runner.py maps SLURM_ARRAY_TASK_ID (0-9) → (lead, receptor) pair.
# With 30 runs in the full matrix and 10 equil tasks, this covers one
# (lead, receptor) per task; production replicas share the same equil state.
python -m kinetidiff.molecular_dynamics.simulation.runner \
    --config "${CONFIG}" \
    --repo-root "${REPO_ROOT}" \
    --equil-only

echo "Equilibration task ${SLURM_ARRAY_TASK_ID} completed."

# ── Memory audit hint (run manually after job completes) ─────────────────────
# sacct -j ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} -o JobID,State,ReqMem,MaxRSS --units=G
