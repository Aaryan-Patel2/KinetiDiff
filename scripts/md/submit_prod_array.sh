#!/bin/bash
# =============================================================================
# submit_prod_array.sh — Production MD job array (30 runs × 100 ns)
# Managed by: engagebot-hpc (partition selection, resource flags)
# Written by: rsi-compute-manager (Python logic)
# =============================================================================
# 12 tasks: 3 leads × 2 receptors × 2 replicas
# 750 ns per run: ~7.5 days on H200/RTX-Pro-6000-Blackwell (OpenCL); ~3.5 days if CUDA 12+ native.
# Uses mit_preemptable + --requeue for checkpoint-resume across ~13 × 12 h allocations per task.
#
# Dependency: must run AFTER submit_equil_array.sh (add --dependency=afterok:<JOBID>).
#
# Usage:
#   EQUIL_JOB=$(sbatch scripts/md/submit_equil_array.sh | awk '{print $NF}')
#   sbatch --dependency=afterok:${EQUIL_JOB} scripts/md/submit_prod_array.sh
# =============================================================================
#SBATCH -p mit_preemptable
#SBATCH -J kd_prod
#SBATCH -o logs/md/prod_%A_%a.log
#SBATCH -e logs/md/prod_%A_%a.err
#SBATCH -a 0-5%4                  # 6 tasks (1 replica); %4 = mit_preemptable hard limit, leaves mit_normal_gpu free
#SBATCH --gres=gpu:rtx_pro_6000:1        # 1× RTX Pro 6000 per task; SLURM packs tasks per node
#SBATCH -c 8                             # 8 CPUs per task
#SBATCH --mem=32G
#SBATCH -t 12:00:00                      # 12 h per allocation; --requeue handles full 100 ns
#SBATCH --requeue                        # requeue on preemption

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
# SLURM sets CUDA_VISIBLE_DEVICES to isolate this task's GPU; NVIDIA OpenCL ICD respects it.

echo "=============================================="
echo "SLURM_JOB_ID:        ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "SLURM_ARRAY_TASK_COUNT: ${SLURM_ARRAY_TASK_COUNT}"
echo "SLURM_NODELIST:      ${SLURM_NODELIST}"
echo "Repo root:           ${REPO_ROOT}"
echo "Scratch:             ${SCRATCH}"
echo "=============================================="

# ── Production run for this task ──────────────────────────────────────────────
# SIGTERM is caught by production.py → checkpoint saved → exit(0) → SLURM requeues.
python -m kinetidiff.molecular_dynamics.simulation.runner \
    --config "${CONFIG}" \
    --repo-root "${REPO_ROOT}"

echo "Production task ${SLURM_ARRAY_TASK_ID} finished (or checkpointed for requeue)."

# ── Memory audit (run manually): ─────────────────────────────────────────────
# sacct -j ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} -o JobID,State,ReqMem,MaxRSS --units=G

# ── Archive completed trajectories to pool after the full array finishes: ────
# (run from login node once all tasks complete)
# cp -r ~/orcd/scratch/kinetidiff-md/trajectories/ ~/orcd/pool/kinetidiff-md/
