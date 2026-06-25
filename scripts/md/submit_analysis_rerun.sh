#!/bin/bash
# =============================================================================
# submit_analysis_rerun.sh — Re-run analysis with corrected trajectory.py +
#                            mmgbsa.py (post-16041526 bug fixes)
# Managed by: engagebot-hpc
# =============================================================================
# Changes vs submit_analysis_array.sh:
#   - conda activation uses explicit miniforge source path (robust in non-
#     interactive SLURM shells; avoids `module load miniforge` + conda init
#     dependency that can fail on certain node images)
#   - Time limit reduced to 02:00:00 (observed wall time ~13 min; 2h is safe)
#   - trajectory.py: --receptor flag removed (not a real CLI argument)
#   - mmgbsa.py: now uses vacuum interaction energy (fixed from broken OBC2)
#   - Log prefix changed to analysis_rerun_%A_%a for easy diff vs prior run
#
# Usage:
#   # Quick test — 1 task only:
#   sbatch -a 0 scripts/md/submit_analysis_rerun.sh
#
#   # Full 12-task array:
#   sbatch scripts/md/submit_analysis_rerun.sh
#
#   # Quick test on mit_quicktest (15-min limit, override partition):
#   sbatch -p mit_quicktest -t 00:15:00 -a 0-1 scripts/md/submit_analysis_rerun.sh
# =============================================================================
#SBATCH -p mit_normal
#SBATCH -J kd_analysis_rerun
#SBATCH -o logs/md/analysis_rerun_%A_%a.log
#SBATCH -e logs/md/analysis_rerun_%A_%a.err
#SBATCH -a 0-11                   # 12 tasks: 3 leads × 2 receptors × 2 replicas
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00               # Observed wall time ~15-20 min; 2h provides margin

set -euo pipefail

# ── Environment activation ────────────────────────────────────────────────────
module purge
module load miniforge
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kinetidiff-md

# Use the env's python directly as a fallback guard
PYTHON="/home/aaru0302/.conda/envs/kinetidiff-md/bin/python"

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="${SLURM_SUBMIT_DIR:-/orcd/home/002/aaru0302/KinetiDiff}"
CONFIG="${REPO_ROOT}/configs/simulation.yaml"
SCRATCH="/home/aaru0302/orcd/scratch"
TRAJ_DIR="${SCRATCH}/kinetidiff-md/trajectories"
SYS_DIR="${SCRATCH}/kinetidiff-md/systems"
ANALYSIS_OUT="${SCRATCH}/kinetidiff-md/analysis"

mkdir -p "${ANALYSIS_OUT}"
export ORCD_SCRATCH="${SCRATCH}"

echo "========================================================"
echo "Job:       ${SLURM_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID} / $((SLURM_ARRAY_TASK_COUNT - 1))"
echo "Node:      $(hostname)"
echo "Repo root: ${REPO_ROOT}"
echo "========================================================"

# ── Build run matrix and resolve this task's runs ────────────────────────────
# Emits one line per run: "L1 WT 1"
RUN_INFO=$(${PYTHON} - <<PYEOF
import sys, os
sys.path.insert(0, "${REPO_ROOT}/src")
from kinetidiff.molecular_dynamics.simulation.runner import build_run_matrix
from omegaconf import OmegaConf

cfg = OmegaConf.load("${CONFIG}")
cfg_dict = OmegaConf.to_container(cfg, resolve=True)
matrix = build_run_matrix(cfg_dict["campaign"])

task_id   = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
my_runs = matrix[task_id::num_tasks]

for lig_id, receptor, replica_id, seed in my_runs:
    print(f"{lig_id} {receptor} {replica_id}")
PYEOF
)

echo "This task will analyse: ${RUN_INFO}"

# ── Per-run analysis loop ─────────────────────────────────────────────────────
while IFS=" " read -r LIG REC REP; do
    TAG="${LIG}_${REC}_rep${REP}"
    TRAJ="${TRAJ_DIR}/${TAG}.dcd"
    TOP="${SYS_DIR}/topology_${LIG}_${REC}.pdb"
    OUT="${ANALYSIS_OUT}/${TAG}"
    mkdir -p "${OUT}"

    # Ligand resname in simulation PDB: L1→L01, L2→L02, L3→L03
    LIG_DIGIT=$(echo "${LIG}" | tr -d 'L')
    LIG_RESNAME="L0${LIG_DIGIT}"

    # Lookup SMILES from campaign config
    SMILES=$(${PYTHON} -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${CONFIG}')
print(cfg.campaign.ligands.${LIG}.smiles)
")

    echo ""
    echo "-- [$(date '+%H:%M:%S')] Starting ${TAG} --"
    echo "   Trajectory: ${TRAJ}"
    echo "   Topology:   ${TOP}"
    echo "   Output:     ${OUT}"
    echo "   LIG resname: ${LIG_RESNAME}"

    if [[ ! -f "${TRAJ}" ]]; then
        echo "ERROR: trajectory not found: ${TRAJ}" >&2
        exit 1
    fi
    if [[ ! -f "${TOP}" ]]; then
        echo "ERROR: topology not found: ${TOP}" >&2
        exit 1
    fi

    # ── 1. Trajectory analysis (RMSD, RMSF, H-bonds, pocket contacts,
    #        ACVR1 structural — now MDTraj-only, no MDAnalysis frame loops)
    ${PYTHON} -m kinetidiff.molecular_dynamics.analysis.trajectory \
        --topology    "${TOP}" \
        --trajectory  "${TRAJ}" \
        --out-dir     "${OUT}" \
        --lig-resname "${LIG_RESNAME}" \
        --min-hbond-occupancy 0.10 \
        --stride 2 \
        --frame-time-ps 10.0

    # ── 2. MM-GBSA: vacuum protein-ligand interaction energy
    #       100 OpenMM single-point evals, Reference platform, ~3 min
    ${PYTHON} -m kinetidiff.molecular_dynamics.analysis.mmgbsa \
        --topology    "${TOP}" \
        --trajectory  "${TRAJ}" \
        --smiles      "${SMILES}" \
        --lig-resname "${LIG_RESNAME}" \
        --out-csv     "${OUT}/mmgbsa.csv" \
        --n-snapshots 100

    echo "-- [$(date '+%H:%M:%S')] Done: ${TAG} --"

done <<< "${RUN_INFO}"

echo ""
echo "Analysis task ${SLURM_ARRAY_TASK_ID} complete."
