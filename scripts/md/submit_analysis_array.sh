#!/bin/bash
# =============================================================================
# submit_analysis_array.sh — CPU analysis array (RMSD/RMSF/H-bonds/MM-GBSA)
# Managed by: engagebot-hpc
# =============================================================================
# Runs on mit_normal (CPU only, no GPU needed for MDAnalysis/MDTraj).
# 30 tasks: one per (lead, receptor, replica) — same matrix as production.
# Dependency: must run AFTER all production tasks complete.
#
# Usage:
#   PROD_JOB=<production_array_jobid>
#   sbatch --dependency=afterok:${PROD_JOB} scripts/md/submit_analysis_array.sh
# =============================================================================
#SBATCH -p mit_normal
#SBATCH -J kd_analysis
#SBATCH -o logs/md/analysis_%A_%a.log
#SBATCH -e logs/md/analysis_%A_%a.err
#SBATCH -a 0-5                    # 6 tasks: 3 leads × 2 receptors × 1 replica
#SBATCH -c 8                      # Increased for faster MDAnalysis trajectory parsing
#SBATCH --mem=64G
#SBATCH -t 06:00:00               # Analysis (RMSD/RMSF/H-bonds/MM-GBSA) ≈ 30-60 min per run

set -euo pipefail

module purge
module load miniforge
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kinetidiff-md

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
CONFIG="${KD_MD_CONFIG:-${REPO_ROOT}/configs/simulation.yaml}"
SCRATCH="${HOME}/orcd/scratch"
ANALYSIS_OUT="${SCRATCH}/kinetidiff-md/analysis"

mkdir -p "${ANALYSIS_OUT}"

export ORCD_SCRATCH="${SCRATCH}"

# ── Build the 30-run matrix and find this task's run ──────────────────────────
# Python helper: emits "L1 WT 2" on stdout for the task's (lig, rec, rep)
RUN_INFO=$(python3 - <<PYEOF
import sys
sys.path.insert(0, "${REPO_ROOT}/src")
from kinetidiff.molecular_dynamics.simulation.runner import build_run_matrix
from omegaconf import OmegaConf
import os

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

# ── Run analysis for each run assigned to this task ───────────────────────────
while IFS=" " read -r LIG REC REP; do
    TAG="${LIG}_${REC}_rep${REP}"
    TRAJ="${SCRATCH}/kinetidiff-md/trajectories/${TAG}.dcd"
    TOP="${SCRATCH}/kinetidiff-md/systems/topology_${LIG}_${REC}.pdb"
    SYS_XML="${SCRATCH}/kinetidiff-md/systems/system_${LIG}_${REC}.xml"
    OUT="${ANALYSIS_OUT}/${TAG}"
    mkdir -p "${OUT}"

    echo "-- Analysing ${TAG} --"

    LIG_RESNAME=$(python3 -c "from omegaconf import OmegaConf; print(OmegaConf.load('${CONFIG}').campaign.ligands.${LIG}.resname)")

    # Lookup SMILES for this ligand from the campaign config
    SMILES=$(python3 -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${CONFIG}')
print(cfg.campaign.ligands.${LIG}.smiles)
")

    # RMSD / RMSF / H-bonds / Pocket contacts
    # NOTE: OpenMM DCD header DELTA=0 (corrupted). MDTraj defaults to 1 ps/frame.
    # Actual frame time: report_interval(25000) × timestep(2fs) = 50 ps.
    # With stride=10: effective frame time = 50 * 10 = 500 ps per stride.
    python -m kinetidiff.molecular_dynamics.analysis.trajectory \
        --topology "${TOP}" \
        --trajectory "${TRAJ}" \
        --out-dir "${OUT}" \
        --lig-resname "${LIG_RESNAME}" \
        --min-hbond-occupancy 0.10 \
        --stride 10 \
        --frame-time-ps 500.0

    # MM-GBSA: ΔG_bind = E_complex(OBC2) − E_protein(OBC2) − E_ligand(OBC2)
    python -m kinetidiff.molecular_dynamics.analysis.mmgbsa \
        --topology "${TOP}" \
        --trajectory "${TRAJ}" \
        --smiles "${SMILES}" \
        --lig-resname "${LIG_RESNAME}" \
        --out-csv "${OUT}/mmgbsa.csv" \
        --n-snapshots 100

    echo "-- Done: ${TAG} --"

done <<< "${RUN_INFO}"

echo "Analysis task ${SLURM_ARRAY_TASK_ID} complete."
