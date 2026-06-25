#!/bin/bash
# =============================================================================
# sync_to_drive_bundle.sh — Assemble the < 100 GB Drive-upload bundle
# Run from: ORCD Engaging login node (after analysis array completes)
# =============================================================================
# What ends up in the bundle (stripped water, strided 10×):
#   - Water-stripped + strided analysis trajectories  (~3–5 GB)
#   - Publication-quality figures PNG + SVG           (~50 MB)
#   - CSVs: RMSD, RMSF, H-bonds, MM-GBSA, selectivity, master results
#   - Bundle manifest JSON
#
# What stays on ORCD Pool (NOT uploaded):
#   - Full solvated production DCDs (~30–50 GB)
#   - System XML + topology PDB files
#   - Checkpoint files (.chk)
#
# Usage (run from login node):
#   bash scripts/md/sync_to_drive_bundle.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
SCRATCH="${HOME}/orcd/scratch/kinetidiff-md"
BUNDLE="${HOME}/kinetidiff-md-bundle"
CONFIG="${REPO_ROOT}/configs/simulation.yaml"

echo "================================================"
echo "KinetiDiff MD Drive-upload bundle assembler"
echo "================================================"
echo "SCRATCH: ${SCRATCH}"
echo "BUNDLE:  ${BUNDLE}"

mkdir -p "${BUNDLE}/trajectories_stripped" \
         "${BUNDLE}/figures" \
         "${BUNDLE}/analysis_csvs"

# ── 1. Strip and stride all trajectories ────────────────────────────────────
echo "[1/4] Stripping and striding trajectories..."
module purge
module load miniforge
conda activate kinetidiff-md

python3 - <<'PYEOF'
import sys
sys.path.insert(0, "${REPO_ROOT}/src")
from pathlib import Path
from kinetidiff.molecular_dynamics.analysis.postprocess import strip_and_stride
from omegaconf import OmegaConf

cfg = OmegaConf.load("${CONFIG}")
cfg_dict = OmegaConf.to_container(cfg, resolve=True)
scratch = Path("${SCRATCH}")
bundle  = Path("${BUNDLE}/trajectories_stripped")
stride  = cfg_dict["analysis"]["bundle_stride"]

for traj in sorted((scratch / "trajectories").glob("*_rep*.dcd")):
    lig_rec = "_".join(traj.stem.split("_")[:2])
    top = scratch / "systems" / f"topology_{lig_rec}.pdb"
    if not top.exists():
        print(f"WARN: topology missing for {traj.stem}, skipping.")
        continue
    # Infer resname from ligand ID
    lig_id = traj.stem.split("_")[0]  # e.g. L1
    lig_num = lig_id.replace("L", "0")
    resname = f"L{lig_num.zfill(2)}"
    strip_and_stride(top, traj, bundle, stride=stride, lig_resname=resname)

print("Stripping complete.")
PYEOF

# ── 2. Collect figures ───────────────────────────────────────────────────────
echo "[2/4] Collecting figures..."
find "${SCRATCH}/figures" -name "*.png" -o -name "*.svg" 2>/dev/null \
    | xargs -I{} cp {} "${BUNDLE}/figures/" || echo "  (no figures found yet)"

# ── 3. Collect analysis CSVs ─────────────────────────────────────────────────
echo "[3/4] Collecting analysis CSVs..."
find "${SCRATCH}/analysis" -name "*.csv" -o -name "*.json" 2>/dev/null \
    | xargs -I{} cp {} "${BUNDLE}/analysis_csvs/" || echo "  (no CSVs found yet)"

# Also copy master results if present
MASTER="${SCRATCH}/master_results.csv"
[ -f "${MASTER}" ] && cp "${MASTER}" "${BUNDLE}/"

# ── 4. Write bundle manifest ─────────────────────────────────────────────────
echo "[4/4] Writing bundle manifest..."
python3 - <<'PYEOF'
import sys
sys.path.insert(0, "${REPO_ROOT}/src")
from pathlib import Path
from kinetidiff.molecular_dynamics.analysis.postprocess import make_drive_bundle_manifest

bundle = Path("${BUNDLE}")
make_drive_bundle_manifest(
    bundle_dir=bundle,
    analysis_dir=bundle / "trajectories_stripped",
    figures_dir=bundle / "figures",
    master_csv=bundle / "master_results.csv",
)
PYEOF

# ── 5. Measure bundle size ────────────────────────────────────────────────────
echo ""
echo "Bundle size:"
du -sh "${BUNDLE}"
echo ""
echo "File counts:"
find "${BUNDLE}" -type f | wc -l
echo ""
echo "================================================"
echo "MANUAL DRIVE UPLOAD:"
echo "  scp -r ${USER}@engaging-login.mit.edu:${BUNDLE}/ ."
echo "  Then upload kinetidiff-md-bundle/ folder to Google Drive."
echo "================================================"
echo "KEEP ON ORCD POOL (do NOT upload — too large):"
echo "  ${SCRATCH}/trajectories/   (full solvated DCDs)"
echo "  ${SCRATCH}/systems/        (topology PDBs + system XMLs)"
echo "  ${SCRATCH}/checkpoints/    (checkpoint files)"
echo "================================================"
