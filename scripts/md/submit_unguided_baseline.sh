#!/bin/bash
# =============================================================================
# submit_unguided_baseline.sh — Generate n=1000 unguided GCDM molecules,
#                               Vina-dock all of them against WT ACVR1.
#
# Output: results/no_guidance/unguided_1k_molecules.smi
#         results/no_guidance/unguided_1k_vina_docked.csv
#
# This produces the statistical baseline for the STS paper:
#   "Unguided GCDM generates molecules with mean Vina X ± Y kcal/mol (n=1000)"
#   vs guided: "Vina-guided GCDM achieves mean X ± Y kcal/mol (n=Z, p<0.001)"
#
# Usage:
#   sbatch scripts/md/submit_unguided_baseline.sh
# =============================================================================
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH -J kd_unguided_1k
#SBATCH -o logs/md/unguided_baseline_%j.log
#SBATCH -e logs/md/unguided_baseline_%j.err
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 06:00:00   # n=1000 unguided ≈ 2-3 h on RTX Pro 6000; docking ≈ 1 h

set -euo pipefail

module purge
module load cuda   # Blackwell SM 12.0 requires CUDA 12+ system libs

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
CKPT="${REPO_ROOT}/data/checkpoints/bindingmoad_ca_cond_gcpnet.ckpt"
PDB="${REPO_ROOT}/data/structures/prepped/ACVR1_WT_prepped.pdb"
RECEPTOR_PDBQT="${REPO_ROOT}/results/redocked_poses/receptor_WT.pdbqt"
OUT_DIR="${REPO_ROOT}/results/no_guidance"
PYTHON="/home/aaru0302/.conda/envs/tsm_gcdm/bin/python"

# Rocky Linux 8 system libstdc++ lacks GLIBCXX_3.4.29 needed by openbabel.
# Prepend the conda env's newer libstdc++ so it's found first.
export LD_LIBRARY_PATH="/home/aaru0302/.conda/envs/tsm_gcdm/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${OUT_DIR}" "${REPO_ROOT}/logs/md"

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: checkpoint not found at ${CKPT}" >&2; exit 1
fi

echo "Job: ${SLURM_JOB_ID} | Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES:-0}"

# ── Step 1: Generate 1000 unguided molecules — NO guidance, NO QED filter ────
echo ""
echo "[Step 1] Generating 1000 unguided GCDM molecules..."
${PYTHON} - <<PYEOF
import sys, time
sys.path.insert(0, "${REPO_ROOT}/src")
sys.path.insert(0, "${REPO_ROOT}/src/kinetidiff/gcdm")

import torch
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")
if device == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Load GCDM
from kinetidiff.gcdm.lightning_modules import LigandPocketDDPM
import pathlib
torch.serialization.add_safe_globals([pathlib.PosixPath])
print("  Loading GCDM checkpoint...")
model = LigandPocketDDPM.load_from_checkpoint(
    "${CKPT}", map_location=device, weights_only=False)
model = model.to(device).eval()
print("  GCDM loaded.")

# Find pocket residues from WT structure (8 Å around pocket center)
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
pdb = PDBParser(QUIET=True).get_structure("", "${PDB}")[0]
center = np.array([-17.66, -13.65, 38.41])
pocket_ids = []
for chain in pdb:
    for res in chain:
        if "CA" in res:
            if np.linalg.norm(res["CA"].get_coord() - center) <= 8.0:
                pocket_ids.append(f"{chain.id}:{res.id[1]}")
print(f"  Pocket residues within 8 Å: {len(pocket_ids)}")

# Generate in batches of 50
from rdkit import Chem
N_TARGET = 1000
BATCH_SIZE = 50
all_smiles = []
start = time.time()
n_batches = (N_TARGET + BATCH_SIZE - 1) // BATCH_SIZE

for b in range(n_batches):
    n = min(BATCH_SIZE, N_TARGET - len(all_smiles))
    try:
        mols = model.generate_ligands(
            "${PDB}", n_samples=n, pocket_ids=pocket_ids,
            ref_ligand=None, sanitize=True, largest_frag=True, relax_iter=0)
        for mol in mols:
            if mol is not None:
                smi = Chem.MolToSmiles(mol)
                if smi:
                    all_smiles.append(smi)
        elapsed = time.time() - start
        rate = (b + 1) * BATCH_SIZE / elapsed
        eta = (n_batches - b - 1) * BATCH_SIZE / rate
        print(f"  Batch {b+1}/{n_batches}: {len(all_smiles)} valid | {elapsed:.0f}s elapsed | ETA {eta:.0f}s")
    except Exception as e:
        print(f"  Batch {b+1} failed: {e}")

# Save all SMILES (no QED/MW filter — want unbiased baseline)
out = Path("${OUT_DIR}") / "unguided_1k_molecules.smi"
with open(out, "w") as f:
    f.write("\n".join(sorted(set(all_smiles))))
print(f"\n  Saved {len(all_smiles)} SMILES → {out}")
PYEOF

N=$(wc -l < "${OUT_DIR}/unguided_1k_molecules.smi")
echo "Step 1 complete: ${N} molecules generated."

# ── Step 2: Vina-dock all molecules against WT ACVR1 ─────────────────────────
echo ""
echo "[Step 2] Vina-docking ${N} molecules (exhaustiveness=8 for speed)..."
${PYTHON} - <<'PYEOF'
import csv, sys, tempfile, statistics
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina

REPO_ROOT = Path(__file__) if False else Path("REPO_PLACEHOLDER")
REPO_ROOT = Path("/orcd/home/002/aaru0302/KinetiDiff")
SMILES_FILE = REPO_ROOT / "results/no_guidance/unguided_1k_molecules.smi"
RECEPTOR    = REPO_ROOT / "results/redocked_poses/receptor_WT.pdbqt"
OUT_CSV     = REPO_ROOT / "results/no_guidance/unguided_1k_vina_docked.csv"

POCKET = (-17.66, -13.65, 38.41)
BOX    = (25.0, 25.0, 25.0)
SEED   = 42

smiles_list = [s.strip() for s in open(SMILES_FILE) if s.strip()]
print(f"  Docking {len(smiles_list)} molecules...")

rows = []
for i, smi in enumerate(smiles_list):
    if i % 100 == 0:
        print(f"  Progress: {i}/{len(smiles_list)} ({len(rows)} docked OK)")
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0: continue
        AllChem.MMFFOptimizeMolecule(mol)
        prep = MoleculePreparation()
        pdbqt = None
        for setup in prep.prepare(mol):
            s, ok, _ = PDBQTWriterLegacy.write_string(setup)
            if ok: pdbqt = s; break
        if pdbqt is None: continue
        with tempfile.NamedTemporaryFile(suffix=".pdbqt", mode="w", delete=False) as f:
            f.write(pdbqt); tmp = Path(f.name)
        v = Vina(sf_name="vina", seed=SEED, verbosity=0)
        v.set_receptor(str(RECEPTOR))
        v.set_ligand_from_file(str(tmp))
        v.compute_vina_maps(center=list(POCKET), box_size=list(BOX))
        v.dock(exhaustiveness=8, n_poses=1)
        score = float(v.energies(n_poses=1)[0][0])
        tmp.unlink(missing_ok=True)
        rows.append({"smiles": smi, "vina_score_kcal_mol": score, "source": "unguided_gcdm"})
    except Exception:
        pass

with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["smiles","vina_score_kcal_mol","source"])
    w.writeheader(); w.writerows(rows)

scores = [r["vina_score_kcal_mol"] for r in rows]
print(f"\n  Docked {len(rows)}/{len(smiles_list)} successfully")
print(f"  Vina: mean={statistics.mean(scores):.2f}  "
      f"std={statistics.stdev(scores):.2f}  "
      f"best={min(scores):.2f}  median={statistics.median(scores):.2f}")
print(f"  Results → {OUT_CSV}")
PYEOF

echo ""
echo "Unguided baseline complete."
echo "  SMILES: ${OUT_DIR}/unguided_1k_molecules.smi"
echo "  Docked: ${OUT_DIR}/unguided_1k_vina_docked.csv"
