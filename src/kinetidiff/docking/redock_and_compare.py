#!/usr/bin/env python3
"""
Unified Redocking and Comparison Pipeline
==========================================
Pre-ranks molecules by existing predicted pKd scores, redocks the top N
with identical Vina parameters, and generates comparison tables.

Usage:
    python redock_and_compare.py \
        --receptor-pdbqt ../data/converted_structures/receptor_siteA.pdbqt \
        --results-dir ../results/ \
        --output-dir ../results/redocking/ \
        --seed 42 \
        --n-workers 10 \
        --top-n 3000 \
        --save-top-k 100
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem
from rdkit.Chem import QED, AllChem, Descriptors, rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DockingConfig:
    """Docking parameters - identical for all molecules"""
    center_x: float = 24.87
    center_y: float = -12.54
    center_z: float = 38.40
    size_x: float = 22.0
    size_y: float = 22.0
    size_z: float = 22.0
    exhaustiveness: int = 4
    num_modes: int = 1
    energy_range: float = 1.0
    seed: int = 42
    timeout_seconds: int = 300
    R: float = 1.987   # cal/(mol*K)
    T: float = 298.0   # K


# Guidance categories and their SMILES file locations
GUIDANCE_CATEGORIES = {
    'HNN-Denovo': {
        'smiles_files': [
            'hnn_guided/generated5k/gcdm_guided_molecules.smi',
            'hnn_guided/generated_2k_run1/gcdm_guided_molecules.smi',
            'hnn_guided/generated_2k_gpu1/gcdm_guided_molecules.smi',
        ],
        'format': 'tsv',
    },
    'Vina-Direct': {
        'smiles_files': [
            'vina_guided/generated_local_10k/molecules.smi',
        ],
        'format': 'auto',
    },
    'Multi-Objective': {
        'smiles_files': [
            'multi_objective/multi_objective_5k_A100/molecules.smi',
        ],
        'format': 'auto',
    },
    'No-Guidance': {
        'smiles_files': [
            'no_guidance/generated10/gcdm_guided_molecules.smi',
        ],
        'format': 'tsv',
    },
}


# ============================================================================
# SMILES Loading
# ============================================================================

def load_smiles_file(filepath: str, fmt: str = 'auto') -> list[tuple[str, str]]:
    """Load SMILES from file. Returns list of (smiles, mol_id) tuples."""
    molecules = []

    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return molecules

    with open(filepath) as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return molecules

    if fmt == 'auto':
        first_line = lines[0]
        fmt = 'tsv' if '\t' in first_line else 'smiles_only'

    base_name = Path(filepath).parent.name

    for i, line in enumerate(lines):
        if fmt == 'tsv':
            parts = line.split('\t')
            smiles = parts[0].strip()
            mol_id = parts[1].strip() if len(parts) > 1 else f"{base_name}_{i:05d}"
        else:
            smiles = line.strip()
            mol_id = f"{base_name}_{i:05d}"

        if smiles and smiles != 'SMILES':
            molecules.append((smiles, mol_id))

    return molecules


def load_all_molecules(results_dir: str) -> dict[str, list[tuple[str, str]]]:
    """Load all molecules organized by guidance category"""
    categories = {}

    for cat_name, cat_info in GUIDANCE_CATEGORIES.items():
        molecules = []
        for smi_file in cat_info['smiles_files']:
            filepath = os.path.join(results_dir, smi_file)
            mols = load_smiles_file(filepath, cat_info['format'])
            logger.info(f"  {cat_name}: loaded {len(mols)} from {smi_file}")
            molecules.extend(mols)

        categories[cat_name] = molecules
        logger.info(f"  {cat_name} total: {len(molecules)} molecules")

    return categories


# ============================================================================
# Pre-Ranking: Load Predicted Scores
# ============================================================================

def load_predicted_scores(results_dir: str) -> dict[str, float]:
    """
    Load pre-computed pKd scores for all molecules from their prediction files.
    Returns dict mapping SMILES -> predicted pKd.
    """
    scores = {}
    results_path = Path(results_dir)

    # HNN-Denovo: gcdm_guided_predictions.json -> predicted_affinity
    for subdir in ['generated5k', 'generated_2k_run1', 'generated_2k_gpu1']:
        json_path = results_path / 'hnn_guided' / subdir / 'gcdm_guided_predictions.json'
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            for mol in data:
                smi = mol.get('smiles', '')
                if smi:
                    scores[smi] = mol.get('predicted_affinity', 0.0)

    # Vina-Direct: results.json -> pkd_approx
    vina_json = results_path / 'vina_guided' / 'generated_local_10k' / 'results.json'
    if vina_json.exists():
        with open(vina_json) as f:
            data = json.load(f)
        for mol in data.get('molecules', []):
            smi = mol.get('smiles', '')
            if smi:
                scores[smi] = mol.get('pkd_approx', 0.0)

    # Multi-Objective: summary.csv -> pKd
    csv_path = results_path / 'multi_objective' / 'multi_objective_5k_A100' / 'summary.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            smi = row.get('SMILES', '')
            if smi:
                scores[smi] = row.get('pKd', 0.0)

    # No-Guidance: gcdm_guided_predictions.json -> predicted_affinity
    json_path = results_path / 'no_guidance' / 'generated10' / 'gcdm_guided_predictions.json'
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        for mol in data:
            smi = mol.get('smiles', '')
            if smi:
                scores[smi] = mol.get('predicted_affinity', 0.0)

    return scores


def prerank_and_select(
    categories: dict[str, list[tuple[str, str]]],
    scores: dict[str, float],
    top_n: int,
) -> dict[str, list[tuple[str, str]]]:
    """
    Rank all molecules across categories by predicted pKd, take top N.
    Always includes all No-Guidance molecules for baseline.
    Returns filtered categories dict.
    """
    # Build flat list with scores
    all_mols = []
    for cat_name, molecules in categories.items():
        for smiles, mol_id in molecules:
            pkd = scores.get(smiles, 0.0)
            all_mols.append((pkd, cat_name, smiles, mol_id))

    # Sort by pKd descending
    all_mols.sort(key=lambda x: x[0], reverse=True)

    # Always include all No-Guidance molecules (baseline)
    baseline = [(pkd, cat, smi, mid) for pkd, cat, smi, mid in all_mols if cat == 'No-Guidance']
    non_baseline = [(pkd, cat, smi, mid) for pkd, cat, smi, mid in all_mols if cat != 'No-Guidance']

    # Take top N from non-baseline, then add baseline
    selected = non_baseline[:top_n - len(baseline)] + baseline

    # Rebuild categories dict
    filtered = {cat: [] for cat in categories}
    for pkd, cat, smi, mid in selected:
        filtered[cat].append((smi, mid))

    return filtered


# ============================================================================
# Molecule Processing
# ============================================================================

def smiles_to_3d_mol(smiles: str, seed: int = 42) -> Chem.Mol | None:
    """Convert SMILES to 3D molecule with optimized conformer"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda x: x.GetNumHeavyAtoms())

        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.useSmallRingTorsions = True
        result = AllChem.EmbedMolecule(mol, params)

        if result == -1:
            params2 = AllChem.ETKDGv2()
            params2.randomSeed = seed
            result = AllChem.EmbedMolecule(mol, params2)

        if result == -1:
            result = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=seed)

        if result == -1 or mol.GetNumConformers() == 0:
            return None

        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass

        return mol
    except Exception:
        return None


def mol_to_pdbqt(mol: Chem.Mol) -> str | None:
    """Convert RDKit mol to PDBQT string using meeko"""
    try:
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        for setup in mol_setups:
            pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
            if is_ok:
                return pdbqt_string
        return None
    except Exception:
        return None


def compute_properties(mol: Chem.Mol) -> dict[str, float]:
    """Compute all molecular properties"""
    try:
        mol_no_h = Chem.RemoveHs(mol)
        props = {
            'mw': Descriptors.MolWt(mol_no_h),
            'logp': Descriptors.MolLogP(mol_no_h),
            'qed': QED.qed(mol_no_h),
            'sa_score': sascorer.calculateScore(mol_no_h),
            'hbd': rdMolDescriptors.CalcNumHBD(mol_no_h),
            'hba': rdMolDescriptors.CalcNumHBA(mol_no_h),
            'tpsa': Descriptors.TPSA(mol_no_h),
            'rot_bonds': Descriptors.NumRotatableBonds(mol_no_h),
            'heavy_atoms': mol_no_h.GetNumHeavyAtoms(),
            'rings': rdMolDescriptors.CalcNumRings(mol_no_h),
        }
        props['lipinski_violations'] = sum([
            props['mw'] > 500,
            props['logp'] > 5,
            props['hbd'] > 5,
            props['hba'] > 10,
        ])
        return props
    except Exception:
        return {}


def vina_to_pkd(vina_affinity: float, R: float = 1.987, T: float = 298.0) -> float:
    """Convert Vina binding affinity (kcal/mol) to pKd"""
    try:
        delta_g_cal = vina_affinity * 1000.0
        kd = np.exp(delta_g_cal / (R * T))
        return -np.log10(kd)
    except Exception:
        return np.nan


# ============================================================================
# Docking
# ============================================================================

def run_vina_docking(
    ligand_pdbqt: str,
    receptor_pdbqt: str,
    config: DockingConfig,
    work_dir: str
) -> float | None:
    """Run AutoDock Vina and return best affinity"""
    try:
        ligand_file = os.path.join(work_dir, "ligand.pdbqt")
        with open(ligand_file, 'w') as f:
            f.write(ligand_pdbqt)

        output_file = os.path.join(work_dir, "docked.pdbqt")

        cmd = [
            'vina',
            '--receptor', receptor_pdbqt,
            '--ligand', ligand_file,
            '--out', output_file,
            '--center_x', str(config.center_x),
            '--center_y', str(config.center_y),
            '--center_z', str(config.center_z),
            '--size_x', str(config.size_x),
            '--size_y', str(config.size_y),
            '--size_z', str(config.size_z),
            '--exhaustiveness', str(config.exhaustiveness),
            '--num_modes', str(config.num_modes),
            '--energy_range', str(config.energy_range),
            '--seed', str(config.seed),
            '--cpu', '1',
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=config.timeout_seconds
        )

        if result.returncode != 0:
            return None

        for line in result.stdout.split('\n'):
            stripped = line.strip()
            if stripped.startswith('1'):
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        pass

        return None
    except (subprocess.TimeoutExpired, Exception):
        return None


def dock_single_molecule(args: tuple) -> dict[str, Any]:
    """Dock a single molecule (designed for ProcessPoolExecutor)"""
    smiles, mol_id, category, receptor_pdbqt, config = args

    result = {
        'mol_id': mol_id,
        'smiles': smiles,
        'category': category,
        'docked': False,
        'vina_score': None,
        'pkd': None,
    }

    try:
        mol = smiles_to_3d_mol(smiles, seed=config.seed)
        if mol is None:
            result['error'] = 'embedding_failed'
            return result

        props = compute_properties(mol)
        result.update(props)

        pdbqt = mol_to_pdbqt(mol)
        if pdbqt is None:
            result['error'] = 'pdbqt_failed'
            return result

        with tempfile.TemporaryDirectory() as work_dir:
            affinity = run_vina_docking(pdbqt, receptor_pdbqt, config, work_dir)

        if affinity is not None:
            result['docked'] = True
            result['vina_score'] = affinity
            result['pkd'] = vina_to_pkd(affinity, config.R, config.T)
        else:
            result['error'] = 'docking_failed'

    except Exception as e:
        result['error'] = str(e)

    return result


# ============================================================================
# Markdown Table Generation
# ============================================================================

def generate_comparison_tables(df: pd.DataFrame, config: DockingConfig) -> str:
    """Generate static markdown comparison tables"""
    lines = []
    lines.append("# Redocking Comparison: Guidance Types")
    lines.append("")
    lines.append("All molecules redocked with identical AutoDock Vina parameters:")
    lines.append("- **Receptor**: ACVR1 wild-type (3MTF)")
    lines.append(f"- **Center**: ({config.center_x}, {config.center_y}, {config.center_z})")
    lines.append(f"- **Box**: {config.size_x} x {config.size_y} x {config.size_z} A")
    lines.append(f"- **Exhaustiveness**: {config.exhaustiveness}")
    lines.append(f"- **Seed**: {config.seed}")
    lines.append("")

    docked = df[df['docked']].copy()

    # ---- Table 1: Summary Statistics ----
    lines.append("## Table 1: Summary Statistics by Guidance Type")
    lines.append("")
    lines.append("| Guidance Type | N Total | N Docked | Dock Rate | Mean Vina (kcal/mol) | Std Vina | Mean pKd | Std pKd | Mean SA | Mean QED | Mean MW | Mean LogP |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

    for cat in GUIDANCE_CATEGORIES:
        cat_all = df[df['category'] == cat]
        cat_docked = docked[docked['category'] == cat]
        n_total = len(cat_all)
        n_docked = len(cat_docked)

        if n_total == 0:
            continue

        dock_rate = f"{n_docked/n_total*100:.1f}%"

        if n_docked > 0:
            mean_vina = f"{cat_docked['vina_score'].mean():.2f}"
            std_vina = f"{cat_docked['vina_score'].std():.2f}"
            mean_pkd = f"{cat_docked['pkd'].mean():.2f}"
            std_pkd = f"{cat_docked['pkd'].std():.2f}"
            mean_sa = f"{cat_docked['sa_score'].mean():.2f}"
            mean_qed = f"{cat_docked['qed'].mean():.3f}"
            mean_mw = f"{cat_docked['mw'].mean():.1f}"
            mean_logp = f"{cat_docked['logp'].mean():.2f}"
        else:
            mean_vina = std_vina = mean_pkd = std_pkd = "N/A"
            mean_sa = mean_qed = mean_mw = mean_logp = "N/A"

        lines.append(f"| {cat} | {n_total} | {n_docked} | {dock_rate} | {mean_vina} | {std_vina} | {mean_pkd} | {std_pkd} | {mean_sa} | {mean_qed} | {mean_mw} | {mean_logp} |")

    # Overall row
    n_total = len(df)
    n_docked = len(docked)
    if n_docked > 0:
        lines.append(f"| **Overall** | {n_total} | {n_docked} | {n_docked/n_total*100:.1f}% | {docked['vina_score'].mean():.2f} | {docked['vina_score'].std():.2f} | {docked['pkd'].mean():.2f} | {docked['pkd'].std():.2f} | {docked['sa_score'].mean():.2f} | {docked['qed'].mean():.3f} | {docked['mw'].mean():.1f} | {docked['logp'].mean():.2f} |")
    lines.append("")

    # ---- Table 2: Top 10 per category ----
    lines.append("## Table 2: Top 10 Molecules Per Category (by Vina Score)")
    lines.append("")

    for cat in GUIDANCE_CATEGORIES:
        cat_docked = docked[docked['category'] == cat].sort_values('vina_score')
        if len(cat_docked) == 0:
            continue

        lines.append(f"### {cat}")
        lines.append("")

        top10 = cat_docked.head(10)
        lines.append("| Rank | Mol ID | SMILES | Vina (kcal/mol) | pKd | SA | QED | MW | LogP | HBD | HBA |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

        for rank, (_, row) in enumerate(top10.iterrows(), 1):
            smiles_short = row['smiles'][:50] + "..." if len(str(row['smiles'])) > 50 else row['smiles']
            lines.append(
                f"| {rank} | {row['mol_id']} | `{smiles_short}` | "
                f"{row['vina_score']:.2f} | {row['pkd']:.2f} | "
                f"{row.get('sa_score', 'N/A'):.2f} | {row.get('qed', 'N/A'):.3f} | "
                f"{row.get('mw', 'N/A'):.1f} | {row.get('logp', 'N/A'):.2f} | "
                f"{int(row.get('hbd', 0))} | {int(row.get('hba', 0))} |"
            )
        lines.append("")

    # ---- Table 3: Drug-likeness Pass Rates ----
    lines.append("## Table 3: Drug-likeness Filter Pass Rates")
    lines.append("")
    lines.append("| Guidance Type | N Docked | Lipinski Pass (<=1 viol) | SA < 6 | QED > 0.3 | pKd > 6.0 | All Pass |")
    lines.append("|---|---|---|---|---|---|---|")

    for cat in GUIDANCE_CATEGORIES:
        cat_docked = docked[docked['category'] == cat]
        n = len(cat_docked)

        if n == 0:
            continue

        lipinski = (cat_docked['lipinski_violations'] <= 1).sum()
        sa_pass = (cat_docked['sa_score'] < 6).sum()
        qed_pass = (cat_docked['qed'] > 0.3).sum()
        pkd_pass = (cat_docked['pkd'] > 6.0).sum()
        all_pass = (
            (cat_docked['lipinski_violations'] <= 1) &
            (cat_docked['sa_score'] < 6) &
            (cat_docked['qed'] > 0.3) &
            (cat_docked['pkd'] > 6.0)
        ).sum()

        lines.append(
            f"| {cat} | {n} | "
            f"{lipinski} ({lipinski/n*100:.1f}%) | "
            f"{sa_pass} ({sa_pass/n*100:.1f}%) | "
            f"{qed_pass} ({qed_pass/n*100:.1f}%) | "
            f"{pkd_pass} ({pkd_pass/n*100:.1f}%) | "
            f"{all_pass} ({all_pass/n*100:.1f}%) |"
        )
    lines.append("")

    # ---- Table 4: Best Molecule Per Category ----
    active_cats = [cat for cat in GUIDANCE_CATEGORIES if len(docked[docked['category'] == cat]) > 0]
    lines.append("## Table 4: Best Molecule Per Category (by Vina Score)")
    lines.append("")
    lines.append("| Property | " + " | ".join(active_cats) + " |")
    lines.append("|---|" + "|".join(["---"] * len(active_cats)) + "|")

    best_mols = {}
    for cat in active_cats:
        cat_docked = docked[docked['category'] == cat].sort_values('vina_score')
        best_mols[cat] = cat_docked.iloc[0]

    properties = [
        ('Mol ID', 'mol_id', 's'),
        ('SMILES', 'smiles', '.50s'),
        ('Vina (kcal/mol)', 'vina_score', '.2f'),
        ('pKd', 'pkd', '.2f'),
        ('SA Score', 'sa_score', '.2f'),
        ('QED', 'qed', '.3f'),
        ('MW (Da)', 'mw', '.1f'),
        ('LogP', 'logp', '.2f'),
        ('HBD', 'hbd', 'd'),
        ('HBA', 'hba', 'd'),
        ('TPSA', 'tpsa', '.1f'),
        ('Rot. Bonds', 'rot_bonds', 'd'),
        ('Heavy Atoms', 'heavy_atoms', 'd'),
        ('Lipinski Viol.', 'lipinski_violations', 'd'),
    ]

    for prop_name, prop_key, fmt in properties:
        vals = []
        for cat in active_cats:
            mol = best_mols[cat]
            val = mol.get(prop_key, 'N/A')
            if val is None or (isinstance(val, float) and np.isnan(val)):
                vals.append("N/A")
            elif prop_key == 'smiles':
                s = str(val)[:50]
                vals.append(f"`{s}`")
            elif 'd' in fmt:
                vals.append(str(int(val)))
            else:
                vals.append(f"{val:{fmt}}")
        lines.append(f"| {prop_name} | " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by `docking/redock_and_compare.py`*")

    return "\n".join(lines)


def generate_top_k_table(df: pd.DataFrame, k: int) -> str:
    """Generate markdown table of top K molecules across all categories"""
    docked = df[df['docked']].sort_values('vina_score').head(k)

    lines = []
    lines.append(f"# Top {k} Molecules by Vina Docking Score")
    lines.append("")
    lines.append("Ranked by actual AutoDock Vina score (lower = stronger binding).")
    lines.append("")
    lines.append("| Rank | Category | Mol ID | SMILES | Vina (kcal/mol) | pKd | SA | QED | MW | LogP | HBD | HBA | TPSA | Lipinski |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    for rank, (_, row) in enumerate(docked.iterrows(), 1):
        smi = row['smiles'][:45] + "..." if len(str(row['smiles'])) > 45 else row['smiles']
        lines.append(
            f"| {rank} | {row['category']} | {row['mol_id']} | `{smi}` | "
            f"{row['vina_score']:.2f} | {row['pkd']:.2f} | "
            f"{row.get('sa_score', 0):.2f} | {row.get('qed', 0):.3f} | "
            f"{row.get('mw', 0):.1f} | {row.get('logp', 0):.2f} | "
            f"{int(row.get('hbd', 0))} | {int(row.get('hba', 0))} | "
            f"{row.get('tpsa', 0):.1f} | {int(row.get('lipinski_violations', 0))} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("*Generated by `docking/redock_and_compare.py`*")

    return "\n".join(lines)


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified Redocking and Comparison Pipeline'
    )
    parser.add_argument('--receptor-pdbqt', type=str, required=True,
                        help='Path to receptor PDBQT file')
    parser.add_argument('--results-dir', type=str, default='../results/',
                        help='Results directory with guidance type subdirs')
    parser.add_argument('--output-dir', type=str, default='../results/redocking/',
                        help='Output directory for redocking results')
    parser.add_argument('--n-workers', type=int, default=4,
                        help='Number of parallel docking workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Vina random seed for reproducibility')
    parser.add_argument('--top-n', type=int, default=0,
                        help='Pre-rank and dock only top N molecules (0 = all)')
    parser.add_argument('--save-top-k', type=int, default=100,
                        help='Save top K results after redocking (default 100)')
    parser.add_argument('--center', type=str, default='24.87,-12.54,38.40',
                        help='Binding site center x,y,z')
    parser.add_argument('--box-size', type=str, default='22,22,22',
                        help='Search box size x,y,z')
    parser.add_argument('--exhaustiveness', type=int, default=4,
                        help='Vina exhaustiveness parameter')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Per-molecule docking timeout (seconds)')

    args = parser.parse_args()

    cx, cy, cz = [float(x) for x in args.center.split(',')]
    bx, by, bz = [float(x) for x in args.box_size.split(',')]

    config = DockingConfig(
        center_x=cx, center_y=cy, center_z=cz,
        size_x=bx, size_y=by, size_z=bz,
        exhaustiveness=args.exhaustiveness,
        seed=args.seed,
        timeout_seconds=args.timeout,
    )

    if not os.path.exists(args.receptor_pdbqt):
        logger.error(f"Receptor not found: {args.receptor_pdbqt}")
        sys.exit(1)

    try:
        subprocess.run(['vina', '--version'], capture_output=True, timeout=10)
    except FileNotFoundError:
        logger.error("AutoDock Vina not found on PATH.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Step 1: Load molecules ----
    logger.info("=" * 60)
    logger.info("UNIFIED REDOCKING & COMPARISON PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Seed: {config.seed}")
    logger.info(f"Center: ({config.center_x}, {config.center_y}, {config.center_z})")
    logger.info(f"Box: ({config.size_x}, {config.size_y}, {config.size_z})")
    logger.info(f"Exhaustiveness: {config.exhaustiveness}")
    logger.info(f"Workers: {args.n_workers}")
    if args.top_n > 0:
        logger.info(f"Pre-ranking: top {args.top_n} by predicted pKd")
    logger.info(f"Saving top {args.save_top_k} after redocking")
    logger.info("")

    logger.info("[Step 1] Loading molecules by guidance category...")
    categories = load_all_molecules(args.results_dir)

    total = sum(len(v) for v in categories.values())
    logger.info(f"Total molecules across all categories: {total}")

    # ---- Step 1.5: Pre-rank and select top N ----
    if args.top_n > 0:
        logger.info("")
        logger.info(f"[Step 1.5] Pre-ranking by predicted pKd, selecting top {args.top_n}...")
        scores = load_predicted_scores(args.results_dir)
        logger.info(f"  Loaded predicted scores for {len(scores)} molecules")

        categories = prerank_and_select(categories, scores, args.top_n)

        for cat_name, mols in categories.items():
            if len(mols) > 0:
                logger.info(f"  {cat_name}: {len(mols)} selected")

        selected_total = sum(len(v) for v in categories.values())
        logger.info(f"  Total selected for docking: {selected_total}")

    # ---- Step 2: Dock molecules ----
    logger.info("")
    logger.info("[Step 2] Docking molecules with identical parameters...")

    work_items = []
    for cat_name, molecules in categories.items():
        for smiles, mol_id in molecules:
            work_items.append((smiles, mol_id, cat_name, args.receptor_pdbqt, config))

    logger.info(f"Total docking jobs: {len(work_items)}")

    all_results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(dock_single_molecule, item): item for item in work_items}

        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)
            completed += 1

            if completed % 50 == 0 or completed == len(work_items):
                n_docked = sum(1 for r in all_results if r['docked'])
                elapsed_per = completed  # just for logging
                logger.info(f"  Progress: {completed}/{len(work_items)} ({n_docked} docked)")

    # ---- Step 3: Aggregate and analyze ----
    logger.info("")
    logger.info("[Step 3] Aggregating results...")

    df = pd.DataFrame(all_results)

    for cat in GUIDANCE_CATEGORIES:
        cat_df = df[df['category'] == cat]
        cat_docked = cat_df[cat_df['docked']]
        n_total = len(cat_df)
        n_docked = len(cat_docked)
        if n_total == 0:
            continue
        if n_docked > 0:
            logger.info(
                f"  {cat}: {n_docked}/{n_total} docked, "
                f"mean Vina={cat_docked['vina_score'].mean():.2f}, "
                f"mean pKd={cat_docked['pkd'].mean():.2f}"
            )
        else:
            logger.info(f"  {cat}: {n_docked}/{n_total} docked")

    # ---- Step 4: Generate outputs ----
    logger.info("")
    logger.info("[Step 4] Generating outputs...")

    # Save full CSV
    csv_path = os.path.join(args.output_dir, 'redocking_all_results.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"  Saved full results: {csv_path}")

    # Comparison tables
    md_content = generate_comparison_tables(df, config)
    md_path = os.path.join(args.output_dir, 'comparison_tables.md')
    with open(md_path, 'w') as f:
        f.write(md_content)
    logger.info(f"  Saved comparison tables: {md_path}")

    # Top K output
    docked_df = df[df['docked']].sort_values('vina_score')
    top_k = docked_df.head(args.save_top_k)

    top_csv_path = os.path.join(args.output_dir, f'top_{args.save_top_k}.csv')
    top_k.to_csv(top_csv_path, index=False)
    logger.info(f"  Saved top {args.save_top_k}: {top_csv_path}")

    top_md_content = generate_top_k_table(df, args.save_top_k)
    top_md_path = os.path.join(args.output_dir, f'top_{args.save_top_k}.md')
    with open(top_md_path, 'w') as f:
        f.write(top_md_content)
    logger.info(f"  Saved top {args.save_top_k} table: {top_md_path}")

    # Print tables to stdout
    print("\n" + md_content)
    print("\n" + top_md_content)

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Results: {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
