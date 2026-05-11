#!/usr/bin/env python3
"""
Comprehensive Molecular Docking and Ranking Pipeline
=====================================================
Performs AutoDock Vina docking on generated molecules and creates
multi-objective rankings with full analysis.

Date: January 2026
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Contrib.SA_Score import sascorer

# Setup logging
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
    """Configuration for docking parameters"""
    # Receptor and binding site
    receptor_pdb: str = ""
    center_x: float = 24.87
    center_y: float = -12.54
    center_z: float = 38.4
    size_x: float = 22.0
    size_y: float = 22.0
    size_z: float = 22.0
    
    # Vina parameters
    exhaustiveness: int = 8
    num_modes: int = 9
    energy_range: float = 3.0
    
    # Processing
    n_workers: int = 4
    timeout_seconds: int = 300
    
    # Physical constants for pKd calculation
    R: float = 1.987  # cal/(mol·K)
    T: float = 298.0  # K


# ============================================================================
# Molecule Processing
# ============================================================================

def smiles_to_3d_mol(smiles: str, n_conformers: int = 1) -> Chem.Mol | None:
    """Convert SMILES to 3D molecule with optimized conformer"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Check for multiple fragments - keep only largest
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) > 1:
            # Keep largest fragment by heavy atom count
            mol = max(frags, key=lambda x: x.GetNumHeavyAtoms())
            logger.debug(f"Multiple fragments detected, keeping largest ({mol.GetNumHeavyAtoms()} heavy atoms)")
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D conformer with multiple fallback methods
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useSmallRingTorsions = True
        
        result = AllChem.EmbedMolecule(mol, params)
        
        if result == -1:
            # Try ETKDG (version 2)
            params2 = AllChem.ETKDGv2()
            params2.randomSeed = 42
            result = AllChem.EmbedMolecule(mol, params2)
        
        if result == -1:
            # Try basic embedding with random coords
            result = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
        
        if result == -1 or mol.GetNumConformers() == 0:
            logger.debug(f"All embedding methods failed for: {smiles[:50]}...")
            return None
        
        # Optimize with MMFF, but don't fail if optimization doesn't converge
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except:
            # Try UFF as fallback
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except:
                pass  # Use unoptimized conformer
        
        return mol
    except Exception as e:
        logger.debug(f"Failed to convert SMILES to 3D: {smiles[:50]}... - {e}")
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
    except Exception as e:
        logger.debug(f"Failed to convert mol to PDBQT: {e}")
        return None


def calculate_sa_score(mol: Chem.Mol) -> float:
    """Calculate synthetic accessibility score (1-10, lower is better)"""
    try:
        # Need to remove Hs for SA score calculation
        mol_no_h = Chem.RemoveHs(mol)
        return sascorer.calculateScore(mol_no_h)
    except Exception as e:
        logger.debug(f"Failed to calculate SA score: {e}")
        return 10.0  # Return worst score on failure


def vina_to_pkd(vina_affinity: float, R: float = 1.987, T: float = 298.0) -> float:
    """
    Convert Vina binding affinity (kcal/mol) to pKd
    
    ΔG = -RT ln(Kd)
    Kd = exp(-ΔG / RT)
    pKd = -log10(Kd)
    
    Note: Vina gives ΔG in kcal/mol, need to convert to cal/mol for R
    """
    try:
        # Convert kcal to cal
        delta_g_cal = vina_affinity * 1000.0
        
        # Calculate Kd in molar
        kd = np.exp(delta_g_cal / (R * T))
        
        # Convert to pKd
        pkd = -np.log10(kd)
        return pkd
    except:
        return np.nan


# ============================================================================
# Receptor Preparation
# ============================================================================

def prepare_receptor(pdb_path: str, output_dir: str) -> str:
    """Prepare receptor PDBQT file from PDB using proper PDBQT format"""
    output_pdbqt = os.path.join(output_dir, "receptor.pdbqt")
    
    # Check if receptor already exists (pre-prepared)
    if os.path.exists(output_pdbqt):
        logger.info(f"Using existing receptor PDBQT: {output_pdbqt}")
        return output_pdbqt
    
    # Check if there's a receptor in the lowercase docking folder
    alternative_receptor = os.path.join(os.path.dirname(output_dir), 'docking', 'docking_results_batch', 'receptor.pdbqt')
    if os.path.exists(alternative_receptor):
        shutil.copy(alternative_receptor, output_pdbqt)
        logger.info(f"Copied receptor PDBQT from: {alternative_receptor}")
        return output_pdbqt
    
    try:
        # Read PDB and create proper PDBQT with vdW and Elec columns
        with open(pdb_path) as f:
            pdb_content = f.read()
        
        pdbqt_lines = []
        pdbqt_lines.append(f"REMARK  Name = {pdb_path}")
        pdbqt_lines.append("REMARK                            x       y       z     vdW  Elec       q    Type")
        pdbqt_lines.append("REMARK                         _______ _______ _______ _____ _____    ______ ____")
        
        for line in pdb_content.split('\n'):
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_name = line[12:16].strip()
                
                # Determine AD4 type
                if atom_name[0] == 'C':
                    ad_type = 'C'
                elif atom_name[0] == 'N':
                    ad_type = 'NA'  # Nitrogen acceptor for backbone N
                elif atom_name[0] == 'O':
                    ad_type = 'OA'  # Oxygen acceptor
                elif atom_name[0] == 'S':
                    ad_type = 'SA'  # Sulfur acceptor
                elif atom_name[0] == 'H':
                    ad_type = 'HD'
                elif atom_name[0] == 'P':
                    ad_type = 'P'
                else:
                    ad_type = 'A'
                
                # Format: columns 1-54 from PDB, then vdW/Elec/charge/type
                # Must match exact format: 0.00  0.00    +0.000 TYPE
                base = line[:54].ljust(54)
                pdbqt_line = f"{base}  0.00  0.00    +0.000 {ad_type.ljust(2)}"
                pdbqt_lines.append(pdbqt_line)
            elif line.startswith('TER') or line.startswith('END'):
                pdbqt_lines.append(line)
        
        with open(output_pdbqt, 'w') as f:
            f.write('\n'.join(pdbqt_lines))
        
        logger.info(f"Created receptor PDBQT: {output_pdbqt}")
        return output_pdbqt
        
    except Exception as e:
        logger.error(f"Failed to prepare receptor: {e}")
        raise


# ============================================================================
# Docking
# ============================================================================

def run_vina_docking(
    ligand_pdbqt: str,
    receptor_pdbqt: str,
    config: DockingConfig,
    work_dir: str
) -> tuple[float | None, str | None]:
    """
    Run AutoDock Vina docking
    Returns: (best_affinity, output_pdbqt_content)
    """
    try:
        # Write ligand PDBQT to temp file
        ligand_file = os.path.join(work_dir, "ligand.pdbqt")
        with open(ligand_file, 'w') as f:
            f.write(ligand_pdbqt)
        
        output_file = os.path.join(work_dir, "docked.pdbqt")
        
        # Build Vina command
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
        ]
        
        # Run Vina
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds
        )
        
        if result.returncode != 0:
            logger.debug(f"Vina failed: {result.stderr}")
            return None, None
        
        # Parse output to get best affinity
        best_affinity = None
        for line in result.stdout.split('\n'):
            if line.strip().startswith('1'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        best_affinity = float(parts[1])
                        break
                    except ValueError:
                        pass
        
        # Read output poses
        output_content = None
        if os.path.exists(output_file):
            with open(output_file) as f:
                output_content = f.read()
        
        return best_affinity, output_content
        
    except subprocess.TimeoutExpired:
        logger.debug("Vina docking timed out")
        return None, None
    except Exception as e:
        logger.debug(f"Docking error: {e}")
        return None, None


def dock_molecule(
    smiles: str,
    mol_id: str,
    receptor_pdbqt: str,
    config: DockingConfig,
    output_dir: str
) -> dict[str, Any]:
    """
    Full docking pipeline for a single molecule
    Returns dict with results
    """
    result = {
        'mol_id': mol_id,
        'smiles': smiles,
        'docking_success': False,
        'vina_affinity': None,
        'actual_pkd': None,
        'sa_score': None,
        'error': None
    }
    
    try:
        # Convert SMILES to 3D
        mol = smiles_to_3d_mol(smiles)
        if mol is None:
            result['error'] = 'Failed to generate 3D conformer'
            return result
        
        # Calculate SA score
        result['sa_score'] = calculate_sa_score(mol)
        
        # Convert to PDBQT
        pdbqt = mol_to_pdbqt(mol)
        if pdbqt is None:
            result['error'] = 'Failed to generate PDBQT'
            return result
        
        # Create temp directory for docking
        with tempfile.TemporaryDirectory() as work_dir:
            # Run docking
            affinity, docked_poses = run_vina_docking(
                pdbqt, receptor_pdbqt, config, work_dir
            )
            
            if affinity is not None:
                result['docking_success'] = True
                result['vina_affinity'] = affinity
                result['actual_pkd'] = vina_to_pkd(affinity, config.R, config.T)
                
                # Save best pose
                if docked_poses:
                    pose_file = os.path.join(output_dir, 'poses', f'{mol_id}_docked.pdbqt')
                    os.makedirs(os.path.dirname(pose_file), exist_ok=True)
                    with open(pose_file, 'w') as f:
                        f.write(docked_poses)
            else:
                result['error'] = 'Docking failed'
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


# ============================================================================
# Data Loading
# ============================================================================

def load_molecules_from_json(json_path: str, source_name: str) -> list[dict]:
    """Load molecules from JSON file with source tracking"""
    molecules = []
    
    try:
        with open(json_path) as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # Standard format: list of molecule dicts
            for i, mol in enumerate(data):
                mol['source'] = source_name
                mol['mol_id'] = f"{source_name}_{i:04d}"
                molecules.append(mol)
                
        elif isinstance(data, dict):
            # Multi-objective format with nested structure
            if 'top_molecules' in data:
                for i, mol in enumerate(data['top_molecules']):
                    # Normalize keys
                    normalized = {
                        'smiles': mol.get('smiles'),
                        'mol_weight': mol.get('mol_weight'),
                        'qed': mol.get('qed'),
                        'logp': mol.get('logp'),
                        'predicted_affinity': mol.get('pkd', mol.get('predicted_affinity')),
                        'affinity_uncertainty': mol.get('uncertainty', mol.get('affinity_uncertainty')),
                        'predicted_koff': mol.get('koff', mol.get('predicted_koff')),
                        'residence_time': mol.get('residence_time'),
                        'source': source_name,
                        'mol_id': f"{source_name}_{i:04d}",
                        # Additional fields from multi-objective
                        'total_score': mol.get('total_score'),
                        'sa_value': mol.get('sa_value'),  # May already have SA
                    }
                    molecules.append(normalized)
        
        logger.info(f"Loaded {len(molecules)} molecules from {source_name}")
        
    except Exception as e:
        logger.error(f"Failed to load {json_path}: {e}")
    
    return molecules


def scan_and_load_all_molecules(generation_dir: str) -> list[dict]:
    """Scan generation directory and load all molecules"""
    all_molecules = []
    
    # Define expected file patterns
    file_patterns = [
        ('gcdm_guided_predictions.json', 'standard'),
        ('final_results.json', 'multi_objective'),
    ]
    
    # Scan subfolders
    for subfolder in os.listdir(generation_dir):
        subfolder_path = os.path.join(generation_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        
        # Find JSON file
        for pattern, format_type in file_patterns:
            json_path = os.path.join(subfolder_path, pattern)
            if os.path.exists(json_path):
                molecules = load_molecules_from_json(json_path, subfolder)
                all_molecules.extend(molecules)
                break
        else:
            logger.warning(f"No recognized JSON file in {subfolder}")
    
    return all_molecules


# ============================================================================
# Multi-objective Ranking
# ============================================================================

def calculate_multi_objective_score(
    row: pd.Series,
    weights: dict[str, float] = None
) -> float:
    """
    Calculate multi-objective score
    
    Default weights:
    - Vina affinity: 0.4 (more negative = better)
    - SA score (inverted): 0.2 (lower SA = better)
    - QED: 0.15 (higher = better)
    - pKd accuracy (inverted error): 0.15 (lower error = better)
    - Residence time: 0.1 (higher = better)
    """
    if weights is None:
        weights = {
            'vina_affinity': 0.4,
            'sa_score': 0.2,
            'qed': 0.15,
            'pkd_accuracy': 0.15,
            'residence_time': 0.1
        }
    
    score = 0.0
    
    # Vina affinity (normalize to 0-1, more negative = better)
    if pd.notna(row.get('vina_affinity')):
        # Typical range: -12 to 0
        vina_norm = min(max(-row['vina_affinity'] / 12.0, 0), 1)
        score += weights['vina_affinity'] * vina_norm
    
    # SA score (invert: lower is better)
    if pd.notna(row.get('sa_score')):
        # Range: 1-10
        sa_norm = 1 - (row['sa_score'] - 1) / 9.0
        score += weights['sa_score'] * max(sa_norm, 0)
    
    # QED (already 0-1, higher is better)
    if pd.notna(row.get('qed')):
        score += weights['qed'] * row['qed']
    
    # pKd accuracy (lower error = better)
    if pd.notna(row.get('pkd_error')):
        # Typical error range: 0-5
        accuracy_norm = 1 - min(row['pkd_error'] / 5.0, 1)
        score += weights['pkd_accuracy'] * accuracy_norm
    
    # Residence time (higher = better, typical range 0-50)
    if pd.notna(row.get('residence_time')):
        rt_norm = min(row['residence_time'] / 50.0, 1)
        score += weights['residence_time'] * rt_norm
    
    return score


def rank_molecules(df: pd.DataFrame) -> pd.DataFrame:
    """Add ranking columns to dataframe"""
    
    # Calculate pKd error
    df['pkd_error'] = abs(df['predicted_affinity'] - df['actual_pkd'])
    
    # Primary rank by Vina affinity (ascending - more negative is better)
    df['rank_vina'] = df['vina_affinity'].rank(ascending=True, na_option='bottom')
    
    # Calculate multi-objective scores
    df['multi_objective_score'] = df.apply(calculate_multi_objective_score, axis=1)
    df['rank_multi'] = df['multi_objective_score'].rank(ascending=False, na_option='bottom')
    
    # Sort by multi-objective score
    df = df.sort_values('multi_objective_score', ascending=False)
    
    return df


# ============================================================================
# Analysis and Visualization
# ============================================================================

def generate_analysis_report(df: pd.DataFrame, output_dir: str) -> str:
    """Generate comprehensive analysis report"""
    
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE MOLECULAR DOCKING ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary statistics
    report.append("## SUMMARY STATISTICS")
    report.append("-" * 40)
    report.append(f"Total molecules processed: {len(df)}")
    report.append(f"Successful dockings: {df['docking_success'].sum()}")
    report.append(f"Success rate: {df['docking_success'].mean()*100:.1f}%")
    report.append("")
    
    # Per-source statistics
    report.append("## SOURCE BREAKDOWN")
    report.append("-" * 40)
    for source in df['source'].unique():
        source_df = df[df['source'] == source]
        report.append(f"\n{source}:")
        report.append(f"  Total: {len(source_df)}")
        report.append(f"  Docked: {source_df['docking_success'].sum()}")
        if source_df['docking_success'].sum() > 0:
            docked = source_df[source_df['docking_success']]
            report.append(f"  Vina affinity: {docked['vina_affinity'].mean():.2f} ± {docked['vina_affinity'].std():.2f} kcal/mol")
            report.append(f"  pKd (actual): {docked['actual_pkd'].mean():.2f} ± {docked['actual_pkd'].std():.2f}")
    
    report.append("")
    
    # Docking statistics
    docked_df = df[df['docking_success']]
    if len(docked_df) > 0:
        report.append("## DOCKING RESULTS")
        report.append("-" * 40)
        report.append(f"Vina affinity range: {docked_df['vina_affinity'].min():.2f} to {docked_df['vina_affinity'].max():.2f} kcal/mol")
        report.append(f"Mean Vina affinity: {docked_df['vina_affinity'].mean():.2f} ± {docked_df['vina_affinity'].std():.2f} kcal/mol")
        report.append(f"Actual pKd range: {docked_df['actual_pkd'].min():.2f} to {docked_df['actual_pkd'].max():.2f}")
        report.append("")
        
        report.append("## PREDICTION ACCURACY")
        report.append("-" * 40)
        report.append(f"Mean pKd error: {docked_df['pkd_error'].mean():.2f}")
        report.append(f"Median pKd error: {docked_df['pkd_error'].median():.2f}")
        
        # Correlation
        valid = docked_df.dropna(subset=['predicted_affinity', 'actual_pkd'])
        if len(valid) > 2:
            corr = valid['predicted_affinity'].corr(valid['actual_pkd'])
            report.append(f"Predicted vs Actual pKd correlation: {corr:.3f}")
        report.append("")
        
        report.append("## SYNTHETIC ACCESSIBILITY")
        report.append("-" * 40)
        report.append(f"SA score range: {docked_df['sa_score'].min():.2f} to {docked_df['sa_score'].max():.2f}")
        report.append(f"Mean SA score: {docked_df['sa_score'].mean():.2f}")
        report.append(f"Molecules with SA < 6: {(docked_df['sa_score'] < 6).sum()}")
        report.append("")
    
    # Top 20 molecules
    report.append("## TOP 20 MOLECULES (by multi-objective score)")
    report.append("-" * 40)
    top20 = docked_df.head(20)
    for i, (_, row) in enumerate(top20.iterrows(), 1):
        report.append(f"\n#{i}: {row['mol_id']}")
        report.append(f"   Source: {row['source']}")
        report.append(f"   SMILES: {row['smiles'][:60]}...")
        report.append(f"   Vina: {row['vina_affinity']:.2f} kcal/mol")
        report.append(f"   pKd (actual): {row['actual_pkd']:.2f}")
        report.append(f"   pKd (predicted): {row['predicted_affinity']:.2f}")
        report.append(f"   SA: {row['sa_score']:.2f}, QED: {row['qed']:.3f}")
        report.append(f"   Multi-obj score: {row['multi_objective_score']:.3f}")
    
    report_text = '\n'.join(report)
    
    # Save report
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    return report_text


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create analysis plots"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    docked_df = df[df['docking_success']].copy()
    
    if len(docked_df) == 0:
        logger.warning("No successful dockings to visualize")
        return
    
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Vina affinity distribution
    axes[0, 0].hist(docked_df['vina_affinity'].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Vina Binding Affinity (kcal/mol)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Vina Binding Affinities')
    axes[0, 0].axvline(docked_df['vina_affinity'].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 0].legend()
    
    # SA score distribution
    axes[0, 1].hist(docked_df['sa_score'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_xlabel('SA Score')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Synthetic Accessibility Scores')
    axes[0, 1].axvline(6, color='red', linestyle='--', label='SA=6 threshold')
    axes[0, 1].legend()
    
    # pKd error distribution
    axes[1, 0].hist(docked_df['pkd_error'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('pKd Error (|predicted - actual|)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Distribution of pKd Prediction Errors')
    
    # Predicted vs Actual pKd scatter
    valid = docked_df.dropna(subset=['predicted_affinity', 'actual_pkd'])
    axes[1, 1].scatter(valid['predicted_affinity'], valid['actual_pkd'], alpha=0.5)
    min_val = min(valid['predicted_affinity'].min(), valid['actual_pkd'].min())
    max_val = max(valid['predicted_affinity'].max(), valid['actual_pkd'].max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    axes[1, 1].set_xlabel('Predicted pKd')
    axes[1, 1].set_ylabel('Actual pKd (from Vina)')
    axes[1, 1].set_title('Predicted vs Actual pKd')
    if len(valid) > 2:
        corr = valid['predicted_affinity'].corr(valid['actual_pkd'])
        axes[1, 1].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1, 1].transAxes, va='top')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'distribution_plots.png'), dpi=150)
    plt.close()
    
    # 2. Source comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    sources = docked_df.groupby('source')['vina_affinity'].agg(['mean', 'std']).reset_index()
    sources = sources.sort_values('mean')
    x = range(len(sources))
    ax.barh(x, sources['mean'], xerr=sources['std'], capsize=5, alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels(sources['source'])
    ax.set_xlabel('Mean Vina Binding Affinity (kcal/mol)')
    ax.set_title('Comparison of Binding Affinities by Source')
    ax.axvline(docked_df['vina_affinity'].mean(), color='red', linestyle='--', label='Overall mean')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'source_comparison.png'), dpi=150)
    plt.close()
    
    # 3. Multi-objective score vs Vina
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        docked_df['vina_affinity'],
        docked_df['multi_objective_score'],
        c=docked_df['sa_score'],
        cmap='RdYlGn_r',
        alpha=0.7
    )
    plt.colorbar(scatter, label='SA Score')
    ax.set_xlabel('Vina Binding Affinity (kcal/mol)')
    ax.set_ylabel('Multi-objective Score')
    ax.set_title('Multi-objective Score vs Binding Affinity\n(colored by SA score)')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'multi_objective_vs_vina.png'), dpi=150)
    plt.close()
    
    logger.info(f"Visualizations saved to {fig_dir}")


# ============================================================================
# Main Pipeline
# ============================================================================

def run_docking_pipeline(
    generation_dir: str,
    receptor_pdb: str,
    output_dir: str,
    config: DockingConfig = None,
    n_workers: int = 4
) -> pd.DataFrame:
    """
    Main docking pipeline
    """
    if config is None:
        config = DockingConfig()
    config.receptor_pdb = receptor_pdb
    config.n_workers = n_workers
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE MOLECULAR DOCKING PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Load all molecules
    logger.info("\n[STEP 1] Loading molecules from all sources...")
    all_molecules = scan_and_load_all_molecules(generation_dir)
    logger.info(f"Total molecules loaded: {len(all_molecules)}")
    
    if len(all_molecules) == 0:
        logger.error("No molecules found!")
        return pd.DataFrame()
    
    # Step 2: Prepare receptor
    logger.info("\n[STEP 2] Preparing receptor...")
    receptor_pdbqt = prepare_receptor(receptor_pdb, output_dir)
    
    # Step 3: Dock all molecules
    logger.info(f"\n[STEP 3] Docking {len(all_molecules)} molecules...")
    logger.info(f"Using {n_workers} workers, exhaustiveness={config.exhaustiveness}")
    
    results = []
    
    # Process molecules (can parallelize but keeping sequential for stability)
    for i, mol_data in enumerate(all_molecules):
        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"Processing molecule {i+1}/{len(all_molecules)}")
        
        smiles = mol_data.get('smiles')
        if not smiles:
            continue
        
        result = dock_molecule(
            smiles=smiles,
            mol_id=mol_data['mol_id'],
            receptor_pdbqt=receptor_pdbqt,
            config=config,
            output_dir=output_dir
        )
        
        # Merge original data with docking results
        merged = {**mol_data, **result}
        results.append(merged)
    
    # Step 4: Create dataframe and rank
    logger.info("\n[STEP 4] Creating rankings...")
    df = pd.DataFrame(results)
    df = rank_molecules(df)
    
    # Step 5: Save results
    logger.info("\n[STEP 5] Saving results...")
    
    # Full results CSV
    csv_path = os.path.join(output_dir, 'docking_results_comprehensive.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved comprehensive results to {csv_path}")
    
    # Top 20 summary
    top20_path = os.path.join(output_dir, 'top_20_molecules.csv')
    df.head(20).to_csv(top20_path, index=False)
    
    # JSON export of top molecules
    top_json_path = os.path.join(output_dir, 'top_molecules.json')
    top_mols = df.head(50).to_dict(orient='records')
    with open(top_json_path, 'w') as f:
        json.dump(top_mols, f, indent=2, default=str)
    
    # Step 6: Generate analysis
    logger.info("\n[STEP 6] Generating analysis report...")
    report = generate_analysis_report(df, output_dir)
    print("\n" + report)
    
    # Step 7: Create visualizations
    logger.info("\n[STEP 7] Creating visualizations...")
    create_visualizations(df, output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    
    return df


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Molecular Docking Pipeline')
    _repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument('--generation-dir', type=str,
                       default=str(_repo_root / 'src' / 'results' / 'generated'),
                       help='Directory containing generated molecules')
    parser.add_argument('--receptor', type=str,
                       default=str(_repo_root / 'data' / 'structures' / 'receptor_siteA.pdb'),
                       help='Receptor PDB file')
    parser.add_argument('--output-dir', type=str,
                       default=str(_repo_root / 'src' / 'results' / 'docking'),
                       help='Output directory for results')
    parser.add_argument('--exhaustiveness', type=int, default=8,
                       help='Vina exhaustiveness parameter')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--center-x', type=float, default=24.87)
    parser.add_argument('--center-y', type=float, default=-12.54)
    parser.add_argument('--center-z', type=float, default=38.4)
    parser.add_argument('--box-size', type=float, default=22.0,
                       help='Docking box size (cubic)')
    
    args = parser.parse_args()
    
    config = DockingConfig(
        center_x=args.center_x,
        center_y=args.center_y,
        center_z=args.center_z,
        size_x=args.box_size,
        size_y=args.box_size,
        size_z=args.box_size,
        exhaustiveness=args.exhaustiveness,
    )
    
    df = run_docking_pipeline(
        generation_dir=args.generation_dir,
        receptor_pdb=args.receptor,
        output_dir=args.output_dir,
        config=config,
        n_workers=args.workers
    )
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Total molecules: {len(df)}")
    print(f"Successfully docked: {df['docking_success'].sum()}")
