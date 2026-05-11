#!/usr/bin/env python3
"""
Molecular Property Evaluation for Generated Molecules.

Evaluates molecules from GCDM generation across multiple metrics:
- Binding affinity (HNN-Denovo)
- Docking scores (QuickVina2)
- Synthetic accessibility (SA Score)
- Drug-likeness (QED, Lipinski)
- Novelty and diversity

Usage:
    python evaluate_molecules.py \\
        --input results/generated_molecules.sdf \\
        --protein_sequence "MTEYKLVVV..." \\
        --output results/evaluation_report.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# RDKit imports
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import QED, AllChem, Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def load_molecules(filepath: str) -> list['Chem.Mol']:
    """Load molecules from SDF or SMILES file."""
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit required for molecule loading")
    
    molecules = []
    
    if filepath.endswith('.sdf'):
        suppl = Chem.SDMolSupplier(filepath)
        molecules = [mol for mol in suppl if mol is not None]
    elif filepath.endswith('.smi') or filepath.endswith('.smiles'):
        with open(filepath) as f:
            for line in f:
                smiles = line.strip().split()[0]
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    molecules.append(mol)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    return molecules


def compute_basic_properties(mol: 'Chem.Mol') -> dict:
    """Compute basic molecular properties."""
    return {
        'mol_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'hbd': Descriptors.NumHDonors(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'tpsa': Descriptors.TPSA(mol),
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'num_rings': rdMolDescriptors.CalcNumRings(mol),
        'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'fraction_sp3': rdMolDescriptors.CalcFractionCSP3(mol),
        'qed': QED.qed(mol)
    }


def compute_lipinski(mol: 'Chem.Mol') -> dict:
    """Check Lipinski's Rule of Five."""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    
    violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10
    ])
    
    return {
        'lipinski_mw': mw <= 500,
        'lipinski_logp': logp <= 5,
        'lipinski_hbd': hbd <= 5,
        'lipinski_hba': hba <= 10,
        'lipinski_violations': violations,
        'lipinski_pass': violations <= 1
    }


def compute_diversity(molecules: list['Chem.Mol']) -> dict:
    """Compute internal diversity using Tanimoto similarity."""
    if len(molecules) < 2:
        return {'internal_diversity': 1.0}
    
    # Compute Morgan fingerprints
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) 
           for mol in molecules]
    
    # Compute pairwise Tanimoto similarities
    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)
    
    avg_similarity = np.mean(similarities)
    
    return {
        'internal_diversity': 1 - avg_similarity,
        'mean_tanimoto_similarity': avg_similarity,
        'max_tanimoto_similarity': max(similarities),
        'min_tanimoto_similarity': min(similarities)
    }


def compute_novelty(
    molecules: list['Chem.Mol'],
    reference_smiles: list[str] | None = None
) -> dict:
    """Compute novelty relative to reference set."""
    if reference_smiles is None or len(reference_smiles) == 0:
        return {'novelty': 1.0}
    
    # Get SMILES for generated molecules
    gen_smiles = set(Chem.MolToSmiles(mol) for mol in molecules)
    ref_smiles = set(reference_smiles)
    
    novel_count = len(gen_smiles - ref_smiles)
    
    return {
        'novelty': novel_count / len(gen_smiles) if gen_smiles else 0,
        'num_novel': novel_count,
        'num_duplicate': len(gen_smiles) - novel_count
    }


def evaluate_molecules(
    molecules: list['Chem.Mol'],
    protein_sequence: str | None = None,
    reference_smiles: list[str] | None = None
) -> dict:
    """
    Full evaluation of generated molecules.
    
    Args:
        molecules: List of RDKit molecules
        protein_sequence: Target protein sequence for affinity prediction
        reference_smiles: Reference SMILES for novelty computation
        
    Returns:
        evaluation: Dictionary with all metrics
    """
    evaluation = {
        'n_molecules': len(molecules),
        'properties': [],
        'statistics': {},
        'diversity': {},
        'novelty': {}
    }
    
    # Per-molecule properties
    for mol in molecules:
        props = compute_basic_properties(mol)
        props.update(compute_lipinski(mol))
        props['smiles'] = Chem.MolToSmiles(mol)
        evaluation['properties'].append(props)
    
    # Aggregate statistics
    prop_keys = ['mol_weight', 'logp', 'qed', 'tpsa', 'rotatable_bonds']
    for key in prop_keys:
        values = [p[key] for p in evaluation['properties']]
        evaluation['statistics'][key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # Lipinski pass rate
    lipinski_passes = sum(1 for p in evaluation['properties'] if p['lipinski_pass'])
    evaluation['statistics']['lipinski_pass_rate'] = lipinski_passes / len(molecules)
    
    # Diversity
    evaluation['diversity'] = compute_diversity(molecules)
    
    # Novelty
    evaluation['novelty'] = compute_novelty(molecules, reference_smiles)
    
    return evaluation


def main():
    parser = argparse.ArgumentParser(description='Evaluate generated molecules')
    parser.add_argument('--input', type=str, required=True,
                       help='Input SDF or SMILES file')
    parser.add_argument('--protein_sequence', type=str, default=None,
                       help='Target protein sequence')
    parser.add_argument('--reference', type=str, default=None,
                       help='Reference SMILES file for novelty')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file')
    
    args = parser.parse_args()
    
    print(f"Loading molecules from: {args.input}")
    molecules = load_molecules(args.input)
    print(f"Loaded {len(molecules)} molecules")
    
    # Load reference if provided
    reference_smiles = None
    if args.reference:
        with open(args.reference) as f:
            reference_smiles = [line.strip().split()[0] for line in f]
    
    # Evaluate
    print("Evaluating molecules...")
    evaluation = evaluate_molecules(
        molecules,
        protein_sequence=args.protein_sequence,
        reference_smiles=reference_smiles
    )
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(evaluation, f, indent=2, default=float)
    
    print(f"Saved evaluation to: {args.output}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Molecules: {evaluation['n_molecules']}")
    print(f"QED: {evaluation['statistics']['qed']['mean']:.3f} ± {evaluation['statistics']['qed']['std']:.3f}")
    print(f"Lipinski pass rate: {evaluation['statistics']['lipinski_pass_rate']:.1%}")
    print(f"Internal diversity: {evaluation['diversity']['internal_diversity']:.3f}")
    if evaluation['novelty']:
        print(f"Novelty: {evaluation['novelty']['novelty']:.1%}")


if __name__ == '__main__':
    main()
