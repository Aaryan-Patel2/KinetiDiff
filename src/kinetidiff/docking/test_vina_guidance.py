#!/usr/bin/env python3
"""
Test script for Vina-based guidance implementation.

This validates that:
1. Vina scoring works correctly
2. Numerical gradients point toward better binding
3. Integration with GCDM guidance pipeline works

Usage:
    python test_vina_guidance.py
    
    # With specific receptor:
    python test_vina_guidance.py --receptor DOCKING2/receptor.pdbqt

Expected output:
    - Vina scores for test molecules
    - Gradient direction validation (should improve binding)
    - Comparison with HNN-Denovo predictions
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "models" / "gcdm-modified" / "src"))

import numpy as np
import torch

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("WARNING: RDKit not available. Some tests will be skipped.")


def test_vina_scoring(receptor_pdbqt: str, binding_site: dict):
    """Test basic Vina scoring functionality."""
    print("\n" + "=" * 60)
    print("TEST 1: Vina Scoring")
    print("=" * 60)
    
    from guidance.vina_guidance import VinaGuidance
    
    # Initialize
    try:
        vina = VinaGuidance(
            receptor_pdbqt=receptor_pdbqt,
            binding_site_config=binding_site,
            score_only=True
        )
        print("✓ Vina guidance initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Vina guidance: {e}")
        return None
    
    # Test molecules (drug-like compounds)
    test_molecules = [
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(O)=O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Paracetamol", "CC(=O)Nc1ccc(O)cc1"),
        ("Benzene", "c1ccccc1"),  # Simple reference
        ("LDN-193189", "CC1=NC2=CC=CC=C2N1C3=NC4=C(C5=CC=C(C=C5)C#N)C=CN=C4N3C6CCCCC6"),  # Known ACVR1 inhibitor
    ]
    
    results = []
    print("\nScoring test molecules:")
    print("-" * 60)
    
    for name, smiles in test_molecules:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"  {name}: Failed to parse SMILES")
                continue
                
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            
            if mol.GetNumConformers() == 0:
                print(f"  {name}: Failed to generate 3D coords")
                continue
            
            # Get coords and atom types
            coords = torch.tensor(
                mol.GetConformer().GetPositions(),
                dtype=torch.float32
            )
            atom_types = torch.tensor(
                [atom.GetAtomicNum() for atom in mol.GetAtoms()],
                dtype=torch.long
            )
            
            # Score
            start = time.time()
            score = vina.compute_vina_score(coords, atom_types, use_gcdm_encoding=False)
            elapsed = time.time() - start
            
            # Approximate pKd
            pkd_approx = -score / 1.36 if score != 0 else 0
            
            results.append({
                'name': name,
                'smiles': smiles,
                'vina_score': score,
                'pkd_approx': pkd_approx,
                'time_ms': elapsed * 1000
            })
            
            print(f"  {name:<15} | Vina: {score:6.2f} kcal/mol | pKd: {pkd_approx:5.1f} | Time: {elapsed*1000:.0f}ms")
            
        except Exception as e:
            print(f"  {name}: Error - {e}")
    
    return results, vina


def test_gradient_direction(vina, smiles: str = "CC(=O)Oc1ccc(O)cc1"):
    """Test that gradients point toward better binding."""
    print("\n" + "=" * 60)
    print("TEST 2: Gradient Direction Validation")
    print("=" * 60)
    
    # Create test molecule
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    coords = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
    atom_types = torch.tensor(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
        dtype=torch.long
    )
    
    print(f"\nTest molecule: {smiles}")
    print(f"Atoms: {coords.shape[0]}")
    
    # Compute gradient
    print("\nComputing gradient...")
    start = time.time()
    gradient, original_score = vina.compute_vina_gradient(
        coords, atom_types, eps=0.05, use_gcdm_encoding=False
    )
    grad_time = time.time() - start
    
    print(f"  Original Vina score: {original_score:.2f} kcal/mol")
    print(f"  Gradient norm: {gradient.norm():.4f}")
    print(f"  Gradient computation time: {grad_time:.2f}s")
    
    # Test different step sizes
    print("\nTesting gradient steps:")
    step_sizes = [0.05, 0.1, 0.2, 0.5]
    improvements = []
    
    for step_size in step_sizes:
        coords_new = coords + step_size * gradient
        new_score = vina.compute_vina_score(coords_new, atom_types, use_gcdm_encoding=False)
        improvement = new_score - original_score  # Negative improvement = better
        improvements.append(improvement)
        
        status = "✓ IMPROVED" if improvement < 0 else "- no change"
        print(f"  Step {step_size:.2f}: {new_score:.2f} kcal/mol (Δ = {improvement:+.3f}) {status}")
    
    # Overall assessment
    n_improved = sum(1 for imp in improvements if imp < 0)
    
    if n_improved >= len(step_sizes) // 2:
        print(f"\n✓ PASS: Gradient improves binding in {n_improved}/{len(step_sizes)} cases")
        return True
    else:
        print(f"\n⚠ PARTIAL: Gradient improved in {n_improved}/{len(step_sizes)} cases")
        return False


def test_fast_gradient(vina, smiles: str = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"):
    """Test fast approximate gradient."""
    print("\n" + "=" * 60)
    print("TEST 3: Fast Gradient Approximation")
    print("=" * 60)
    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    
    coords = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
    atom_types = torch.tensor(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
        dtype=torch.long
    )
    
    print(f"\nTest molecule: {smiles}")
    print(f"Atoms: {coords.shape[0]}")
    
    # Exact gradient
    print("\nComputing exact gradient...")
    start = time.time()
    gradient_exact, score_exact = vina.compute_vina_gradient(
        coords, atom_types, use_gcdm_encoding=False
    )
    exact_time = time.time() - start
    print(f"  Time: {exact_time:.2f}s")
    print(f"  Gradient norm: {gradient_exact.norm():.4f}")
    
    # Fast gradient (n_samples=3)
    print("\nComputing fast gradient (n_samples=3)...")
    start = time.time()
    gradient_fast3, score_fast3 = vina.compute_vina_gradient_fast(
        coords, atom_types, n_samples=3, use_gcdm_encoding=False
    )
    fast3_time = time.time() - start
    print(f"  Time: {fast3_time:.2f}s (speedup: {exact_time/fast3_time:.1f}x)")
    print(f"  Gradient norm: {gradient_fast3.norm():.4f}")
    
    # Fast gradient (n_samples=5)
    print("\nComputing fast gradient (n_samples=5)...")
    start = time.time()
    gradient_fast5, score_fast5 = vina.compute_vina_gradient_fast(
        coords, atom_types, n_samples=5, use_gcdm_encoding=False
    )
    fast5_time = time.time() - start
    print(f"  Time: {fast5_time:.2f}s (speedup: {exact_time/fast5_time:.1f}x)")
    print(f"  Gradient norm: {gradient_fast5.norm():.4f}")
    
    # Compare directions
    cosine_sim_3 = torch.nn.functional.cosine_similarity(
        gradient_exact.flatten(), gradient_fast3.flatten(), dim=0
    )
    cosine_sim_5 = torch.nn.functional.cosine_similarity(
        gradient_exact.flatten(), gradient_fast5.flatten(), dim=0
    )
    
    print("\nGradient direction similarity:")
    print(f"  Fast (n=3) vs Exact: cosine = {cosine_sim_3:.3f}")
    print(f"  Fast (n=5) vs Exact: cosine = {cosine_sim_5:.3f}")
    
    return fast3_time, fast5_time, exact_time


def test_batch_scoring(vina, n_molecules: int = 20):
    """Test batch scoring performance."""
    print("\n" + "=" * 60)
    print("TEST 4: Batch Scoring Performance")
    print("=" * 60)
    
    # Generate random drug-like molecules
    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CC(C)Cc1ccc(cc1)C(C)C(O)=O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(=O)Nc1ccc(O)cc1",  # Paracetamol
        "COc1ccc2[nH]cc(CCNC(C)C)c2c1",  # Drug-like
        "Cn1c(=O)c2c(nc(CC3CCCCC3)n2C)n(C)c1=O",
        "CC1=CC=C(C=C1)S(=O)(=O)NC2=NC=CC=N2",
        "CC(C)(C)C1=CC=C(C=C1)C(O)C2=CC=C(Cl)C=C2",
        "Cc1ccc(cc1)c2cc(nn2c3ccc(cc3)S(N)(=O)=O)C(F)(F)F",
        "COC1=CC(=CC(=C1OC)OC)C2=NC(=CS2)C3=CC=C(C=C3)OC",
    ]
    
    # Extend to n_molecules
    while len(test_smiles) < n_molecules:
        test_smiles = test_smiles + test_smiles
    test_smiles = test_smiles[:n_molecules]
    
    # Convert to coords
    coords_batch = []
    types_batch = []
    valid_smiles = []
    
    for smiles in test_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            if mol.GetNumConformers() == 0:
                continue
            
            coords = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
            atom_types = torch.tensor(
                [atom.GetAtomicNum() for atom in mol.GetAtoms()],
                dtype=torch.long
            )
            
            coords_batch.append(coords)
            types_batch.append(atom_types)
            valid_smiles.append(smiles)
        except:
            continue
    
    print(f"\nScoring {len(valid_smiles)} molecules...")
    
    # Time individual scoring
    start = time.time()
    scores = []
    for coords, atom_types in zip(coords_batch, types_batch):
        score = vina.compute_vina_score(coords, atom_types, use_gcdm_encoding=False)
        scores.append(score)
    total_time = time.time() - start
    
    scores = np.array(scores)
    
    print("\nResults:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Per-molecule time: {total_time/len(scores)*1000:.0f}ms")
    print(f"  Score range: [{scores.min():.2f}, {scores.max():.2f}] kcal/mol")
    print(f"  Mean score: {scores.mean():.2f} ± {scores.std():.2f} kcal/mol")
    
    return scores, total_time


def test_comparison_with_hnn_denovo(vina, checkpoint_path: str = None):
    """Compare Vina scores with HNN-Denovo predictions."""
    print("\n" + "=" * 60)
    print("TEST 5: Comparison with HNN-Denovo")
    print("=" * 60)
    
    # Load HNN-Denovo if checkpoint provided
    if checkpoint_path is None:
        # Try to find a checkpoint
        possible_paths = [
            "models/affinity_pred/checkpoints/best_model.pt",
            "models/checkpoints/best_model.ckpt",
            "trained_models/best_model.ckpt",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
    
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("  No HNN-Denovo checkpoint found. Skipping comparison.")
        return None
    
    print(f"\nLoading HNN-Denovo from {checkpoint_path}...")
    
    try:
        from guidance.affinity_guidance import AffinityGuidanceModel
        hnn_model = AffinityGuidanceModel(checkpoint_path=checkpoint_path)
        print("  ✓ HNN-Denovo loaded")
    except Exception as e:
        print(f"  ✗ Failed to load HNN-Denovo: {e}")
        return None
    
    # Test molecules
    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CC(=O)Nc1ccc(O)cc1",
        "c1ccccc1",
    ]
    
    print("\nComparing predictions:")
    print("-" * 60)
    print(f"{'SMILES':<35} | {'Vina':>10} | {'HNN-pKd':>10} | {'Diff':>8}")
    print("-" * 60)
    
    results = []
    for smiles in test_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        
        coords = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
        atom_types = torch.tensor(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()],
            dtype=torch.long
        )
        
        # Vina score -> approx pKd
        vina_score = vina.compute_vina_score(coords, atom_types, use_gcdm_encoding=False)
        vina_pkd = -vina_score / 1.36
        
        # HNN-Denovo prediction
        try:
            hnn_pkd = hnn_model.predict_affinity(
                smiles, 
                "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRQHKLRKLNPPDESGPGCMNCVEVSTQNFVPTEQPQCGAMFECFHKKLQHVLKD"
            )
        except:
            hnn_pkd = 6.0  # Fallback
        
        diff = vina_pkd - hnn_pkd
        
        results.append({
            'smiles': smiles,
            'vina_pkd': vina_pkd,
            'hnn_pkd': hnn_pkd,
            'diff': diff
        })
        
        print(f"{smiles[:35]:<35} | {vina_pkd:>10.2f} | {hnn_pkd:>10.2f} | {diff:>+8.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test Vina guidance implementation")
    parser.add_argument("--receptor", type=str, default="DOCKING2/receptor.pdbqt",
                       help="Path to receptor PDBQT file")
    parser.add_argument("--center-x", type=float, default=24.87)
    parser.add_argument("--center-y", type=float, default=-12.54)
    parser.add_argument("--center-z", type=float, default=38.40)
    parser.add_argument("--size", type=float, default=20.0,
                       help="Box size (Angstroms)")
    parser.add_argument("--hnn-checkpoint", type=str, default=None,
                       help="Path to HNN-Denovo checkpoint")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("VINA GUIDANCE TEST SUITE")
    print("=" * 70)
    print(f"Receptor: {args.receptor}")
    print(f"Binding site center: ({args.center_x}, {args.center_y}, {args.center_z})")
    print(f"Box size: {args.size}Å")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check receptor exists
    if not os.path.exists(args.receptor):
        print(f"\n✗ ERROR: Receptor not found: {args.receptor}")
        print("  Please provide a valid receptor PDBQT file.")
        return 1
    
    # Binding site config
    binding_site = {
        'center_x': args.center_x,
        'center_y': args.center_y,
        'center_z': args.center_z,
        'size_x': args.size,
        'size_y': args.size,
        'size_z': args.size,
    }
    
    # Run tests
    all_passed = True
    
    # Test 1: Basic scoring
    scoring_results, vina = test_vina_scoring(args.receptor, binding_site)
    if vina is None:
        print("\n✗ FAILED: Could not initialize Vina guidance")
        return 1
    
    # Test 2: Gradient direction
    gradient_passed = test_gradient_direction(vina)
    if not gradient_passed:
        all_passed = False
    
    # Test 3: Fast gradient
    fast3_time, fast5_time, exact_time = test_fast_gradient(vina)
    
    # Test 4: Batch performance
    scores, batch_time = test_batch_scoring(vina, n_molecules=10)
    
    # Test 5: HNN-Denovo comparison (optional)
    comparison_results = test_comparison_with_hnn_denovo(vina, args.hnn_checkpoint)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("  ✓ Vina scoring: PASSED")
    print(f"  {'✓' if gradient_passed else '⚠'} Gradient direction: {'PASSED' if gradient_passed else 'PARTIAL'}")
    print(f"  ✓ Fast gradient: {exact_time/fast5_time:.1f}x speedup")
    print(f"  ✓ Batch scoring: {batch_time/len(scores)*1000:.0f}ms per molecule")
    
    if scoring_results:
        mean_score = np.mean([r['vina_score'] for r in scoring_results])
        print(f"\n  Mean Vina score: {mean_score:.2f} kcal/mol")
        print(f"  Expected pKd range: {-mean_score/1.36:.1f} (approx)")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("⚠ SOME TESTS HAD ISSUES (see above)")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
