#!/usr/bin/env python3
"""GPU Vina guidance test"""
import json
import os
import sys
import time
from datetime import datetime

# Add current dir to path since guidance module is here
sys.path.insert(0, ".")

print("=" * 70)
print("VINA GUIDANCE GPU TEST")
print("=" * 70)
print(f"Date: {datetime.now().isoformat()}")

# Check GPU
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Import Vina guidance
print("\nImporting Vina guidance...")
from vina_guidance import VinaGuidance

print("OK")

# Check receptor
receptor = "receptor.pdbqt"
if not os.path.exists(receptor):
    print(f"ERROR: Receptor not found: {receptor}")
    sys.exit(1)
print(f"Receptor: {receptor}")

# Initialize
print("\nInitializing Vina guidance...")
vina = VinaGuidance(
    receptor_pdbqt=receptor,
    binding_site_config={
        "center_x": 24.87, "center_y": -12.54, "center_z": 38.40,
        "size_x": 20.0, "size_y": 20.0, "size_z": 20.0
    },
    score_only=False,
    timeout=60,
    device="cuda"
)
print("Initialized")

# Test molecules
from rdkit import Chem
from rdkit.Chem import AllChem

molecules = [
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(O)=O"),
    ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("Paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("Benzene", "c1ccccc1"),
    ("LDN-193189", "CC1=NC2=CC=CC=C2N1C3=NC4=C(C5=CC=C(C=C5)C#N)C=CN=C4N3C6CCCCC6"),
    ("Dorsomorphin", "CC1=CC2=C(C=C1)N(C=N2)C3=CC=C(C=C3)C4=NC(=NC(=N4)N)C5=CC=CC=C5"),
]

print("\n" + "=" * 70)
print("SCORING MOLECULES")
print("=" * 70)

results = []
for name, smiles in molecules:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"{name}: Invalid SMILES")
            continue
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Position at binding site
        conf = mol.GetConformer()
        centroid = [0, 0, 0]
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            centroid[0] += pos.x
            centroid[1] += pos.y
            centroid[2] += pos.z
        centroid = [c / mol.GetNumAtoms() for c in centroid]
        
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (
                pos.x - centroid[0] + 24.87,
                pos.y - centroid[1] - 12.54,
                pos.z - centroid[2] + 38.40
            ))
        
        coords = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
        atom_types = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
        
        start = time.time()
        score = vina.compute_vina_score(coords, atom_types, use_gcdm_encoding=False)
        elapsed = time.time() - start
        
        pkd = -score / 1.36
        result = {
            "name": name,
            "smiles": smiles,
            "atoms": mol.GetNumAtoms(),
            "vina_score": round(score, 3),
            "pkd_approx": round(pkd, 2),
            "time_ms": round(elapsed * 1000, 0)
        }
        results.append(result)
        
        print(f"{name:15} | Vina: {score:7.2f} kcal/mol | pKd: {pkd:5.1f} | {elapsed*1000:6.0f}ms")
        
    except Exception as e:
        print(f"{name}: Error - {e}")

# Test gradient computation
print("\n" + "=" * 70)
print("GRADIENT TEST")
print("=" * 70)

smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mol)

# Position at binding site
conf = mol.GetConformer()
centroid = [0, 0, 0]
for i in range(mol.GetNumAtoms()):
    pos = conf.GetAtomPosition(i)
    centroid[0] += pos.x
    centroid[1] += pos.y
    centroid[2] += pos.z
centroid = [c / mol.GetNumAtoms() for c in centroid]

for i in range(mol.GetNumAtoms()):
    pos = conf.GetAtomPosition(i)
    conf.SetAtomPosition(i, (
        pos.x - centroid[0] + 24.87,
        pos.y - centroid[1] - 12.54,
        pos.z - centroid[2] + 38.40
    ))

coords = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
atom_types = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)

print("\nComputing fast gradient (n_samples=5)...")
start = time.time()
gradient, score = vina.compute_vina_gradient_fast(coords, atom_types, n_samples=5, use_gcdm_encoding=False)
elapsed = time.time() - start

print(f"  Original Vina score: {score:.2f} kcal/mol")
print(f"  Gradient norm: {gradient.norm():.4f}")
print(f"  Time: {elapsed:.2f}s")

# Test gradient step
print("\nTesting gradient step improvement...")
step_size = 0.1
coords_new = coords + step_size * gradient
new_score = vina.compute_vina_score(coords_new, atom_types, use_gcdm_encoding=False)
print(f"  After step: {new_score:.2f} kcal/mol")
print(f"  Improvement: {new_score - score:.3f} kcal/mol")

if new_score < score:
    print("  PASS: Gradient improved binding")
else:
    print("  NOTE: Gradient did not improve (may need tuning)")

# Save results
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

output = {
    "date": datetime.now().isoformat(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "pytorch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "results": results,
    "gradient_test": {
        "original_score": score,
        "gradient_norm": gradient.norm().item(),
        "time_seconds": elapsed,
        "new_score": new_score,
        "improvement": new_score - score
    }
}

with open("gpu_test_results.json", "w") as f:
    json.dump(output, f, indent=2)
print("Saved to gpu_test_results.json")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
