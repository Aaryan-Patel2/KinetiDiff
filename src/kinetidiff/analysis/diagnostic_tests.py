#!/usr/bin/env python3
"""
Systematic diagnostic tests for Vina gradient guidance.
Run ALL tests to identify root cause of gradient direction issues.
"""
import argparse
import subprocess
import tempfile
from importlib.resources import files

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from kinetidiff.gcdm.equivariant_diffusion.vina_gradient_guidance import VinaGradientGuidance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Systematic diagnostic tests for Vina gradient guidance."
    )
    parser.add_argument(
        "--receptor-pdbqt",
        required=True,
        help="Path to prepared receptor .pdbqt file.",
    )
    args = parser.parse_args()

    receptor_pdbqt = args.receptor_pdbqt
    binding_site = {
        "center_x": 24.87, "center_y": -12.54, "center_z": 38.40,
        "size_x": 50.0, "size_y": 50.0, "size_z": 50.0,
    }

    print("=" * 60)
    print("VINA GRADIENT GUIDANCE DIAGNOSTIC TESTS")
    print("=" * 60)

    # ============================================================
    # TEST 1: Verify Vina scoring works
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 1: Verify Vina Scoring")
    print("=" * 60)

    mol = Chem.MolFromSmiles("C")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)

    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (
            pos.x + binding_site["center_x"],
            pos.y + binding_site["center_y"],
            pos.z + binding_site["center_z"],
        ))

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_file = f"{tmpdir}/test.pdb"
        pdbqt_file = f"{tmpdir}/test.pdbqt"

        Chem.MolToPDBFile(mol, pdb_file)
        subprocess.run(["obabel", pdb_file, "-O", pdbqt_file, "-h"],
                       capture_output=True, check=True)

        result = subprocess.run([
            "vina",
            "--receptor", receptor_pdbqt,
            "--ligand", pdbqt_file,
            "--score_only",
            "--center_x", str(binding_site["center_x"]),
            "--center_y", str(binding_site["center_y"]),
            "--center_z", str(binding_site["center_z"]),
            "--size_x", str(binding_site["size_x"]),
            "--size_y", str(binding_site["size_y"]),
            "--size_z", str(binding_site["size_z"]),
        ], capture_output=True, text=True)

        score = None
        for line in result.stdout.split('\n'):
            if 'Estimated Free Energy' in line:
                score = float(line.split(':')[1].split()[0])
                break

        print(f"Methane at pocket center: {score} kcal/mol")
        if score is not None:
            print("TEST 1 PASSED: Vina scoring works")
        else:
            print("TEST 1 FAILED: Could not get Vina score")

    # ============================================================
    # TEST 2: Verify gradient direction with SIMPLE function
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2: Verify Gradient Direction (Mock Function)")
    print("=" * 60)

    def mock_vina(coords):
        """Mock Vina: score = distance from origin. Lower distance = better."""
        distance = coords.norm()
        return distance.item()

    coords = torch.tensor([[10.0, 0.0, 0.0]])
    score_initial = mock_vina(coords)
    print(f"Initial position: {coords[0].tolist()}")
    print(f"Initial score: {score_initial:.2f}")

    eps = 0.1
    gradient = torch.zeros_like(coords)

    for dim in range(3):
        coords_plus = coords.clone()
        coords_plus[0, dim] += eps
        score_plus = mock_vina(coords_plus)
        gradient[0, dim] = (score_plus - score_initial) / eps

    print(f"Gradient: {gradient[0].tolist()}")
    print("  (Should point AWAY from origin since score increases with distance)")

    coords_after = coords - 1.0 * gradient
    score_after = mock_vina(coords_after)

    print(f"After gradient descent: {coords_after[0].tolist()}")
    print(f"Score after: {score_after:.2f}")
    print(f"Change: {score_after - score_initial:+.2f}")

    if score_after < score_initial:
        print("TEST 2 PASSED: Gradient descent minimizes score")
    else:
        print("TEST 2 FAILED: Gradient descent increased score!")

    # ============================================================
    # TEST 3: Test actual _compute_vina_score with RDKit mol
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 3: Test VinaGradientGuidance._compute_vina_score")
    print("=" * 60)

    guidance = VinaGradientGuidance(
        receptor_pdbqt=receptor_pdbqt,
        binding_site_config=binding_site,
        n_gradient_atoms=3,
    )

    mol = Chem.MolFromSmiles("CCCCCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)

    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (
            pos.x + binding_site["center_x"],
            pos.y + binding_site["center_y"],
            pos.z + binding_site["center_z"],
        ))

    score = guidance._compute_vina_score(mol, debug=False)
    print(f"Hexane at pocket center: {score} kcal/mol")

    if score is not None:
        print("TEST 3 PASSED: _compute_vina_score works")
    else:
        print("TEST 3 FAILED: Could not compute Vina score")

    # ============================================================
    # TEST 4: Test gradient by perturbing molecule coordinates
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 4: Test Gradient Direction with Real Vina")
    print("=" * 60)

    if score is not None:
        conf = mol.GetConformer()
        original_coords = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            original_coords.append([pos.x, pos.y, pos.z])
        original_coords = np.array(original_coords)

        score_center = score

        eps = 0.5
        conf.SetAtomPosition(0, (
            original_coords[0, 0] + eps,
            original_coords[0, 1],
            original_coords[0, 2],
        ))
        score_plus_x = guidance._compute_vina_score(mol, debug=False)

        conf.SetAtomPosition(0, (
            original_coords[0, 0] - eps,
            original_coords[0, 1],
            original_coords[0, 2],
        ))
        score_minus_x = guidance._compute_vina_score(mol, debug=False)

        conf.SetAtomPosition(0, tuple(original_coords[0]))

        print(f"Score at x:     {score_center:.3f}")
        print(f"Score at x+{eps}: {score_plus_x:.3f}")
        print(f"Score at x-{eps}: {score_minus_x:.3f}")

        gradient_x = (score_plus_x - score_minus_x) / (2 * eps)
        print(f"Numerical gradient (central diff): {gradient_x:.4f}")

        print("\nInterpretation:")
        print(f"  If we move in -gradient direction: x -= {gradient_x:.4f}")

        predicted_direction = "decreasing x" if gradient_x > 0 else "increasing x"
        better_score = score_minus_x if gradient_x > 0 else score_plus_x

        print(f"  Gradient descent would move toward {predicted_direction}")
        print(f"  Expected score after descent: {better_score:.3f}")

        if better_score < score_center:
            print("TEST 4 PASSED: Gradient descent would improve score")
        elif abs(better_score - score_center) < 0.01:
            print("TEST 4 WARNING: Score barely changes with perturbation")
        else:
            print("TEST 4 FAILED: Gradient descent would worsen score!")

    # ============================================================
    # TEST 5: Test compute_gradient function
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 5: Test compute_gradient Function")
    print("=" * 60)

    n_atoms = 6
    coords_tensor = torch.zeros(n_atoms, 3)
    for i in range(n_atoms):
        coords_tensor[i, 0] = binding_site["center_x"] + i * 1.5
        coords_tensor[i, 1] = binding_site["center_y"]
        coords_tensor[i, 2] = binding_site["center_z"]

    atom_types = torch.zeros(n_atoms, dtype=torch.long)
    lig_mask = torch.zeros(n_atoms, dtype=torch.long)
    pocket_center = torch.tensor([
        binding_site["center_x"],
        binding_site["center_y"],
        binding_site["center_z"],
    ])

    print(f"Testing compute_gradient with {n_atoms} atoms...")
    print(f"Coords center: {coords_tensor.mean(dim=0).tolist()}")

    gradient, metrics = guidance.compute_gradient(
        coords_tensor, atom_types, lig_mask, timestep=10, pocket_center=pocket_center
    )

    print("\nResults:")
    print(f"  valid_molecule: {metrics['valid_molecule']}")
    print(f"  vina_score: {metrics['vina_score']}")
    print(f"  gradient_norm: {metrics['gradient_norm']:.4f}")

    if metrics['valid_molecule'] and metrics['vina_score'] is not None:
        print(f"  gradient shape: {gradient.shape}")
        print(f"  gradient mean: {gradient.mean().item():.4f}")
        print("TEST 5 PASSED: compute_gradient works")
    else:
        print("TEST 5 FAILED: Could not compute gradient")

    # ============================================================
    # TEST 6: Test single gradient descent step
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 6: Single Gradient Descent Step")
    print("=" * 60)

    if metrics['valid_molecule'] and metrics['vina_score'] is not None:
        score_before = metrics['vina_score']

        lr = 0.3
        coords_after = coords_tensor - lr * gradient

        gradient_after, metrics_after = guidance.compute_gradient(
            coords_after, atom_types, lig_mask, timestep=10, pocket_center=pocket_center
        )

        score_after = metrics_after['vina_score']

        print(f"Score before: {score_before:.3f}")
        print(f"Score after:  {score_after:.3f}")
        print(f"Change: {score_after - score_before:+.3f}")

        if score_after is not None:
            if score_after < score_before:
                print("TEST 6 PASSED: Gradient descent improved score!")
            elif abs(score_after - score_before) < 0.5:
                print("TEST 6 WARNING: Score barely changed")
            else:
                print("TEST 6 FAILED: Gradient descent made score WORSE!")
                print("   >>> GRADIENT DIRECTION IS WRONG <<<")
        else:
            print("TEST 6 FAILED: Could not score after descent")

    # ============================================================
    # TEST 7: Multiple gradient descent iterations
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 7: Multiple Gradient Descent Iterations")
    print("=" * 60)

    coords_iter = coords_tensor.clone()
    scores = []

    for iteration in range(5):
        gradient, metrics = guidance.compute_gradient(
            coords_iter, atom_types, lig_mask, timestep=10, pocket_center=pocket_center
        )

        if not metrics['valid_molecule'] or metrics['vina_score'] is None:
            print(f"Iter {iteration}: Invalid molecule")
            break

        scores.append(metrics['vina_score'])

        if iteration > 0:
            change = scores[-1] - scores[-2]
            symbol = "OK" if change < 0 else "BAD"
            print(f"Iter {iteration}: score={scores[-1]:.3f} (change={change:+.3f}) {symbol}")
        else:
            print(f"Iter {iteration}: score={scores[-1]:.3f} (initial)")

        lr = 0.3
        coords_iter = coords_iter - lr * gradient

    if len(scores) > 1:
        improvements = sum(1 for i in range(1, len(scores)) if scores[i] < scores[i-1])
        total_change = scores[-1] - scores[0]

        print("\nSummary:")
        print(f"  Iterations with improvement: {improvements}/{len(scores)-1}")
        print(f"  Total change: {total_change:+.3f}")

        if improvements >= len(scores) - 2:
            print("TEST 7 PASSED: Consistent improvement")
        else:
            print("TEST 7 FAILED: Inconsistent gradient direction")

    # ============================================================
    # TEST 8: Check code for sign issues
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 8: Code Inspection for Sign Issues")
    print("=" * 60)

    guidance_file = files("kinetidiff.gcdm.equivariant_diffusion") / "vina_gradient_guidance.py"
    with open(guidance_file) as f:
        code = f.read()

    print("Searching for gradient negations...")
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if 'valid_gradient' in line and '=' in line:
            if '-' in line.split('=')[1]:
                print(f"  Line {i+1}: {line.strip()}")

    model_file = files("kinetidiff.gcdm.equivariant_diffusion") / "conditional_model.py"
    with open(model_file) as f:
        model_code = f.read()

    print("\nSearching for gradient application...")
    lines = model_code.split('\n')
    for i, line in enumerate(lines):
        if 'gradient' in line.lower() and ('coords' in line or 'z_lig' in line):
            if '+' in line or '-' in line:
                print(f"  Line {i+1}: {line.strip()}")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print("""
Key findings:
- If TEST 6 failed: Gradient direction is WRONG
  Fix: Check for erroneous negations in compute_gradient()

- If TEST 7 failed: Gradients are inconsistent
  Fix: Check numerical stability, try larger epsilon

- If both passed but GCDM still fails:
  Fix: Check gradient APPLICATION in conditional_model.py
""")


if __name__ == "__main__":
    main()
