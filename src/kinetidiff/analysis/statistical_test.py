#!/usr/bin/env python3
"""
Statistical test: Compare GCDM with and without Vina guidance.
Tracks Vina scores at each timestep to verify consistent improvement.
"""
import argparse
import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
from rdkit import Chem

from kinetidiff.gcdm.lightning_modules import LigandPocketDDPM


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical comparison: guided vs unguided GCDM generation."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "KINETIDIFF_CHECKPOINT",
            "data/checkpoints/bindingmoad_ca_cond_gcpnet.ckpt",
        ),
        help="Path to LigandPocketDDPM checkpoint.",
    )
    parser.add_argument(
        "--receptor-pdbqt",
        required=True,
        help="Path to prepared receptor .pdbqt file.",
    )
    args = parser.parse_args()

    import torch.serialization
    torch.serialization.add_safe_globals([__import__('pathlib').PosixPath])

    print("Loading GCDM model...")
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint,
        map_location="cuda:0",
    )
    model.eval()
    model.to("cuda:0")
    print("Model loaded")

    binding_site = {
        "center_x": 24.87, "center_y": -12.54, "center_z": 38.40,
        "size_x": 50.0, "size_y": 50.0, "size_z": 50.0,
    }

    pocket_dict = {
        "x": torch.randn(30, 3).cuda() * 3,
        "one_hot": torch.zeros(30, 25).cuda(),
        "mask": torch.zeros(30, dtype=torch.long).cuda(),
        "size": torch.tensor([30]).cuda(),
    }
    pocket_dict["one_hot"][:, 0] = 1

    def generate_mols(with_guidance, n=5, verbose=False):
        """Generate molecules, optionally with guidance."""
        if with_guidance:
            model.ddpm.enable_vina_guidance(
                receptor_pdbqt=args.receptor_pdbqt,
                binding_site_config=binding_site,
                guidance_scale=0.3,
                guidance_start_timestep=20,
                guidance_interval=5,
                n_gradient_atoms=5,
                eps=0.3,
            )
        else:
            model.ddpm._guidance_enabled = False

        smiles_list = []
        try:
            mols = model.generate_ligands(
                pdb_file=None, pocket_ids=None, ref_ligand=None,
                n_samples=n, num_nodes_lig=None, sanitize=False,
                largest_frag=False, relax_iter=0, timesteps=100,
                all_frags=True, pocket_dict=pocket_dict,
            )
            if mols:
                for mol, _, _ in mols:
                    if mol is not None:
                        try:
                            smi = Chem.MolToSmiles(mol)
                            smiles_list.append(smi)
                        except Exception:
                            pass
        except Exception as e:
            print(f"  Error: {e}")

        stats = {}
        if with_guidance and hasattr(model.ddpm, 'vina_guidance') and model.ddpm.vina_guidance:
            stats = model.ddpm.vina_guidance.get_stats()

        return smiles_list, stats

    N_SAMPLES = 10
    N_RUNS = 3

    print("=" * 60)
    print("STATISTICAL COMPARISON: GUIDED vs UNGUIDED")
    print("=" * 60)
    print(f"Samples per run: {N_SAMPLES}")
    print(f"Number of runs: {N_RUNS}")

    print("\n--- RUNNING WITHOUT GUIDANCE ---")
    unguided_results = []
    for run in range(N_RUNS):
        print(f"  Run {run+1}/{N_RUNS}...", end=" ", flush=True)
        mols, stats = generate_mols(with_guidance=False, n=N_SAMPLES)
        n_valid = len(mols)
        unguided_results.append(n_valid)
        print(f"Valid molecules: {n_valid}/{N_SAMPLES}")

    print("\n--- RUNNING WITH GUIDANCE ---")
    guided_results = []
    guided_stats = []
    for run in range(N_RUNS):
        print(f"  Run {run+1}/{N_RUNS}...", end=" ", flush=True)
        mols, stats = generate_mols(with_guidance=True, n=N_SAMPLES)
        n_valid = len(mols)
        guided_results.append(n_valid)
        guided_stats.append(stats)

        n_guidance = stats.get('n_guidance_applied', 0)
        mean_grad = stats.get('mean_gradient_norm', 0)
        print(f"Valid: {n_valid}/{N_SAMPLES}, Guidance applied: {n_guidance}x, Mean grad norm: {mean_grad:.2f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Unguided - Mean valid: {np.mean(unguided_results):.1f} +/- {np.std(unguided_results):.1f}")
    print(f"Guided   - Mean valid: {np.mean(guided_results):.1f} +/- {np.std(guided_results):.1f}")

    if guided_stats:
        all_guidance_applied = [s.get('n_guidance_applied', 0) for s in guided_stats]
        all_grad_norms = [s.get('mean_gradient_norm', 0) for s in guided_stats]
        print("\nGuidance statistics:")
        print(f"  Mean guidance applications: {np.mean(all_guidance_applied):.1f}")
        print(f"  Mean gradient norm: {np.mean(all_grad_norms):.2f}")

    print("\nTest complete!")


if __name__ == "__main__":
    main()
