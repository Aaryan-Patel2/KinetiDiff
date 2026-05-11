#!/usr/bin/env python3
"""Compare guided vs unguided generation."""
import argparse
import os

from rdkit import Chem

from kinetidiff.gcdm.lightning_modules import LigandPocketDDPM


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare guided vs unguided generation.")
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

    def generate_mols(with_guidance, n=5):
        """Generate molecules, optionally with guidance."""
        if with_guidance:
            model.ddpm.enable_vina_guidance(
                receptor_pdbqt=args.receptor_pdbqt,
                binding_site_config=binding_site,
                guidance_scale=0.5,
                guidance_start_timestep=20,
                guidance_interval=5,
                n_gradient_atoms=3,
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
        return smiles_list

    print("\n=== Generating 3 molecules WITHOUT guidance ===")
    baseline = generate_mols(with_guidance=False, n=3)
    print(f"Baseline valid: {len(baseline)}")
    for i, smi in enumerate(baseline):
        print(f"  {i+1}. {smi[:60]}...")

    print("\n=== Generating 3 molecules WITH Vina guidance ===")
    guided = generate_mols(with_guidance=True, n=3)
    print(f"Guided valid: {len(guided)}")
    for i, smi in enumerate(guided):
        print(f"  {i+1}. {smi[:60]}...")

    print("\n=== SUMMARY ===")
    print(f"Baseline valid: {len(baseline)}")
    print(f"Guided valid: {len(guided)}")

    if hasattr(model.ddpm, "vina_guidance") and model.ddpm.vina_guidance:
        stats = model.ddpm.vina_guidance.get_stats()
        print("\nGuidance Statistics:")
        print(f"  Total Vina calls: {stats['n_vina_calls']}")
        print(f"  Total time: {stats['total_time']:.1f}s")
        print(f"  Guidance applied: {stats['n_guidance_applied']} times")

    print("\nComparison complete!")


if __name__ == "__main__":
    main()
