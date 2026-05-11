#!/usr/bin/env python3
"""
Quick Start: GCDM Generation with Real-Time Affinity Guidance

Generates molecules with REAL-TIME gradient-based guidance during denoising.
Affinity gradients steer molecules toward high-affinity binders during generation,
not after (post-hoc).

Usage:
    python quick_start.py \
        --n_samples 100 \
        --top_k 10 \
        --guided \
        --guidance_scale 50.0 \
        --pdb data/structures/receptor_siteA.pdb
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED, Descriptors

# BioPython compatibility - handle different versions
try:
    from Bio.PDB.Polypeptide import three_to_one
except ImportError:
    # Newer BioPython versions moved this
    from Bio.PDB.Polypeptide import protein_letters_3to1
    def three_to_one(residue_name):
        return protein_letters_3to1.get(residue_name, 'X')

PROJECT_ROOT = Path(__file__).parent.parent

from kinetidiff.guidance.affinity_guidance import AffinityGuidanceModel
from kinetidiff.guidance.docking_guidance import DockingGuidance
from kinetidiff.guidance.multi_objective import AdaptiveMultiObjectiveGuidance
from kinetidiff.guidance.sa_guidance import SAGuidance


def setup_gcdm_model(checkpoint_path: str, device: str = 'cuda'):
    """Load GCDM model from official gcdm-clone"""
    try:
        import pathlib

        import torch

        from kinetidiff.gcdm.lightning_modules import LigandPocketDDPM

        # Fix for PyTorch 2.6+ weights_only issue
        torch.serialization.add_safe_globals([pathlib.PosixPath])

        model = LigandPocketDDPM.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            weights_only=False  # Allow older checkpoints
        )
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading GCDM model: {e}")
        raise


def find_pocket_residues_from_coords(
    pdb_file: str,
    center_coords: tuple,
    radius: float = 8.0
) -> list[str]:
    """
    Find pocket residues within radius of center coordinates.

    Args:
        pdb_file: Path to PDB file
        center_coords: (x, y, z) coordinates of binding site center
        radius: Distance cutoff in Angstroms

    Returns:
        List of pocket residue IDs (e.g., ['A:123', 'A:124', ...])
    """
    import numpy as np
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    center = np.array(center_coords)
    pocket_residues = []

    for model in structure:
        for chain in model:
            for residue in chain:
                # Get CA atom (or any heavy atom if CA not present)
                ca_atom = None
                if 'CA' in residue:
                    ca_atom = residue['CA']
                else:
                    # Use first heavy atom
                    atoms = [a for a in residue.get_atoms() if a.element != 'H']
                    if atoms:
                        ca_atom = atoms[0]

                if ca_atom:
                    coord = ca_atom.get_coord()
                    distance = np.linalg.norm(coord - center)

                    if distance <= radius:
                        res_id = f"{chain.id}:{residue.id[1]}"
                        pocket_residues.append(res_id)

    return sorted(set(pocket_residues))


def generate_molecules(
    gcdm_model,
    pdb_file: str,
    ref_ligand: str,
    n_samples: int,
    batch_size: int = 50,
    device: str = 'cuda',
    binding_site_coords: tuple = None
) -> list[Chem.Mol]:
    """Generate molecules using GCDM"""
    print(f"\nGenerating {n_samples} molecules...")
    print(f"   PDB: {Path(pdb_file).name}")

    # Determine pocket residues
    if binding_site_coords:
        print(f"   Binding site coords: {binding_site_coords}")
        pocket_ids = find_pocket_residues_from_coords(pdb_file, binding_site_coords, radius=8.0)
        print(f"   Found {len(pocket_ids)} pocket residues within 8A")
        print(f"   Pocket: {pocket_ids[:5]}..." if len(pocket_ids) > 5 else f"   Pocket: {pocket_ids}")
    else:
        print(f"   Pocket: {ref_ligand}")
        # Use pocket_ids for siteA pocket (residues 203-232)
        pocket_ids = ['A:' + str(i) for i in range(203, 233)]

    print(f"   Batch size: {batch_size}\n")

    all_molecules = []
    start_time = time.time()

    for batch_idx in range(0, n_samples, batch_size):
        batch_n = min(batch_size, n_samples - batch_idx)
        batch_start = time.time()

        print(f"   Batch {batch_idx // batch_size + 1}: {batch_n} molecules...", end=" ", flush=True)

        try:
            molecules = gcdm_model.generate_ligands(
                pdb_file,
                n_samples=batch_n,
                pocket_ids=pocket_ids,
                ref_ligand=None,  # MUST be None when using pocket_ids
                num_nodes_lig=None,
                sanitize=True,
                largest_frag=True,
                relax_iter=0,
                sample_chain=False,
                resamplings=1,
                jump_length=1,
                timesteps=None
            )

            all_molecules.extend(molecules)

            batch_time = time.time() - batch_start
            elapsed_total = time.time() - start_time
            rate = len(all_molecules) / elapsed_total
            eta = (n_samples - len(all_molecules)) / rate if rate > 0 else 0

            print(f"done ({batch_time:.1f}s) | Total: {len(all_molecules)}/{n_samples} | ETA: {eta/60:.1f}min")

        except Exception as e:
            import traceback
            print(f"\nError generating batch: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            continue

    elapsed = time.time() - start_time
    print(f"\nGenerated {len(all_molecules)} molecules in {elapsed/60:.1f} minutes")
    print(f"   Rate: {len(all_molecules)/elapsed:.1f} mol/sec")

    return all_molecules


def generate_molecules_guided(
    gcdm_model,
    pdb_file: str,
    n_samples: int,
    affinity_checkpoint: str,
    protein_sequence: str,
    batch_size: int = 50,
    device: str = 'cuda',
    binding_site_coords: tuple = None,
    guidance_scale: float = 1.0,
    guidance_start_fraction: float = 0.0,
    guidance_end_fraction: float = 0.8,
    guidance_interval: int = 10,
    target_affinity: float = 7.0
) -> list[Chem.Mol]:
    """
    Generate molecules with REAL-TIME affinity guidance during denoising.

    This is the key improvement: gradients from the affinity model steer
    generation toward high-affinity molecules in real-time, not post-hoc.
    """
    print(f"\nGUIDED Generation: {n_samples} molecules with real-time affinity steering")
    print(f"   Guidance scale: {guidance_scale}")
    print(f"   Target affinity: {target_affinity} pKd")

    # Load affinity model using the working AffinityGuidanceModel
    affinity_guidance = None
    try:
        affinity_guidance = AffinityGuidanceModel(
            checkpoint_path=affinity_checkpoint,
            device=device,
            use_uncertainty=True
        )
        print("   Affinity guidance model loaded successfully")
    except Exception as e:
        print(f"   Could not load affinity guidance: {e}")
        print("   Falling back to unguided generation")
        return generate_molecules(gcdm_model, pdb_file, "A:1", n_samples,
                                   batch_size, device, binding_site_coords)

    # Determine pocket
    if binding_site_coords:
        print(f"   Binding site coords: {binding_site_coords}")
        pocket_ids = find_pocket_residues_from_coords(pdb_file, binding_site_coords, radius=8.0)
        print(f"   Found {len(pocket_ids)} pocket residues within 8A")
    else:
        pocket_ids = ['A:' + str(i) for i in range(203, 233)]

    print(f"   Batch size: {batch_size}\n")

    all_molecules = []
    all_guidance_history = []
    start_time = time.time()

    # Load PDB and prepare pocket once
    from Bio.PDB import PDBParser
    # BioPython compatibility - handle different versions
    try:
        from Bio.PDB.Polypeptide import three_to_one
    except ImportError:
        from Bio.PDB.Polypeptide import protein_letters_3to1
        def three_to_one(residue_name):
            return protein_letters_3to1.get(residue_name, 'X')
    import torch.nn.functional as F

    pdb_struct = PDBParser(QUIET=True).get_structure('', pdb_file)[0]
    residues = [
        pdb_struct[x.split(':')[0]][(' ', int(x.split(':')[1]), ' ')]
        for x in pocket_ids
    ]

    # Get pocket coordinates and types
    pocket_coord = torch.tensor(np.array(
        [res['CA'].get_coord() for res in residues if 'CA' in res]),
        device=device, dtype=torch.float32)

    pocket_type_encoder = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    pocket_types = torch.tensor(
        [pocket_type_encoder.get(three_to_one(res.get_resname()), 0)
         for res in residues if 'CA' in res], device=device)

    pocket_one_hot = F.one_hot(pocket_types, num_classes=20)

    for batch_idx in range(0, n_samples, batch_size):
        batch_n = min(batch_size, n_samples - batch_idx)
        batch_start = time.time()

        print(f"   Batch {batch_idx // batch_size + 1}: {batch_n} molecules (GUIDED)...", end=" ", flush=True)

        try:
            # Create pocket dict for this batch
            pocket_size = torch.tensor([len(pocket_coord)] * batch_n,
                                       device=device, dtype=torch.long)
            pocket_mask = torch.repeat_interleave(
                torch.arange(batch_n, device=device, dtype=torch.long),
                len(pocket_coord)
            )

            pocket = {
                'x': pocket_coord.repeat(batch_n, 1),
                'one_hot': pocket_one_hot.repeat(batch_n, 1).float(),
                'size': pocket_size,
                'mask': pocket_mask
            }

            # Sample number of atoms per ligand
            num_nodes_lig = gcdm_model.ddpm.size_distribution.sample_conditional(
                n1=None, n2=pocket['size'])

            # Guided sampling via DDPM
            # We need to hook into the sampling loop
            ddpm = gcdm_model.ddpm

            # Add guidance components temporarily
            ddpm._affinity_guidance = affinity_guidance
            ddpm._protein_sequence = protein_sequence
            ddpm._guidance_scale = guidance_scale
            ddpm._guidance_interval = guidance_interval
            ddpm._target_affinity = target_affinity

            # Use modified sample_given_pocket with guidance
            xh_lig, xh_pocket, lig_mask, pocket_mask_out, guidance_history = \
                sample_given_pocket_with_guidance(
                    ddpm, pocket, num_nodes_lig,
                    affinity_guidance=affinity_guidance,
                    protein_sequence=protein_sequence,
                    guidance_scale=guidance_scale,
                    guidance_interval=guidance_interval,
                    guidance_start_fraction=guidance_start_fraction,
                    guidance_end_fraction=guidance_end_fraction,
                    target_affinity=target_affinity,
                    verbose=False
                )

            all_guidance_history.extend(guidance_history)

            # Build molecules
            from kinetidiff.gcdm.analysis.molecule_builder import build_molecule, process_molecule

            lig_mask_cpu = lig_mask.cpu()
            x = xh_lig[:, :gcdm_model.x_dims].detach().cpu()
            atom_type = xh_lig[:, gcdm_model.x_dims:].argmax(1).detach().cpu()

            import kinetidiff.gcdm.utils as gcdm_utils
            for mol_pc in zip(gcdm_utils.batch_to_list(x, lig_mask_cpu),
                            gcdm_utils.batch_to_list(atom_type, lig_mask_cpu)):
                mol = build_molecule(*mol_pc, gcdm_model.dataset_info, add_coords=True)
                mol = process_molecule(mol, add_hydrogens=False, sanitize=True,
                                       relax_iter=0, largest_frag=True)
                if mol is not None:
                    all_molecules.append(mol)

            batch_time = time.time() - batch_start
            elapsed_total = time.time() - start_time
            rate = len(all_molecules) / elapsed_total if elapsed_total > 0 else 0
            eta = (n_samples - len(all_molecules)) / rate if rate > 0 else 0

            # Get mean affinity from guidance history
            mean_aff = np.mean([h['affinity'] for h in guidance_history]) if guidance_history else 0

            # Memory monitoring
            mem_used = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9

            print(f"done ({batch_time:.1f}s) | Valid: {len(all_molecules)} | pKd: {mean_aff:.2f} | Mem: {mem_used:.1f}GB/{mem_reserved:.1f}GB")

            # Clear cache after each batch
            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f"\nError in guided generation: {e}")
            print(f"   Traceback: {traceback.format_exc()[:500]}")
            # Clear cache even on error
            torch.cuda.empty_cache()
            continue

    elapsed = time.time() - start_time
    print(f"\nGUIDED: Generated {len(all_molecules)} molecules in {elapsed/60:.1f} minutes")

    if all_guidance_history:
        final_affinities = [h['affinity'] for h in all_guidance_history[-len(all_molecules):]]
        if final_affinities:
            print(f"   Mean final pKd: {np.mean(final_affinities):.2f}")

    return all_molecules


def sample_given_pocket_with_guidance(
    ddpm,
    pocket,
    num_nodes_lig,
    affinity_guidance,
    protein_sequence: str,
    guidance_scale: float = 1.0,
    guidance_interval: int = 10,
    guidance_start_fraction: float = 0.0,
    guidance_end_fraction: float = 0.8,
    target_affinity: float = 7.0,
    timesteps=None,
    verbose=True
):
    """
    Modified sampling with real-time affinity guidance injected.

    This is the CORE of the guided generation - injects gradients at each step.
    """
    # Handle imports
    try:
        from torch_scatter import scatter_add, scatter_mean
    except ImportError:
        from kinetidiff.gcdm.torch_scatter_impl import scatter_add, scatter_mean

    import kinetidiff.gcdm.utils as gcdm_utils

    timesteps = ddpm.T if timesteps is None else timesteps
    n_samples = len(pocket['size'])
    device = pocket['x'].device

    _, pocket_norm = ddpm.normalize(pocket=pocket)
    xh0_pocket = torch.cat([pocket_norm['x'], pocket_norm['one_hot']], dim=1)

    lig_mask = gcdm_utils.num_nodes_to_batch_mask(n_samples, num_nodes_lig, device)

    # Initialize from noise near pocket center
    mu_lig_x = scatter_mean(pocket_norm['x'], pocket_norm['mask'], dim=0)
    mu_lig_h = torch.zeros((n_samples, ddpm.atom_nf), device=device)
    mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)[lig_mask]
    sigma = torch.ones_like(pocket['size']).unsqueeze(1)

    z_lig, xh_pocket = ddpm.sample_normal_zero_com(
        mu_lig, xh0_pocket, sigma, lig_mask, pocket_norm['mask'])

    # Guidance step bounds
    guidance_start_step = int(timesteps * (1 - guidance_end_fraction))
    guidance_end_step = int(timesteps * (1 - guidance_start_fraction))

    guidance_history = []

    # Main denoising loop with guidance
    for s in reversed(range(0, timesteps)):
        s_array = torch.full((n_samples, 1), fill_value=s, device=device)
        t_array = s_array + 1
        s_array = s_array / timesteps
        t_array = t_array / timesteps

        # Standard denoising step - use no_grad for memory efficiency
        with torch.no_grad():
            z_lig, xh_pocket = ddpm.sample_p_zs_given_zt(
                s_array, t_array, z_lig, xh_pocket, lig_mask, pocket_norm['mask'])

        # ========== APPLY GUIDANCE ==========
        if (affinity_guidance is not None and
            guidance_start_step <= s < guidance_end_step and
            s % guidance_interval == 0):

            z_lig, metrics = apply_affinity_guidance_step(
                z_lig, lig_mask, pocket_norm,
                affinity_guidance, protein_sequence,
                guidance_scale, target_affinity,
                s, timesteps, ddpm.n_dims
            )
            guidance_history.append(metrics)

            # CRITICAL: Detach z_lig to break computational graph and free memory
            z_lig = z_lig.detach()

            if verbose and s % 100 == 0:
                print(f"Step {s}: pKd={metrics['affinity']:.2f}")
        # ====================================

        # Clear CUDA cache every 10 steps to prevent memory accumulation
        if s % 10 == 0:
            torch.cuda.empty_cache()

    # Final sampling p(x,h | z_0)
    x_lig, h_lig, x_pocket, h_pocket = ddpm.sample_p_xh_given_z0(
        z_lig, xh_pocket, lig_mask, pocket_norm['mask'], n_samples)

    # Correct CoM drift
    try:
        from torch_scatter import scatter_add
    except ImportError:
        from kinetidiff.gcdm.torch_scatter_impl import scatter_add

    max_cog = scatter_add(x_lig, lig_mask, dim=0).abs().max().item()
    if max_cog > 5e-2:
        x_lig, x_pocket = ddpm.remove_mean_batch(
            x_lig, x_pocket, lig_mask, pocket_norm['mask'])

    xh_lig = torch.cat([x_lig, h_lig], dim=1)
    xh_pocket = torch.cat([x_pocket, h_pocket], dim=1)

    return xh_lig, xh_pocket, lig_mask, pocket_norm['mask'], guidance_history


def apply_affinity_guidance_step(
    z_lig: torch.Tensor,
    lig_mask: torch.Tensor,
    pocket: dict,
    affinity_guidance,
    protein_sequence: str,
    guidance_scale: float,
    target_affinity: float,
    step: int,
    total_steps: int,
    n_dims: int = 3
):
    """
    Apply single step of affinity gradient guidance.

    Computes gradient of affinity w.r.t. coordinates and applies it.
    """
    device = z_lig.device
    n_samples = int(lig_mask.max().item()) + 1

    all_affinities = []
    all_grad_norms = []

    # Timestep-dependent scaling (stronger early, weaker late)
    t_fraction = step / total_steps
    timestep_scale = t_fraction

    for mol_idx in range(n_samples):
        mol_mask = (lig_mask == mol_idx)

        # Extract coords and features
        mol_z = z_lig[mol_mask]
        mol_coords = mol_z[:, :n_dims]
        mol_features = mol_z[:, n_dims:]

        # Get pocket coords
        pocket_mask = (pocket['mask'] == mol_idx)
        pocket_coords = pocket['x'][pocket_mask]

        try:
            # Clone and enable grad only for gradient computation
            mol_coords_grad = mol_coords.clone().detach().requires_grad_(True)

            with torch.enable_grad():
                gradient, affinity = affinity_guidance.compute_affinity_gradient(
                    mol_coords_grad,
                    mol_features.detach(),  # No grad needed for features
                    pocket_coords.detach(),  # No grad needed for pocket
                    protein_sequence,
                    target_affinity
                )

            # Scale and clip gradient - detach immediately
            scaled_grad = (gradient * guidance_scale * timestep_scale).detach()
            grad_norm = scaled_grad.norm().item()  # Convert to python float

            if grad_norm > 1.0:
                scaled_grad = scaled_grad / grad_norm

            # Apply gradient with no_grad context
            with torch.no_grad():
                z_lig[mol_mask, :n_dims] = z_lig[mol_mask, :n_dims] + scaled_grad

            all_affinities.append(affinity)
            all_grad_norms.append(grad_norm)

            # FREE MEMORY IMMEDIATELY
            del gradient, scaled_grad, mol_coords_grad

        except Exception:
            all_affinities.append(0.0)
            all_grad_norms.append(0.0)

    metrics = {
        'affinity': np.mean(all_affinities),
        'grad_norm': np.mean(all_grad_norms),
        'timestep': step,
        'scale': timestep_scale * guidance_scale
    }

    return z_lig, metrics


def score_molecules(
    molecules: list[Chem.Mol],
    affinity_guidance: AffinityGuidanceModel | None = None,
    docking_guidance: DockingGuidance | None = None,
    sa_guidance: SAGuidance | None = None,
    protein_sequence: str | None = None
) -> list[dict]:
    """Score molecules with guidance models"""
    print(f"\nScoring {len(molecules)} molecules...")

    metrics_list = []

    for i, mol in enumerate(molecules):
        if i > 0 and i % max(100, len(molecules) // 10) == 0:
            print(f"   Progress: {i}/{len(molecules)}")

        try:
            smiles = Chem.MolToSmiles(mol)
            if not smiles:
                continue

            metrics = {
                'smiles': smiles,
                'mol_weight': float(Descriptors.MolWt(mol)),
                'logp': float(Descriptors.MolLogP(mol)),
                'qed': float(QED.qed(mol)),
            }

            # Apply filters: QED >= 0.3 and MW < 400
            if metrics['qed'] < 0.3 or metrics['mol_weight'] >= 400:
                continue

            # Affinity prediction
            if affinity_guidance and protein_sequence:
                try:
                    affinity = affinity_guidance.predict_affinity(smiles, protein_sequence)
                    metrics['affinity_pkd'] = float(affinity)
                except Exception as e:
                    print(f"   Warning: Affinity prediction failed for {smiles[:30]}: {e}")
                    metrics['affinity_pkd'] = 6.0  # Default moderate affinity
            else:
                metrics['affinity_pkd'] = 6.0

            # Docking score
            if docking_guidance:
                try:
                    docking = docking_guidance.score_molecule(mol)
                    metrics['docking_score'] = float(docking)
                except Exception as e:
                    print(f"   Warning: Docking failed for {smiles[:30]}: {e}")
                    metrics['docking_score'] = -9.0
            else:
                metrics['docking_score'] = -9.0

            # SA score
            if sa_guidance:
                try:
                    sa = sa_guidance.score_molecule(mol)
                    metrics['sa_score'] = float(sa)
                except Exception as e:
                    print(f"   Warning: SA scoring failed for {smiles[:30]}: {e}")
                    metrics['sa_score'] = 3.5
            else:
                metrics['sa_score'] = 3.5
                metrics['docking_score'] = 0.0

            # SA score
            if sa_guidance:
                try:
                    sa = sa_guidance.score_molecule(mol)
                    metrics['sa_score'] = float(sa)
                except:
                    metrics['sa_score'] = 5.0
            else:
                metrics['sa_score'] = 5.0

            metrics_list.append(metrics)

        except Exception:
            continue

    print(f"Scored {len(metrics_list)} molecules")

    return metrics_list


def optimize_and_rank(
    metrics: list[dict],
    optimizer: AdaptiveMultiObjectiveGuidance | None = None,
    top_k: int = 100
) -> list[dict]:
    """Apply multi-objective optimization and select top-K molecules"""
    print(f"\nOptimizing and ranking {len(metrics)} molecules...")

    if optimizer and len(metrics) > 0:
        # Compute combined loss for each molecule
        for m in metrics:
            # Prepare metrics dict for the optimizer
            obj_values = {
                'affinity_pkd': m.get('affinity_pkd', 6.0),
                'docking_score': m.get('docking_score', -9.0),
                'sa_score': m.get('sa_score', 3.0)
            }

            # Compute combined loss using separate tensor args
            import torch
            affinity_t = torch.tensor([obj_values['affinity_pkd']], dtype=torch.float32)
            docking_t = torch.tensor([obj_values['docking_score']], dtype=torch.float32)
            sa_t = torch.tensor([obj_values['sa_score']], dtype=torch.float32)
            loss, _ = optimizer.compute_combined_loss(affinity_t, docking_t, sa_t)
            m['optimizer_score'] = -loss.item()  # Negate so higher is better

        ranked = sorted(metrics, key=lambda m: m.get('optimizer_score', 0), reverse=True)
    else:
        # Fallback: simple composite score
        for m in metrics:
            affinity_norm = max(0, min(1, (m['affinity_pkd'] - 5.0) / 3.0))
            docking_norm = max(0, min(1, (m['docking_score'] + 12) / 7.0))
            sa_norm = max(0, 1 - (m['sa_score'] / 10.0))

            m['composite_score'] = (
                0.3 * affinity_norm +
                0.4 * docking_norm +
                0.3 * sa_norm
            )

        ranked = sorted(metrics, key=lambda m: m.get('composite_score', 0), reverse=True)

    top_molecules = ranked[:min(top_k, len(ranked))]

    print(f"Selected top {len(top_molecules)} molecules")

    # Print top 5
    print("\nTop 5 Molecules:")
    print(f"{'Rank':<5} {'pKd':<7} {'Dock':<8} {'SA':<7} {'Score':<7}")
    print("-" * 40)
    for rank, m in enumerate(top_molecules[:5], 1):
        pKd = m.get('affinity_pkd', 0)
        dock = m.get('docking_score', 0)
        sa = m.get('sa_score', 0)
        score = m.get('composite_score', 0)
        print(f"{rank:<5} {pKd:<7.2f} {dock:<8.2f} {sa:<7.2f} {score:<7.3f}")

    return top_molecules


def save_results(
    molecules: list[dict],
    output_dir: str,
    n_total: int,
    exec_time: float
):
    """Save results to CSV and JSON"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to {output_dir}...")

    # CSV
    csv_path = output_dir / 'top_molecules.csv'
    with open(csv_path, 'w', newline='') as f:
        if molecules:
            writer = csv.DictWriter(f, fieldnames=molecules[0].keys())
            writer.writeheader()
            writer.writerows(molecules)
    print(f"   {csv_path.name}")

    # JSON summary
    json_path = output_dir / 'generation_summary.json'
    with open(json_path, 'w') as f:
        json.dump({
            'total_generated': n_total,
            'total_saved': len(molecules),
            'execution_time_seconds': exec_time,
            'execution_time_minutes': exec_time / 60,
            'top_5': molecules[:5],
            'statistics': {
                'mean_affinity': float(np.mean([m.get('affinity_pkd', 0) for m in molecules])),
                'mean_docking': float(np.mean([m.get('docking_score', 0) for m in molecules])),
                'mean_sa': float(np.mean([m.get('sa_score', 0) for m in molecules])),
                'mean_composite': float(np.mean([m.get('composite_score', 0) for m in molecules])),
            }
        }, f, indent=2)
    print(f"   {json_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='GCDM Generation with Post-Hoc Guidance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_start.py --n_samples 3000 --top_k 100
  python quick_start.py --pdb data/structures/receptor.pdb --ref_ligand A:5
  python quick_start.py --n_samples 100 --top_k 20
        """
    )

    # Auto-detect checkpoints
    def find_checkpoint(pattern: str) -> str:
        """Find checkpoint matching pattern in models/checkpoints/"""
        checkpoints_dir = PROJECT_ROOT / "models" / "checkpoints"
        if checkpoints_dir.exists():
            for ckpt in checkpoints_dir.glob(f"*{pattern}*"):
                return str(ckpt)
        return None

    gcdm_ckpt_default = find_checkpoint("gcdm") or find_checkpoint("bindingmoad")
    gcdm_ckpt_default = gcdm_ckpt_default or str(PROJECT_ROOT / "models" / "gcdm-clone" / "checkpoints" / "bindingmoad_ca_cond_gcpnet.ckpt")

    affinity_ckpt_default = find_checkpoint("affinity") or find_checkpoint("best_model")
    affinity_ckpt_default = affinity_ckpt_default or str(PROJECT_ROOT / "models" / "affinity_pred" / "checkpoints" / "best_model.pt")

    # GCDM & paths
    parser.add_argument('--gcdm_checkpoint',
                       default=gcdm_ckpt_default,
                       help='Path to GCDM checkpoint')
    parser.add_argument('--pdb',
                       default=str(PROJECT_ROOT / 'data' / 'structures' / 'receptor_siteA.pdb'),
                       help='Receptor PDB file')
    parser.add_argument('--ref_ligand',
                       default='A:1',
                       help='Reference ligand for pocket (chain:residue)')
    # ACVR1 (ALK2) kinase domain sequence for affinity prediction
    ACVR1_SEQUENCE = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS"

    parser.add_argument('--protein_sequence',
                       default=ACVR1_SEQUENCE,
                       help='Protein sequence for affinity prediction (default: ACVR1 kinase domain)')
    parser.add_argument('--output',
                       default=str(PROJECT_ROOT / 'results' / 'quick_start'),
                       help='Output directory')

    # Generation parameters
    parser.add_argument('--n_samples', type=int, default=3000,
                       help='Number of molecules to generate')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='Batch size for generation')
    parser.add_argument('--top_k', type=int, default=100,
                       help='Number of top molecules to save')

    # REAL-TIME GUIDANCE (NEW!)
    parser.add_argument('--guided', action='store_true', default=False,
                       help='Use REAL-TIME affinity guidance during generation (not post-hoc)')
    parser.add_argument('--guidance_scale', type=float, default=50.0,
                       help='Strength of real-time guidance (higher = stronger steering)')
    parser.add_argument('--guidance_interval', type=int, default=10,
                       help='Apply guidance every N denoising steps')
    parser.add_argument('--guidance_start', type=float, default=0.0,
                       help='Fraction of generation to start guidance (0.0 = immediately)')
    parser.add_argument('--guidance_end', type=float, default=0.8,
                       help='Fraction of generation to stop guidance (0.8 = stop at 80%)')

    # Guidance parameters
    parser.add_argument('--affinity_checkpoint',
                       default=affinity_ckpt_default,
                       help='Path to affinity predictor checkpoint')
    parser.add_argument('--use_affinity', action='store_true', default=True,
                       help='Use affinity guidance')
    parser.add_argument('--use_docking', action='store_true', default=True,
                       help='Use docking guidance')
    parser.add_argument('--use_sa', action='store_true', default=True,
                       help='Use SA guidance')

    # Thresholds
    parser.add_argument('--affinity_target', type=float, default=7.0,
                       help='Target affinity (pKd)')
    parser.add_argument('--affinity_min', type=float, default=6.0,
                       help='Minimum affinity threshold')
    parser.add_argument('--docking_target', type=float, default=-11.0,
                       help='Target docking score (kcal/mol)')
    parser.add_argument('--docking_max', type=float, default=-10.0,
                       help='Maximum docking threshold (kcal/mol)')
    parser.add_argument('--sa_target', type=float, default=3.0,
                       help='Target SA score')
    parser.add_argument('--sa_max', type=float, default=3.5,
                       help='Maximum SA threshold')

    # Device
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device: cuda or cpu')

    args = parser.parse_args()

    if args.guided:
        print("\n" + "=" * 80)
        print("GCDM GENERATION WITH REAL-TIME AFFINITY GUIDANCE")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("GCDM GENERATION WITH POST-HOC GUIDANCE")
    print("=" * 80)
    print(f"\nDevice: {args.device}")
    if args.device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Load GCDM model
    print("\nLoading GCDM model...")
    try:
        gcdm_model = setup_gcdm_model(args.gcdm_checkpoint, device=args.device)
        print("   GCDM model loaded")
    except Exception as e:
        print(f"   Failed to load GCDM: {e}")
        sys.exit(1)

    # Load guidance models
    print("\nLoading guidance models...")

    affinity_guidance = None
    if args.use_affinity:
        try:
            affinity_guidance = AffinityGuidanceModel(args.affinity_checkpoint)
            print("   Affinity guidance loaded")
        except Exception as e:
            print("   Affinity guidance disabled (use docking + SA scoring)")
            print(f"   Debug: {e}")
            affinity_guidance = None

    docking_guidance = None
    if args.use_docking:
        try:
            docking_guidance = DockingGuidance(use_fast_approximation=True)
            print("   Docking guidance loaded")
        except Exception as e:
            print(f"   Docking guidance failed: {e}")

    sa_guidance = None
    if args.use_sa:
        try:
            sa_guidance = SAGuidance(target_sa=args.sa_target, max_sa=args.sa_max)
            print("   SA guidance loaded")
        except Exception as e:
            print(f"   SA guidance failed: {e}")

    # Load multi-objective optimizer
    print("\nLoading multi-objective optimizer...")
    try:
        optimizer = AdaptiveMultiObjectiveGuidance(
            strategy='adaptive_threshold',
            affinity_target=args.affinity_target,
            affinity_min=args.affinity_min,
            docking_target=args.docking_target,
            docking_max=args.docking_max,
            sa_target=args.sa_target,
            sa_max=args.sa_max
        )
        print("   Optimizer loaded")
    except Exception as e:
        print(f"   Optimizer failed: {e}")
        optimizer = None

    # Generate
    start_time = time.time()

    # ACVR1 binding site coordinates (active site)
    ACVR1_BINDING_SITE = (24.87, -12.54, 38.40)

    if args.guided:
        # REAL-TIME GUIDANCE: Gradients applied during denoising
        print("\nUsing REAL-TIME affinity guidance during generation")
        print("   This steers molecules toward higher affinity IN REAL-TIME")

        molecules = generate_molecules_guided(
            gcdm_model,
            args.pdb,
            n_samples=args.n_samples,
            affinity_checkpoint=args.affinity_checkpoint,
            protein_sequence=args.protein_sequence,
            batch_size=args.batch_size,
            device=args.device,
            binding_site_coords=ACVR1_BINDING_SITE,
            guidance_scale=args.guidance_scale,
            guidance_start_fraction=args.guidance_start,
            guidance_end_fraction=args.guidance_end,
            guidance_interval=args.guidance_interval,
            target_affinity=args.affinity_target
        )
    else:
        # POST-HOC SCORING: Generate first, score after
        molecules = generate_molecules(
            gcdm_model,
            args.pdb,
            args.ref_ligand,
            args.n_samples,
            batch_size=args.batch_size,
            device=args.device,
            binding_site_coords=ACVR1_BINDING_SITE
        )

    if not molecules:
        print("No molecules generated!")
        sys.exit(1)

    # Score
    metrics = score_molecules(
        molecules,
        affinity_guidance=affinity_guidance,
        docking_guidance=docking_guidance,
        sa_guidance=sa_guidance,
        protein_sequence=args.protein_sequence
    )

    if not metrics:
        print("No molecules scored!")
        sys.exit(1)

    # Optimize and rank
    top_molecules = optimize_and_rank(
        metrics,
        optimizer=optimizer,
        top_k=args.top_k
    )

    # Save
    elapsed = time.time() - start_time
    save_results(
        top_molecules,
        args.output,
        len(molecules),
        elapsed
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total generated:        {len(molecules)}")
    print(f"Valid molecules:        {len(metrics)}")
    print(f"Top molecules saved:    {len(top_molecules)}")
    print(f"Total time:             {elapsed/60:.1f} minutes")
    print("\nTop molecules Statistics:")
    if top_molecules:
        print(f"  Mean Affinity (pKd):  {np.mean([m.get('affinity_pkd', 0) for m in top_molecules]):.2f}")
        print(f"  Mean Docking:         {np.mean([m.get('docking_score', 0) for m in top_molecules]):.2f}")
        print(f"  Mean SA Score:        {np.mean([m.get('sa_score', 0) for m in top_molecules]):.2f}")

    print(f"\nComplete! Results saved to: {args.output}\n")


if __name__ == '__main__':
    main()
