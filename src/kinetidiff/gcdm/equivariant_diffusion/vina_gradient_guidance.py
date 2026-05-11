"""
Vina Gradient Guidance for GCDM Diffusion

This module implements TRUE gradient-guided diffusion by injecting
Vina score gradients at each denoising timestep.

Key difference from post-hoc filtering:
- POST-HOC: generate → score → filter (what we had before)
- GUIDED: at each timestep: denoise → compute gradient → apply → continue

The gradient steers molecules toward lower Vina scores DURING generation.
"""

import subprocess
import time
from pathlib import Path

import numpy as np
import torch

# RDKit
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')


# GCDM atom type mapping (index -> atomic number)
# Based on GCDM's BindingMOAD encoding
GCDM_ATOM_MAP = {
    0: 6,   # C
    1: 7,   # N
    2: 8,   # O
    3: 9,   # F
    4: 15,  # P
    5: 16,  # S
    6: 17,  # Cl
    7: 35,  # Br
    8: 53,  # I
}

# Inverse map for debugging
ATOMIC_NUM_TO_SYMBOL = {
    6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}


class GradientProcessor:
    """
    ML-inspired gradient processing for stable diffusion guidance.

    Instead of naive clipping, we treat raw Vina gradients as noisy signals
    that need proper normalization and activation:

    1. Running statistics (like BatchNorm): track gradient mean/variance
    2. Normalization: standardize to zero mean, unit variance
    3. Activation: tanh to smoothly bound to [-1, 1]
    4. Scaling: learnable or adaptive output scaling

    This provides much smoother, more stable gradients than hard clipping.
    """

    def __init__(
        self,
        momentum: float = 0.1,  # EMA momentum for running stats
        eps: float = 1e-5,  # Numerical stability
        output_scale: float = 1.0,  # Final scaling factor
        use_tanh: bool = True,  # Apply tanh activation
        warmup_steps: int = 10,  # Steps before using running stats
    ):
        """
        Args:
            momentum: Exponential moving average momentum (like BatchNorm)
            eps: Small constant for numerical stability
            output_scale: Scale factor applied after normalization
            use_tanh: Whether to apply tanh activation
            warmup_steps: Number of steps to collect stats before normalizing
        """
        self.momentum = momentum
        self.eps = eps
        self.output_scale = output_scale
        self.use_tanh = use_tanh
        self.warmup_steps = warmup_steps

        # Running statistics (per dimension: x, y, z)
        self.running_mean = torch.zeros(3)
        self.running_var = torch.ones(3)
        self.n_updates = 0

        # Statistics for monitoring
        self.raw_grad_norms = []
        self.processed_grad_norms = []

    def update_running_stats(self, gradient: torch.Tensor):
        """Update running mean and variance with new gradient batch."""
        if gradient.numel() == 0:
            return

        # Compute per-dimension mean and var
        # gradient shape: [n_atoms, 3]
        grad_mean = gradient.mean(dim=0)  # [3]
        grad_var = gradient.var(dim=0, unbiased=False) + self.eps  # [3]

        # Move to same device
        if self.running_mean.device != gradient.device:
            self.running_mean = self.running_mean.to(gradient.device)
            self.running_var = self.running_var.to(gradient.device)

        # EMA update (like BatchNorm)
        if self.n_updates == 0:
            # First update: just set
            self.running_mean = grad_mean.detach()
            self.running_var = grad_var.detach()
        else:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * grad_mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * grad_var.detach()

        self.n_updates += 1

    def process(self, gradient: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        """
        Process raw gradient through normalization and activation.

        Args:
            gradient: Raw gradient tensor [n_atoms, 3]
            update_stats: Whether to update running statistics

        Returns:
            Processed gradient tensor [n_atoms, 3]
        """
        if gradient.numel() == 0:
            return gradient

        # Track raw gradient norm
        raw_norm = gradient.norm().item()
        self.raw_grad_norms.append(raw_norm)

        # Skip processing if gradient is zero
        if raw_norm < self.eps:
            self.processed_grad_norms.append(0.0)
            return gradient

        # Update running statistics
        if update_stats:
            self.update_running_stats(gradient)

        # During warmup, use batch statistics
        if self.n_updates < self.warmup_steps:
            mean = gradient.mean(dim=0, keepdim=True)
            std = gradient.std(dim=0, keepdim=True) + self.eps
        else:
            # Use running statistics
            mean = self.running_mean.unsqueeze(0)
            std = torch.sqrt(self.running_var).unsqueeze(0) + self.eps

        # Normalize: (x - mean) / std
        normalized = (gradient - mean) / std

        # Apply activation (tanh bounds to [-1, 1])
        if self.use_tanh:
            activated = torch.tanh(normalized)
        else:
            activated = normalized

        # Scale output
        processed = activated * self.output_scale

        # Track processed norm
        self.processed_grad_norms.append(processed.norm().item())

        return processed

    def get_stats(self) -> dict:
        """Get processing statistics."""
        return {
            'n_updates': self.n_updates,
            'running_mean': self.running_mean.tolist() if self.n_updates > 0 else [0, 0, 0],
            'running_var': self.running_var.tolist() if self.n_updates > 0 else [1, 1, 1],
            'avg_raw_norm': np.mean(self.raw_grad_norms) if self.raw_grad_norms else 0,
            'avg_processed_norm': np.mean(self.processed_grad_norms) if self.processed_grad_norms else 0,
            'max_raw_norm': max(self.raw_grad_norms) if self.raw_grad_norms else 0,
            'max_processed_norm': max(self.processed_grad_norms) if self.processed_grad_norms else 0,
        }

    def reset_stats(self):
        """Reset running statistics."""
        self.running_mean = torch.zeros(3)
        self.running_var = torch.ones(3)
        self.n_updates = 0
        self.raw_grad_norms = []
        self.processed_grad_norms = []


def get_adaptive_scale(timestep: int) -> float:
    """Adaptive guidance-strength multiplier — small early (noisy), large late (refined).

    Why: gradient signal-to-noise is poor at high t (random coords); aggressive scaling
    there pushes molecules into unphysical configurations. Late steps refine — heavier
    guidance is safe.
    """
    if timestep > 600:
        return 0.1
    if timestep > 400:
        return 0.3
    if timestep > 200:
        return 0.7
    return 1.5


class VinaGradientGuidance:
    """
    Compute Vina gradients for GCDM diffusion guidance.

    This class is designed to be called at each denoising timestep
    to provide gradient updates that steer generation toward
    high-affinity molecules.

    Performance optimizations:
    - Only compute gradients at specified intervals
    - Use --score_only mode (10-100x faster than docking)
    - Sample subset of atoms for gradient estimation
    - Adaptive guidance scaling based on timestep
    """

    def __init__(
        self,
        receptor_pdbqt: str,
        binding_site_config: dict[str, float],
        vina_executable: str = 'vina',
        guidance_start_timestep: int = 400,  # t < 400 (clean molecules)
        guidance_interval: int = 20,  # Apply every N steps
        n_gradient_atoms: int = 5,  # Sample this many atoms for gradient
        eps: float = 0.2,  # Finite difference step (Angstroms)
        max_grad_norm: float = 5.0,  # Maximum gradient norm (for fallback clipping)
        max_vina_for_guidance: float = 500.0,  # Skip if Vina worse than this (high since we use GradientProcessor)
        device: str = 'cpu',
        # Gradient processing parameters
        use_gradient_processor: bool = True,
        gradient_momentum: float = 0.1,
        gradient_output_scale: float = 1.0,
        gradient_use_tanh: bool = True,
        gradient_warmup_steps: int = 5,
    ):
        """
        Args:
            receptor_pdbqt: Path to prepared receptor
            binding_site_config: Dict with center_x/y/z and size_x/y/z
            guidance_start_timestep: Only guide when t < this (0-1000 scale)
            guidance_interval: Apply gradient every N timesteps
            n_gradient_atoms: Number of atoms to sample for gradient
            eps: Perturbation size for numerical gradient
            max_grad_norm: Clip gradients to this maximum norm
            max_vina_for_guidance: Skip guidance if Vina score exceeds this
            use_gradient_processor: Whether to use ML-style gradient processing
            gradient_momentum: EMA momentum for running stats
            gradient_output_scale: Scale factor for processed gradients
            gradient_use_tanh: Whether to apply tanh activation
            gradient_warmup_steps: Steps before using running stats
        """
        self.receptor_pdbqt = Path(receptor_pdbqt)
        if not self.receptor_pdbqt.exists():
            raise FileNotFoundError(f"Receptor not found: {receptor_pdbqt}")

        self.binding_site = binding_site_config
        self.vina_executable = vina_executable
        self.guidance_start_timestep = guidance_start_timestep
        self.guidance_interval = guidance_interval
        self.n_gradient_atoms = n_gradient_atoms
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        self.max_vina_for_guidance = max_vina_for_guidance
        self.device = device

        # Initialize gradient processor
        self.use_gradient_processor = use_gradient_processor
        if use_gradient_processor:
            self.gradient_processor = GradientProcessor(
                momentum=gradient_momentum,
                output_scale=gradient_output_scale,
                use_tanh=gradient_use_tanh,
                warmup_steps=gradient_warmup_steps,
            )
            print(f"  Gradient processor: BatchNorm + {'tanh' if gradient_use_tanh else 'linear'}, scale={gradient_output_scale}")
        else:
            self.gradient_processor = None

        # Cache directory
        self.cache_dir = Path('/tmp/vina_guidance_cache')
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Check Vina availability
        self._check_vina()

        # Performance tracking
        self.vina_calls = 0
        self.total_time = 0
        self.guidance_applied = 0

        print("✓ VinaGradientGuidance initialized")
        print(f"  Receptor: {self.receptor_pdbqt}")
        print(f"  Start timestep: {guidance_start_timestep} (of 1000)")
        print(f"  Interval: every {guidance_interval} steps")
        print(f"  Gradient atoms: {n_gradient_atoms}")

    def _check_vina(self):
        """Verify Vina is available."""
        try:
            result = subprocess.run(
                [self.vina_executable, '--version'],
                capture_output=True, text=True, timeout=10
            )
            self.vina_available = True
            print(f"  Vina: {self.vina_executable} (available)")
        except Exception as e:
            self.vina_available = False
            print(f"  ⚠ Vina not available: {e}")

    def should_apply(self, timestep: int, total_timesteps: int = 1000) -> bool:
        """
        Determine if guidance should be applied at this timestep.

        Args:
            timestep: Current step (0 = final, total_timesteps = start)
            total_timesteps: Total number of timesteps

        Returns:
            bool: True if guidance should be applied
        """
        if not self.vina_available:
            return False

        # Scale timestep to 0-1000 range
        t_scaled = int(timestep * 1000 / total_timesteps) if total_timesteps != 1000 else timestep

        # Only guide clean molecules (late in diffusion)
        if t_scaled > self.guidance_start_timestep:
            return False

        # Only guide at intervals
        if t_scaled % self.guidance_interval != 0:
            return False

        return True

    def compute_gradient(
        self,
        coords: torch.Tensor,  # (n_atoms, 3) or (batch, n_atoms, 3)
        atom_types: torch.Tensor,  # (n_atoms, n_types) one-hot or (n_atoms,) indices
        ligand_mask: torch.Tensor,  # (n_atoms,) boolean mask
        timestep: int,
        pocket_center: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute Vina gradient for coordinate update.

        This is THE KEY FUNCTION for guided diffusion.

        Args:
            coords: Ligand coordinates
            atom_types: Atom type encoding
            ligand_mask: Which atoms are valid ligand atoms
            timestep: Current diffusion timestep
            pocket_center: Center of binding pocket (for fallback)

        Returns:
            gradient: (n_atoms, 3) gradient tensor
            metrics: Dict with vina_score, n_calls, etc.
        """
        start_time = time.time()

        # Handle batched input
        if coords.dim() == 3:
            # Process first sample only for now
            coords = coords[0]
            if atom_types.dim() == 3:
                atom_types = atom_types[0]
            if ligand_mask.dim() == 2:
                ligand_mask = ligand_mask[0]

        # Initialize gradient
        gradient = torch.zeros_like(coords)
        metrics = {
            'vina_score': 0.0,
            'gradient_norm': 0.0,
            'n_vina_calls': 0,
            'valid_molecule': False
        }

        try:
            # Extract valid atoms
            valid_mask = ligand_mask.bool()
            valid_coords = coords[valid_mask]
            valid_atom_types = atom_types[valid_mask] if atom_types.dim() == 1 else atom_types[valid_mask]

            if valid_coords.shape[0] < 3:
                return gradient, metrics

            # Convert to RDKit molecule
            debug_mol = (timestep <= 10)  # Debug at final timesteps
            mol = self._coords_to_rdkit(valid_coords, valid_atom_types, debug=debug_mol)

            if mol is None:
                # Fallback: use distance-based guidance
                if pocket_center is not None:
                    gradient = self._distance_guidance(coords, pocket_center, ligand_mask)
                    metrics['gradient_norm'] = gradient.norm().item()
                return gradient, metrics

            # Compute current Vina score
            debug_vina = (timestep <= 10)  # Debug at final timesteps
            current_score = self._compute_vina_score(mol, debug=debug_vina)
            metrics['vina_score'] = current_score
            metrics['n_vina_calls'] += 1

            if current_score is None:
                return gradient, metrics

            metrics['valid_molecule'] = True

            # Compute numerical gradient (sample atoms)
            n_atoms = valid_coords.shape[0]
            n_sample = min(self.n_gradient_atoms, n_atoms)

            # Prefer heavy atoms (not H)
            if valid_atom_types.dim() == 2:
                atom_indices = valid_atom_types.argmax(dim=-1)
            else:
                atom_indices = valid_atom_types

            # Sample atoms for gradient computation
            sample_indices = torch.randperm(n_atoms)[:n_sample]

            valid_gradient = torch.zeros_like(valid_coords)

            for atom_idx in sample_indices:
                for dim in range(3):
                    # Forward perturbation
                    coords_plus = valid_coords.clone()
                    coords_plus[atom_idx, dim] += self.eps

                    mol_plus = self._coords_to_rdkit(coords_plus, valid_atom_types)
                    if mol_plus is None:
                        continue

                    score_plus = self._compute_vina_score(mol_plus)
                    metrics['n_vina_calls'] += 1

                    if score_plus is not None:
                        # Forward difference: (f(x+h) - f(x)) / h
                        # This gradient points toward INCREASING score
                        valid_gradient[atom_idx, dim] = (score_plus - current_score) / self.eps

            # NOTE: We return the raw gradient (pointing toward higher scores)
            # The caller subtracts this: coords -= scale * gradient
            # So we move OPPOSITE to gradient = toward LOWER scores = what we want!
            # DO NOT negate here - the subtraction in conditional_model.py handles it

            # Propagate gradient to all atoms (simple: use mean for unsampled)
            if n_sample < n_atoms:
                mean_grad = valid_gradient[sample_indices].mean(dim=0)
                for i in range(n_atoms):
                    if i not in sample_indices:
                        valid_gradient[i] = mean_grad

            # Put back into full gradient tensor
            gradient[valid_mask] = valid_gradient

            # Skip guidance entirely for severely clashing molecules
            # (Vina > threshold indicates very bad structure, gradient is meaningless)
            if current_score is not None and current_score > self.max_vina_for_guidance:
                gradient = torch.zeros_like(gradient)
                if debug_mol:
                    print(f"  Vina={current_score:.1f} too high (>{self.max_vina_for_guidance}), skipping gradient")
                metrics['gradient_norm'] = 0.0
            else:
                # GRADIENT PROCESSING: Use ML-style normalization instead of naive clipping
                if self.use_gradient_processor and self.gradient_processor is not None:
                    # Process through BatchNorm + tanh
                    raw_norm = gradient.norm().item()
                    gradient = self.gradient_processor.process(gradient)
                    processed_norm = gradient.norm().item()

                    if debug_mol:
                        print(f"  Gradient processed: {raw_norm:.2f} -> {processed_norm:.4f} (BatchNorm+tanh)")
                else:
                    # Fallback: simple clipping
                    grad_norm = gradient.norm().item()
                    if grad_norm > self.max_grad_norm:
                        gradient = gradient * (self.max_grad_norm / grad_norm)
                        if debug_mol:
                            print(f"  Gradient clipped: {grad_norm:.2f} -> {self.max_grad_norm}")

                metrics['gradient_norm'] = gradient.norm().item()

        except Exception as e:
            print(f"  Gradient computation error: {e}")

        self.vina_calls += metrics['n_vina_calls']
        self.total_time += time.time() - start_time
        self.guidance_applied += 1

        return gradient, metrics

    def _coords_to_rdkit(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        debug: bool = False
    ) -> Chem.Mol | None:
        """
        Convert GCDM coordinates to RDKit molecule.

        Uses MST-based bond inference to create a CONNECTED molecule.
        This is critical because Vina can't handle disconnected fragments.
        """
        try:
            coords_np = coords.cpu().detach().numpy()

            # Get atomic numbers from GCDM type indices
            if atom_types.dim() == 2:
                type_indices = atom_types.argmax(dim=-1).cpu().numpy()
            else:
                type_indices = atom_types.cpu().numpy()

            n_atoms = len(type_indices)
            if n_atoms < 3:
                if debug:
                    print(f"  Too few atoms: {n_atoms}")
                return None

            # Create RWMol
            mol = Chem.RWMol()

            # Add atoms
            atomic_nums = []
            for type_idx in type_indices:
                type_idx = int(type_idx) % len(GCDM_ATOM_MAP)
                atomic_num = GCDM_ATOM_MAP.get(type_idx, 6)
                atomic_nums.append(atomic_num)
                mol.AddAtom(Chem.Atom(atomic_num))

            if debug:
                symbols = [ATOMIC_NUM_TO_SYMBOL.get(n, '?') for n in atomic_nums]
                print(f"  Atoms: {symbols}")

            # Add conformer with coordinates
            conf = Chem.Conformer(n_atoms)
            for i in range(n_atoms):
                conf.SetAtomPosition(i, coords_np[i].tolist())
            mol.AddConformer(conf)

            # CRITICAL: Build a minimum spanning tree to ensure connectivity
            # This creates n-1 bonds connecting all atoms
            # Use Prim's algorithm
            used = set([0])
            remaining = set(range(1, n_atoms))
            n_bonds = 0

            while remaining:
                best_dist = float('inf')
                best_pair = None

                for u in used:
                    for v in remaining:
                        dist = np.linalg.norm(coords_np[u] - coords_np[v])
                        if dist < best_dist:
                            best_dist = dist
                            best_pair = (u, v)

                if best_pair is None:
                    break

                # Add bond even if distance is large (ensures connectivity)
                try:
                    mol.AddBond(best_pair[0], best_pair[1], Chem.BondType.SINGLE)
                    n_bonds += 1
                except:
                    pass

                used.add(best_pair[1])
                remaining.discard(best_pair[1])

            if debug:
                print(f"  MST bonds: {n_bonds}")

            # We should have n-1 bonds for n atoms (tree structure)
            if n_bonds < n_atoms - 1:
                if debug:
                    print(f"  Warning: Not fully connected ({n_bonds} < {n_atoms-1})")

            mol = mol.GetMol()

            # Verify connectivity
            frags = Chem.GetMolFrags(mol)
            if len(frags) > 1:
                if debug:
                    print(f"  Warning: {len(frags)} fragments - keeping largest")
                # Keep only largest fragment
                largest = max(frags, key=len)
                mol = Chem.RWMol(mol)
                atoms_to_remove = []
                for frag in frags:
                    if frag != largest:
                        atoms_to_remove.extend(frag)
                # Remove in reverse order to preserve indices
                for idx in sorted(atoms_to_remove, reverse=True):
                    mol.RemoveAtom(idx)
                mol = mol.GetMol()

                if mol.GetNumAtoms() < 3:
                    if debug:
                        print("  Too few atoms after keeping largest fragment")
                    return None

            # Try to sanitize (often fails but we continue)
            try:
                Chem.SanitizeMol(mol)
            except:
                pass

            return mol

        except Exception as e:
            if debug:
                print(f"  Conversion error: {e}")
            return None

    def _compute_vina_score(self, mol: Chem.Mol, debug: bool = False) -> float | None:
        """Run Vina --score_only on molecule."""
        try:
            import uuid

            # DON'T add hydrogens - obabel will do it, and AddHs + Embed
            # would reset the coordinates we carefully set

            # Create temp files
            pdbqt_path = self.cache_dir / f"lig_{uuid.uuid4().hex[:8]}.pdbqt"
            pdb_path = self.cache_dir / f"lig_{uuid.uuid4().hex[:8]}.pdb"

            # Save as PDB (without hydrogens - obabel adds them)
            try:
                Chem.MolToPDBFile(mol, str(pdb_path))
            except Exception as e:
                if debug:
                    print(f"  PDB write failed: {e}")
                return None

            if not pdb_path.exists() or pdb_path.stat().st_size < 50:
                if debug:
                    print("  PDB file invalid")
                pdb_path.unlink(missing_ok=True)
                return None

            # Convert to PDBQT using obabel (adds hydrogens with -xh)
            convert_cmd = ['obabel', str(pdb_path), '-O', str(pdbqt_path), '-h']
            convert_result = subprocess.run(convert_cmd, capture_output=True, text=True, timeout=10)

            if not pdbqt_path.exists() or pdbqt_path.stat().st_size < 100:
                if debug:
                    print(f"  PDBQT conversion failed: {convert_result.stderr}")
                pdb_path.unlink(missing_ok=True)
                return None

            if debug:
                print(f"  PDBQT created: {pdbqt_path.stat().st_size} bytes")

            # Run Vina --score_only
            cmd = [
                self.vina_executable,
                '--receptor', str(self.receptor_pdbqt),
                '--ligand', str(pdbqt_path),
                '--center_x', str(self.binding_site['center_x']),
                '--center_y', str(self.binding_site['center_y']),
                '--center_z', str(self.binding_site['center_z']),
                '--size_x', str(self.binding_site['size_x']),
                '--size_y', str(self.binding_site['size_y']),
                '--size_z', str(self.binding_site['size_z']),
                '--score_only'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            # Cleanup
            pdb_path.unlink(missing_ok=True)
            pdbqt_path.unlink(missing_ok=True)

            if debug:
                print(f"  Vina return code: {result.returncode}")
                if result.returncode != 0:
                    print(f"  Vina STDERR: {result.stderr[:500] if result.stderr else 'empty'}")
                    print(f"  Vina STDOUT (last 300 chars): {result.stdout[-300:] if result.stdout else 'empty'}")

            # Parse score - look for "Estimated Free Energy" or "Affinity"
            for line in result.stdout.split('\n'):
                if 'Estimated Free Energy of Binding' in line:
                    # Format: "Estimated Free Energy of Binding   : -1.425 (kcal/mol)"
                    parts = line.split(':')
                    if len(parts) >= 2:
                        score_part = parts[1].strip().split()[0]
                        try:
                            score = float(score_part)
                            if debug:
                                print(f"  Vina score: {score}")
                            return score
                        except ValueError:
                            pass
                elif 'Affinity:' in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == 'Affinity:' and i+1 < len(parts):
                            try:
                                score = float(parts[i+1])
                                if debug:
                                    print(f"  Vina score: {score}")
                                return score
                            except ValueError:
                                pass

            if debug:
                print("  Could not parse Vina output")
                # Show relevant part of output
                for line in result.stdout.split('\n'):
                    if 'Energy' in line or 'Affinity' in line:
                        print(f"    {line}")

            return None

        except Exception as e:
            if debug:
                print(f"  Vina scoring error: {e}")
            return None

    def _distance_guidance(
        self,
        coords: torch.Tensor,
        pocket_center: torch.Tensor,
        ligand_mask: torch.Tensor
    ) -> torch.Tensor:
        """Fallback: guide molecule toward pocket center."""
        gradient = torch.zeros_like(coords)

        valid_mask = ligand_mask.bool()
        valid_coords = coords[valid_mask]

        mol_center = valid_coords.mean(dim=0)
        # Gradient must point away from pocket (sign convention: subtract gradient → move toward pocket)
        direction = mol_center - pocket_center
        direction = direction / (direction.norm() + 1e-6)

        # Apply same direction to all atoms (weak guidance)
        gradient[valid_mask] = direction.unsqueeze(0).expand(valid_coords.shape[0], -1) * 0.1

        return gradient

    def get_adaptive_scale(self, timestep: int, base_scale: float = 1.0) -> float:
        return get_adaptive_scale(timestep) * base_scale

    def get_stats(self) -> dict:
        """Get performance statistics."""
        stats = {
            'n_vina_calls': self.vina_calls,
            'total_time': self.total_time,
            'n_guidance_applied': self.guidance_applied,
            'avg_time_per_call': self.total_time / max(self.vina_calls, 1)
        }

        # Add gradient processor statistics if available
        if self.use_gradient_processor and self.gradient_processor is not None:
            proc_stats = self.gradient_processor.get_stats()
            stats['gradient_processor'] = proc_stats
            stats['avg_raw_grad_norm'] = proc_stats['avg_raw_norm']
            stats['avg_processed_grad_norm'] = proc_stats['avg_processed_norm']

        return stats
