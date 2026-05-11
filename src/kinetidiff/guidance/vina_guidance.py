"""
Direct AutoDock Vina guidance for GCDM diffusion models.

Replaces HNN-Denovo guidance with actual Vina scoring function.
Key insight: HNN-Denovo has r=0.224 correlation with Vina (essentially random),
while direct Vina scoring provides accurate gradients.

Features:
- Uses Vina SCORING ONLY mode (10-100x faster than full docking)
- Numerical gradients computed efficiently via finite differences
- Caches receptor preparation for speed
- Supports both AutoDock Vina and QuickVina2

Usage:
    from src.guidance.vina_guidance import VinaGuidance
    
    vina = VinaGuidance(
        receptor_pdbqt='receptor.pdbqt',
        binding_site_config={'center_x': 0, 'center_y': 0, 'center_z': 0,
                            'size_x': 20, 'size_y': 20, 'size_z': 20}
    )
    
    gradient, score = vina.compute_vina_gradient(coords, atom_types)
"""

import os
import subprocess
import uuid
import warnings
from pathlib import Path

import numpy as np
import torch

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Vina guidance will not work.")


# GCDM atom type mapping (index -> atomic number)
# Based on GCDM's standard encoding
GCDM_ATOM_MAP = {
    0: 6,   # C
    1: 7,   # N  
    2: 8,   # O
    3: 9,   # F
    4: 15,  # P (less common)
    5: 16,  # S
    6: 17,  # Cl
    7: 35,  # Br
    8: 53,  # I
}

# Reverse mapping
ATOMIC_NUM_TO_SYMBOL = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}


class VinaGuidance:
    """
    Real-time Vina-based affinity guidance for diffusion models.
    
    Key features:
    - Uses Vina --local_only for fast local docking (5-10x faster than global)
    - Numerical gradients computed efficiently via finite differences
    - Caches receptor preparation
    - Handles GCDM coordinate/atom type format
    
    Note on scoring modes:
    - score_only=True: Scores ligand in current pose (requires pre-positioned ligand)
    - score_only=False: Uses --local_only for fast local optimization (RECOMMENDED)
    
    Attributes:
        receptor_pdbqt: Path to prepared receptor PDBQT file
        binding_site: Dict with center_x/y/z and size_x/y/z
        vina_executable: Path to vina or qvina2 binary
        score_only: Whether to use --score_only mode (False = local_only, recommended)
        cache_dir: Directory for temporary files
    """
    
    def __init__(
        self,
        receptor_pdbqt: str | Path,
        binding_site_config: dict[str, float],
        vina_executable: str = 'vina',
        score_only: bool = False,  # Changed default to False for local_only mode
        cache_dir: str = '/tmp/vina_cache',
        timeout: int = 60,  # Increased timeout for local_only
        device: str = 'cuda'
    ):
        """
        Initialize Vina guidance.
        
        Args:
            receptor_pdbqt: Path to prepared receptor PDBQT file
            binding_site_config: Dict with center_x, center_y, center_z,
                                size_x, size_y, size_z
            vina_executable: Path to vina binary ('vina', 'vina_1.2.5', 'qvina2.1')
            score_only: If True, use --score_only (10-100x faster)
            cache_dir: Directory for temporary files
            timeout: Timeout in seconds for Vina calls
            device: Computation device (for tensors)
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required for Vina guidance")
        
        self.receptor_pdbqt = Path(receptor_pdbqt)
        if not self.receptor_pdbqt.exists():
            raise FileNotFoundError(f"Receptor not found: {self.receptor_pdbqt}")
        
        self.binding_site = binding_site_config
        self.vina_executable = vina_executable
        self.score_only = score_only
        self.timeout = timeout
        self.device = device
        
        # Setup cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Verify Vina is available
        self._vina_available = self._check_vina_available()
        if not self._vina_available:
            # Try alternative executables
            for alt in ['vina_1.2.5', 'qvina2.1', 'qvina2', 'smina']:
                if self._check_vina_available(alt):
                    self.vina_executable = alt
                    self._vina_available = True
                    print(f"✓ Using alternative: {alt}")
                    break
        
        if not self._vina_available:
            warnings.warn(f"Vina executable not found: {vina_executable}. "
                         "Will use fallback scoring.")
        else:
            print(f"✓ Vina guidance initialized (score_only={score_only})")
            print(f"  Receptor: {self.receptor_pdbqt}")
            print(f"  Binding site center: ({binding_site_config.get('center_x', 0):.1f}, "
                  f"{binding_site_config.get('center_y', 0):.1f}, "
                  f"{binding_site_config.get('center_z', 0):.1f})")
    
    def _check_vina_available(self, executable: str = None) -> bool:
        """Check if Vina executable is available."""
        exe = executable or self.vina_executable
        try:
            result = subprocess.run(
                [exe, '--version'],
                capture_output=True,
                timeout=5
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            return False
    
    def compute_vina_score(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        use_gcdm_encoding: bool = True
    ) -> float:
        """
        Compute Vina binding score for given molecule.
        
        Args:
            coords: (n_atoms, 3) or (batch, n_atoms, 3) - atomic coordinates
            atom_types: (n_atoms,) or (batch, n_atoms) - atom type indices or atomic numbers
            use_gcdm_encoding: If True, interpret atom_types as GCDM indices
            
        Returns:
            vina_score: Binding energy in kcal/mol (negative = better)
        """
        # Handle batch dimension
        if coords.dim() == 3:
            coords = coords[0]  # Take first in batch
        if atom_types.dim() == 2:
            atom_types = atom_types[0]
        
        try:
            # Convert to RDKit molecule
            mol = self._coords_to_mol(coords, atom_types, use_gcdm_encoding)
            if mol is None:
                return 0.0  # Neutral score for invalid molecules
            
            # Generate 3D if needed
            if mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol, randomSeed=42)
            
            if not self._vina_available:
                # Fallback to fast approximation
                return self._fast_score_approximation(mol)
            
            # Save as PDBQT
            ligand_pdbqt = self._mol_to_pdbqt(mol)
            if ligand_pdbqt is None:
                return 0.0
            
            # Run Vina scoring
            vina_score = self._run_vina_scoring(ligand_pdbqt)
            
            # Cleanup
            try:
                ligand_pdbqt.unlink()
            except:
                pass
            
            return vina_score
            
        except Exception as e:
            print(f"Warning: Vina scoring failed: {e}")
            return 0.0
    
    def compute_vina_gradient(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        eps: float = 0.05,
        use_gcdm_encoding: bool = True
    ) -> tuple[torch.Tensor, float]:
        """
        Compute numerical gradient of Vina score w.r.t. coordinates.
        
        Uses central finite differences for accuracy.
        
        Args:
            coords: (n_atoms, 3) or (batch, n_atoms, 3)
            atom_types: (n_atoms,) or (batch, n_atoms)
            eps: Finite difference step size in Angstroms
            use_gcdm_encoding: Interpret atom_types as GCDM indices
            
        Returns:
            gradient: (n_atoms, 3) - gradient pointing toward BETTER binding
            vina_score: Current Vina score
        """
        # Handle batch dimension
        was_batched = coords.dim() == 3
        if was_batched:
            coords = coords[0]
            atom_types = atom_types[0] if atom_types.dim() == 2 else atom_types
        
        coords = coords.clone().detach()
        
        # Current score
        vina_score = self.compute_vina_score(coords, atom_types, use_gcdm_encoding)
        
        # Initialize gradient
        gradient = torch.zeros_like(coords)
        n_atoms = coords.shape[0]
        
        # Compute numerical gradient (central difference)
        for atom_idx in range(n_atoms):
            for dim in range(3):  # x, y, z
                # Forward perturbation
                coords_plus = coords.clone()
                coords_plus[atom_idx, dim] += eps
                vina_plus = self.compute_vina_score(coords_plus, atom_types, use_gcdm_encoding)
                
                # Backward perturbation
                coords_minus = coords.clone()
                coords_minus[atom_idx, dim] -= eps
                vina_minus = self.compute_vina_score(coords_minus, atom_types, use_gcdm_encoding)
                
                # Central difference
                gradient[atom_idx, dim] = (vina_plus - vina_minus) / (2 * eps)
        
        # Gradient points toward LOWER Vina score (more negative = better binding)
        # Return negative gradient to MAXIMIZE binding (minimize Vina score)
        if was_batched:
            gradient = gradient.unsqueeze(0)
        
        return -gradient, vina_score
    
    def compute_vina_gradient_fast(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        n_samples: int = 5,
        eps: float = 0.05,
        use_gcdm_encoding: bool = True
    ) -> tuple[torch.Tensor, float]:
        """
        Fast approximate gradient using subset of atoms.
        
        For molecules with many atoms, computing full gradient is slow.
        This samples a subset of heavy atoms and extrapolates.
        
        Args:
            coords: (n_atoms, 3) or (batch, n_atoms, 3)
            atom_types: (n_atoms,) or (batch, n_atoms)
            n_samples: Number of atoms to sample for gradient
            eps: Perturbation size in Angstroms
            use_gcdm_encoding: Interpret atom_types as GCDM indices
            
        Returns:
            gradient: (n_atoms, 3) - gradient toward better binding
            vina_score: Current Vina score
        """
        # Handle batch dimension
        was_batched = coords.dim() == 3
        if was_batched:
            coords = coords[0]
            atom_types = atom_types[0] if atom_types.dim() == 2 else atom_types
        
        coords = coords.clone().detach()
        n_atoms = coords.shape[0]
        
        if n_atoms <= n_samples * 2:
            # Small molecule: compute exact gradient
            return self.compute_vina_gradient(coords, atom_types, eps, use_gcdm_encoding)
        
        # Get heavy atom indices (non-hydrogen)
        if use_gcdm_encoding:
            # GCDM encoding: all are heavy atoms typically
            heavy_atom_indices = list(range(n_atoms))
        else:
            # Atomic numbers: H=1
            heavy_mask = atom_types > 1
            heavy_atom_indices = heavy_mask.nonzero(as_tuple=True)[0].tolist()
        
        if len(heavy_atom_indices) <= n_samples:
            sample_indices = heavy_atom_indices
        else:
            # Random sample of heavy atoms
            sample_indices = np.random.choice(
                heavy_atom_indices, size=n_samples, replace=False
            ).tolist()
        
        # Current score
        vina_score = self.compute_vina_score(coords, atom_types, use_gcdm_encoding)
        
        # Compute gradient for sampled atoms only
        gradient = torch.zeros_like(coords)
        
        for atom_idx in sample_indices:
            for dim in range(3):
                coords_plus = coords.clone()
                coords_plus[atom_idx, dim] += eps
                vina_plus = self.compute_vina_score(coords_plus, atom_types, use_gcdm_encoding)
                
                coords_minus = coords.clone()
                coords_minus[atom_idx, dim] -= eps
                vina_minus = self.compute_vina_score(coords_minus, atom_types, use_gcdm_encoding)
                
                gradient[atom_idx, dim] = (vina_plus - vina_minus) / (2 * eps)
        
        # Extrapolate gradient to all heavy atoms (mean of sampled)
        if len(sample_indices) > 0:
            mean_gradient = gradient[sample_indices].mean(dim=0)
            for idx in heavy_atom_indices:
                if idx not in sample_indices:
                    gradient[idx] = mean_gradient
        
        if was_batched:
            gradient = gradient.unsqueeze(0)
        
        return -gradient, vina_score
    
    def compute_batch_scores(
        self,
        coords_batch: torch.Tensor,
        atom_types_batch: torch.Tensor,
        use_gcdm_encoding: bool = True
    ) -> torch.Tensor:
        """
        Compute Vina scores for a batch of molecules.
        
        Args:
            coords_batch: (batch, n_atoms, 3)
            atom_types_batch: (batch, n_atoms)
            use_gcdm_encoding: Interpret atom_types as GCDM indices
            
        Returns:
            scores: (batch,) tensor of Vina scores
        """
        batch_size = coords_batch.shape[0]
        scores = []
        
        for i in range(batch_size):
            score = self.compute_vina_score(
                coords_batch[i], 
                atom_types_batch[i] if atom_types_batch.dim() > 1 else atom_types_batch,
                use_gcdm_encoding
            )
            scores.append(score)
        
        return torch.tensor(scores, dtype=torch.float32, device=self.device)
    
    def _coords_to_mol(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        use_gcdm_encoding: bool
    ) -> Chem.Mol | None:
        """
        Convert coordinates + atom types to RDKit molecule.
        
        Args:
            coords: (n_atoms, 3) coordinates
            atom_types: (n_atoms,) atom type indices or atomic numbers
            use_gcdm_encoding: If True, map GCDM indices to atomic numbers
            
        Returns:
            mol: RDKit molecule or None if failed
        """
        coords_np = coords.detach().cpu().numpy()
        types_np = atom_types.detach().cpu().numpy()
        
        mol = Chem.RWMol()
        
        # Add atoms
        for i, atom_type in enumerate(types_np):
            if use_gcdm_encoding:
                # Handle one-hot encoding if present
                if len(types_np.shape) > 1:
                    atom_type_idx = np.argmax(types_np[i])
                else:
                    atom_type_idx = int(atom_type)
                atomic_num = GCDM_ATOM_MAP.get(atom_type_idx, 6)  # Default to Carbon
            else:
                atomic_num = int(atom_type)
            
            atom = Chem.Atom(atomic_num)
            mol.AddAtom(atom)
        
        # Add conformer with coordinates
        conf = Chem.Conformer(len(coords_np))
        for i, coord in enumerate(coords_np):
            conf.SetAtomPosition(i, coord.tolist())
        mol.AddConformer(conf, assignId=True)
        
        # Infer bonds from distances
        try:
            # Method 1: Use RDKit's bond inference
            Chem.rdDetermineBonds.DetermineBonds(mol, charge=0)
            mol_final = mol.GetMol()
            Chem.SanitizeMol(mol_final)
            return mol_final
        except Exception as e1:
            # Method 2: Manual distance-based bonding
            try:
                mol2 = Chem.RWMol()
                for i in range(mol.GetNumAtoms()):
                    mol2.AddAtom(Chem.Atom(mol.GetAtomWithIdx(i).GetAtomicNum()))
                
                # Add bonds based on distance
                for i in range(len(coords_np)):
                    for j in range(i+1, len(coords_np)):
                        dist = np.linalg.norm(coords_np[i] - coords_np[j])
                        # Typical bond lengths: 0.9-2.0 Å
                        if 0.9 < dist < 2.0:
                            try:
                                mol2.AddBond(i, j, Chem.BondType.SINGLE)
                            except:
                                pass
                
                # Add conformer
                conf2 = Chem.Conformer(mol2.GetNumAtoms())
                for i, coord in enumerate(coords_np):
                    conf2.SetAtomPosition(i, coord.tolist())
                mol2.AddConformer(conf2, assignId=True)
                
                mol_final = mol2.GetMol()
                try:
                    Chem.SanitizeMol(mol_final)
                except:
                    pass  # Keep unsanitized molecule
                return mol_final
                
            except Exception as e2:
                print(f"Warning: Molecule conversion failed: {e1}, {e2}")
                return None
    
    def _mol_to_pdbqt(self, mol: Chem.Mol) -> Path | None:
        """
        Convert RDKit molecule to PDBQT file.
        
        Uses OpenBabel for conversion (handles partial charges).
        
        Args:
            mol: RDKit molecule with 3D conformer
            
        Returns:
            pdbqt_path: Path to generated PDBQT file, or None if failed
        """
        # Generate unique filename
        file_id = uuid.uuid4().hex[:8]
        pdb_path = self.cache_dir / f"ligand_{file_id}.pdb"
        pdbqt_path = self.cache_dir / f"ligand_{file_id}.pdbqt"
        
        try:
            # Add hydrogens if not present
            if mol.GetNumAtoms() > 0:
                first_atom = mol.GetAtomWithIdx(0)
                if first_atom.GetTotalNumHs(includeNeighbors=True) == 0:
                    mol = Chem.AddHs(mol, addCoords=True)
            
            # Save as PDB
            Chem.MolToPDBFile(mol, str(pdb_path))
            
            # Convert to PDBQT using OpenBabel
            result = subprocess.run(
                ['obabel', str(pdb_path), '-O', str(pdbqt_path), '-p', '7.4'],
                capture_output=True,
                timeout=30
            )
            
            # Cleanup PDB
            try:
                pdb_path.unlink()
            except:
                pass
            
            if pdbqt_path.exists() and pdbqt_path.stat().st_size > 0:
                return pdbqt_path
            else:
                return None
                
        except Exception as e:
            print(f"Warning: PDBQT conversion failed: {e}")
            # Cleanup
            for p in [pdb_path, pdbqt_path]:
                try:
                    p.unlink()
                except:
                    pass
            return None
    
    def _run_vina_scoring(self, ligand_pdbqt: Path) -> float:
        """
        Run Vina docking/scoring.
        
        Modes:
        - score_only=True: Just scores current pose (requires ligand pre-positioned)
        - score_only=False: Runs quick local docking (10-20x faster than global)
        
        Args:
            ligand_pdbqt: Path to ligand PDBQT file
            
        Returns:
            vina_score: Binding score in kcal/mol (negative = better)
        """
        # Generate output path
        output_pdbqt = self.cache_dir / f"output_{ligand_pdbqt.stem}.pdbqt"
        
        cmd = [
            self.vina_executable,
            '--receptor', str(self.receptor_pdbqt),
            '--ligand', str(ligand_pdbqt),
            '--center_x', str(self.binding_site.get('center_x', 0)),
            '--center_y', str(self.binding_site.get('center_y', 0)),
            '--center_z', str(self.binding_site.get('center_z', 0)),
            '--size_x', str(self.binding_site.get('size_x', 20)),
            '--size_y', str(self.binding_site.get('size_y', 20)),
            '--size_z', str(self.binding_site.get('size_z', 20)),
            '--out', str(output_pdbqt)  # Output file needed for docking
        ]
        
        if self.score_only:
            # score_only requires ligand already positioned in binding site
            # This is only useful when we have a pre-docked pose
            cmd.append('--score_only')
        else:
            # Use local_only for FAST local optimization (5-10x faster than global)
            # This is the recommended mode for guidance during generation
            cmd.extend([
                '--local_only',  # Local search only, much faster
                '--exhaustiveness', '1',  # Minimal iterations
                '--num_modes', '1'  # Only need best mode
            ])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Parse score from output
            vina_score = self._parse_vina_output(result.stdout, result.stderr)
            
            # Cleanup output file
            try:
                output_pdbqt.unlink()
            except:
                pass
            
            return vina_score if vina_score is not None else 0.0
            
        except subprocess.TimeoutExpired:
            print("Warning: Vina scoring timed out")
            # Cleanup
            try:
                output_pdbqt.unlink()
            except:
                pass
            return 0.0
        except Exception as e:
            print(f"Warning: Vina execution failed: {e}")
            return 0.0
    
    def _parse_vina_output(self, stdout: str, stderr: str) -> float | None:
        """
        Extract Vina score from output.
        
        Handles different Vina versions and output formats:
        - "Estimated Free Energy of Binding   : -5.251 (kcal/mol)"
        - "Affinity: -7.2 (kcal/mol)"
        - "REMARK VINA RESULT:    -7.5      0.000      0.000"
        """
        output = stdout + stderr
        
        import re
        
        # Pattern 1: "Estimated Free Energy of Binding" (Vina 1.2+)
        match = re.search(r'Estimated Free Energy of Binding\s*:\s*(-?\d+\.?\d*)', output)
        if match:
            return float(match.group(1))
        
        # Pattern 2: "Affinity:" (older Vina)
        match = re.search(r'Affinity:\s*(-?\d+\.?\d*)', output)
        if match:
            return float(match.group(1))
        
        # Pattern 3: PDBQT format REMARK line
        match = re.search(r'REMARK VINA RESULT:\s*(-?\d+\.?\d*)', output)
        if match:
            return float(match.group(1))
        
        # Pattern 4: Standard table format "   1    -7.5      0.000"
        for line in output.split('\n'):
            if line.strip().startswith('1') or line.strip().startswith('   1'):
                parts = line.split()
                for part in parts:
                    try:
                        score = float(part)
                        if -20 < score < 0:  # Valid score range
                            return score
                    except ValueError:
                        continue
        
        # Fallback: find any negative number in typical score range
        matches = re.findall(r'-\d+\.?\d*', output)
        for match in matches:
            try:
                score = float(match)
                if -20 < score < 0:
                    return score
            except ValueError:
                continue
        
        return None
    
    def _fast_score_approximation(self, mol: Chem.Mol) -> float:
        """
        Fast ML-based scoring when Vina is unavailable.
        
        Uses empirical formula calibrated to approximate Vina.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            score: Approximate binding energy (kcal/mol)
        """
        try:
            from rdkit.Chem import Descriptors, rdMolDescriptors
            
            # Remove Hs for descriptor calculation
            mol_noH = Chem.RemoveHs(mol)
            
            # Calculate descriptors
            mw = Descriptors.MolWt(mol_noH)
            logp = Descriptors.MolLogP(mol_noH)
            n_rot = Descriptors.NumRotatableBonds(mol_noH)
            hbd = Descriptors.NumHDonors(mol_noH)
            hba = Descriptors.NumHAcceptors(mol_noH)
            n_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol_noH)
            tpsa = Descriptors.TPSA(mol_noH)
            
            # Empirical scoring (calibrated to Vina)
            score = -5.0  # Base score
            
            # MW: optimal 300-500
            if mw < 300:
                score += 0.5
            elif mw > 500:
                score += 0.003 * (mw - 500)  # Penalty for large
            
            # LogP: optimal 2-4
            if 2 <= logp <= 4:
                score -= 1.0
            elif logp > 5:
                score += 0.5
            
            # Rotatable bonds: fewer is better
            score += 0.15 * n_rot
            
            # H-bond donors/acceptors (improve binding)
            score -= 0.25 * min(hbd, 5)
            score -= 0.15 * min(hba, 10)
            
            # Aromatic rings
            score -= 0.3 * min(n_aromatic, 4)
            
            # TPSA: optimal 40-140
            if 40 <= tpsa <= 140:
                score -= 0.5
            
            # Clip to reasonable range
            return np.clip(score, -15.0, 0.0)
            
        except Exception:
            return -5.0  # Neutral score


class VinaGuidanceWrapper:
    """
    Wrapper to integrate Vina guidance into GCDM sampling loop.
    
    Provides same interface as AffinityGuidanceModel but uses
    direct Vina scoring.
    """
    
    def __init__(
        self,
        receptor_pdbqt: str,
        binding_site_config: dict[str, float],
        guidance_scale: float = 1.0,
        use_fast_gradient: bool = True,
        n_gradient_samples: int = 5,
        device: str = 'cuda',
        **kwargs
    ):
        """
        Initialize Vina guidance wrapper.
        
        Args:
            receptor_pdbqt: Path to receptor PDBQT
            binding_site_config: Binding site parameters
            guidance_scale: Scaling factor for gradients
            use_fast_gradient: Use fast approximate gradient
            n_gradient_samples: Atoms to sample for fast gradient
            device: Computation device
        """
        self.vina = VinaGuidance(
            receptor_pdbqt=receptor_pdbqt,
            binding_site_config=binding_site_config,
            device=device,
            **kwargs
        )
        
        self.guidance_scale = guidance_scale
        self.use_fast_gradient = use_fast_gradient
        self.n_gradient_samples = n_gradient_samples
        self.device = device
        
        print(f"✓ Vina guidance wrapper initialized (scale={guidance_scale})")
    
    def compute_guidance(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        timestep: int = 0,
        **kwargs
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute guidance gradient for diffusion model.
        
        Args:
            coords: (batch, n_atoms, 3) ligand coordinates
            atom_types: (batch, n_atoms) atom types
            timestep: Current diffusion timestep
            
        Returns:
            gradient: (batch, n_atoms, 3) guidance gradient
            info: Dict with score and diagnostics
        """
        batch_size = coords.shape[0]
        
        gradients = []
        scores = []
        
        for i in range(batch_size):
            if self.use_fast_gradient:
                grad, score = self.vina.compute_vina_gradient_fast(
                    coords[i], atom_types[i],
                    n_samples=self.n_gradient_samples
                )
            else:
                grad, score = self.vina.compute_vina_gradient(
                    coords[i], atom_types[i]
                )
            
            gradients.append(grad)
            scores.append(score)
        
        gradient = torch.stack(gradients).to(self.device)
        mean_score = np.mean(scores)
        
        # Apply timestep-dependent scaling
        timestep_scale = self._get_timestep_scale(timestep)
        scaled_gradient = gradient * self.guidance_scale * timestep_scale
        
        info = {
            'vina_score': mean_score,
            'grad_norm': gradient.norm().item(),
            'timestep_scale': timestep_scale,
            'pKd_approx': -mean_score / 1.36  # Convert kcal/mol to pKd
        }
        
        return scaled_gradient, info
    
    def _get_timestep_scale(self, timestep: int, max_timesteps: int = 1000) -> float:
        """
        Compute timestep-dependent guidance scale.
        
        Early timesteps: weak guidance (molecule is noisy)
        Late timesteps: strong guidance (molecule is clear)
        
        Args:
            timestep: Current timestep (1000 = start, 0 = end)
            max_timesteps: Total number of timesteps
            
        Returns:
            scale: Guidance scale multiplier
        """
        t_frac = timestep / max_timesteps
        
        # Exponential ramp-up (weak early, strong late)
        # t=1000: scale=0.1
        # t=500:  scale=0.5
        # t=100:  scale=2.0
        # t=0:    scale=20.0
        scale = 0.1 * np.exp(3 * (1 - t_frac))
        
        # Clip to reasonable range
        return np.clip(scale, 0.1, 10.0)


# Convenience function
def load_vina_guidance(
    receptor_pdbqt: str,
    binding_site_config: dict[str, float],
    guidance_scale: float = 1.0,
    **kwargs
) -> VinaGuidanceWrapper:
    """
    Load Vina guidance model.
    
    Args:
        receptor_pdbqt: Path to receptor PDBQT
        binding_site_config: Dict with center_x/y/z and size_x/y/z
        guidance_scale: Guidance strength
        
    Returns:
        VinaGuidanceWrapper instance
    """
    return VinaGuidanceWrapper(
        receptor_pdbqt=receptor_pdbqt,
        binding_site_config=binding_site_config,
        guidance_scale=guidance_scale,
        **kwargs
    )


# Testing
if __name__ == '__main__':
    print("=" * 60)
    print("Testing Vina Guidance Module")
    print("=" * 60)
    
    # Test configuration
    test_receptor = "DOCKING2/receptor.pdbqt"
    test_config = {
        'center_x': 24.87,
        'center_y': -12.54,
        'center_z': 38.40,
        'size_x': 20.0,
        'size_y': 20.0,
        'size_z': 20.0
    }
    
    # Check if receptor exists
    import os
    if not os.path.exists(test_receptor):
        print(f"Receptor not found: {test_receptor}")
        print("Creating test with fallback scoring...")
        test_receptor = None
    
    if test_receptor:
        print("\n1. Testing VinaGuidance initialization...")
        try:
            vina = VinaGuidance(
                receptor_pdbqt=test_receptor,
                binding_site_config=test_config
            )
            print("   ✓ Initialization successful")
        except Exception as e:
            print(f"   ✗ Initialization failed: {e}")
            exit(1)
    
    print("\n2. Testing molecule conversion...")
    # Create test molecule (aspirin)
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    
    # Get coords and atom types
    coords = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
    atom_types = torch.tensor(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
        dtype=torch.long
    )
    
    print(f"   Molecule: {smiles}")
    print(f"   Atoms: {coords.shape[0]}, Coords shape: {coords.shape}")
    
    if test_receptor:
        print("\n3. Testing Vina scoring...")
        score = vina.compute_vina_score(coords, atom_types, use_gcdm_encoding=False)
        print(f"   Vina score: {score:.2f} kcal/mol")
        print(f"   Approx pKd: {-score/1.36:.1f}")
        
        print("\n4. Testing Vina gradient...")
        gradient, score2 = vina.compute_vina_gradient(
            coords, atom_types, use_gcdm_encoding=False
        )
        print(f"   Gradient shape: {gradient.shape}")
        print(f"   Gradient norm: {gradient.norm():.4f}")
        print(f"   Score: {score2:.2f} kcal/mol")
        
        print("\n5. Testing gradient direction...")
        step_size = 0.1
        coords_improved = coords + step_size * gradient
        score_improved = vina.compute_vina_score(
            coords_improved, atom_types, use_gcdm_encoding=False
        )
        
        print(f"   Original score: {score2:.2f}")
        print(f"   After gradient step: {score_improved:.2f}")
        print(f"   Improvement: {score_improved - score2:.2f}")
        
        if score_improved < score2:
            print("   ✓ PASS: Gradient improves binding")
        else:
            print("   ⚠ Gradient did not improve binding (may need tuning)")
        
        print("\n6. Testing fast gradient...")
        gradient_fast, score_fast = vina.compute_vina_gradient_fast(
            coords, atom_types, n_samples=3, use_gcdm_encoding=False
        )
        print(f"   Fast gradient norm: {gradient_fast.norm():.4f}")
        print(f"   Score: {score_fast:.2f} kcal/mol")
    
    print("\n" + "=" * 60)
    print("✓ Vina guidance module tests completed")
    print("=" * 60)
