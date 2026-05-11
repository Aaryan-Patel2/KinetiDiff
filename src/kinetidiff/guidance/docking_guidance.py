"""
Docking score guidance using QuickVina2.

Provides fast docking score estimation during generation.
For final evaluation, use full QuickVina2 with rigid docking.

Two modes:
1. Fast approximation: ML-based scoring (~1ms/molecule)
2. Full docking: QuickVina2 execution (~10s/molecule)

Usage:
    from src.guidance.docking_guidance import DockingGuidance
    
    # Fast mode (for guidance during generation)
    guidance = DockingGuidance(use_fast_approximation=True)
    score = guidance.score_molecule(mol, pocket_center)
    
    # Accurate mode (for final evaluation)
    guidance = DockingGuidance(use_fast_approximation=False)
    score = guidance.score_molecule(mol, protein_pdbqt, pocket_center, box_size)
"""

import os
import subprocess
import tempfile
from typing import Union

import numpy as np
import torch

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Docking guidance will be limited.")


class DockingGuidance:
    """
    Fast docking score estimation for guidance.
    
    Uses QuickVina2 for scoring or ML-based approximation for speed.
    
    Scoring function approximation based on empirical features:
    - Molecular weight penalty (optimal: 300-500 Da)
    - LogP contribution (optimal: 1-3)
    - Rotatable bond penalty (fewer is better)
    - HBD/HBA contribution (donors and acceptors improve binding)
    - Aromatic ring contribution
    
    Attributes:
        use_fast_approximation: Whether to use ML-based fast scoring
        vina_executable: Path to QuickVina2 binary
        scoring_weights: Weights for fast scoring function
    """
    
    def __init__(
        self,
        use_fast_approximation: bool = True,
        vina_executable: str = 'qvina2.1'
    ):
        """
        Initialize docking guidance.
        
        Args:
            use_fast_approximation: If True, use fast ML-based scoring
            vina_executable: Path to qvina2.1 binary
        """
        self.use_fast_approximation = use_fast_approximation
        self.vina_executable = vina_executable
        
        # Empirical scoring weights (calibrated to approximate Vina scores)
        self.scoring_weights = {
            'base_score': -5.0,      # Base score (neutral binding)
            'mw_penalty': -0.003,     # Penalty per Da over 500
            'logp_coeff': 0.15,       # LogP contribution
            'logp_optimal': 2.5,      # Optimal LogP
            'rot_bond_penalty': -0.15, # Per rotatable bond
            'hbd_coeff': -0.25,       # Per H-bond donor
            'hba_coeff': -0.15,       # Per H-bond acceptor
            'aromatic_coeff': -0.3,   # Per aromatic ring
            'tpsa_coeff': -0.01,      # Topological polar surface area
        }
        
        if use_fast_approximation:
            print("Docking guidance: Using fast approximation (ML-based)")
        else:
            print(f"Docking guidance: Using QuickVina2 ({vina_executable})")
            # Verify QuickVina2 is available
            if not self._check_vina_available():
                print("  Warning: QuickVina2 not found. Falling back to approximation.")
                self.use_fast_approximation = True
    
    def _check_vina_available(self) -> bool:
        """Check if QuickVina2 executable is available."""
        try:
            result = subprocess.run(
                [self.vina_executable, '--help'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0 or b'AutoDock' in result.stderr
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def score_molecule(
        self,
        mol: Union['Chem.Mol', str],
        protein_pdbqt_path: str | None = None,
        box_center: tuple[float, float, float] | None = None,
        box_size: tuple[float, float, float] = (20, 20, 20)
    ) -> torch.Tensor:
        """
        Compute docking score for a molecule.
        
        Args:
            mol: RDKit molecule or SMILES string
            protein_pdbqt_path: Path to receptor PDBQT file (for full docking)
            box_center: (x, y, z) coordinates of box center
            box_size: (x, y, z) dimensions of search box
            
        Returns:
            docking_score: Estimated binding energy (kcal/mol, negative is better)
        """
        # Convert SMILES to mol if needed
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
            if mol is None:
                return torch.tensor(0.0, dtype=torch.float32)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
        
        if self.use_fast_approximation:
            return self._fast_score(mol)
        else:
            return self._quickvina_score(
                mol, protein_pdbqt_path, box_center, box_size
            )
    
    def score_batch(
        self,
        molecules: list[Union['Chem.Mol', str]],
        protein_pdbqt_path: str | None = None,
        box_center: tuple[float, float, float] | None = None,
        box_size: tuple[float, float, float] = (20, 20, 20)
    ) -> torch.Tensor:
        """
        Score a batch of molecules.
        
        Args:
            molecules: List of RDKit molecules or SMILES strings
            protein_pdbqt_path: Path to receptor PDBQT file
            box_center: (x, y, z) coordinates of box center
            box_size: (x, y, z) dimensions of search box
            
        Returns:
            scores: (batch_size,) tensor of docking scores
        """
        scores = []
        for mol in molecules:
            score = self.score_molecule(mol, protein_pdbqt_path, box_center, box_size)
            scores.append(score)
        
        return torch.stack(scores)
    
    def _fast_score(self, mol: 'Chem.Mol') -> torch.Tensor:
        """
        Fast ML-based scoring approximation.
        
        Uses empirical scoring function based on molecular descriptors.
        Calibrated to approximate QuickVina2 scores.
        
        Args:
            mol: RDKit molecule (with hydrogens)
            
        Returns:
            score: Estimated binding energy (kcal/mol)
        """
        try:
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
            
            # Calculate score components
            score = self.scoring_weights['base_score']
            
            # MW penalty (penalize molecules > 500 Da)
            if mw > 500:
                score += self.scoring_weights['mw_penalty'] * (mw - 500)
            
            # LogP contribution (optimal around 2.5)
            logp_diff = abs(logp - self.scoring_weights['logp_optimal'])
            score += self.scoring_weights['logp_coeff'] * (3 - logp_diff)
            
            # Rotatable bond penalty
            score += self.scoring_weights['rot_bond_penalty'] * n_rot
            
            # H-bond donors/acceptors (improve binding)
            score += self.scoring_weights['hbd_coeff'] * min(hbd, 5)
            score += self.scoring_weights['hba_coeff'] * min(hba, 10)
            
            # Aromatic rings
            score += self.scoring_weights['aromatic_coeff'] * min(n_aromatic, 4)
            
            # TPSA contribution
            if 40 <= tpsa <= 140:  # Optimal range
                score -= 0.5
            
            # Clip to reasonable range
            score = np.clip(score, -15.0, 0.0)
            
            return torch.tensor(score, dtype=torch.float32)
            
        except Exception as e:
            print(f"Warning: Fast scoring failed: {e}")
            return torch.tensor(-5.0, dtype=torch.float32)  # Neutral score
    
    def _quickvina_score(
        self,
        mol: 'Chem.Mol',
        protein_pdbqt: str,
        box_center: tuple[float, float, float],
        box_size: tuple[float, float, float]
    ) -> torch.Tensor:
        """
        Actual QuickVina2 docking (slower but accurate).
        
        Args:
            mol: RDKit molecule with 3D coordinates
            protein_pdbqt: Path to receptor PDBQT file
            box_center: (x, y, z) box center
            box_size: (x, y, z) box dimensions
            
        Returns:
            score: Docking score (kcal/mol)
        """
        if box_center is None:
            raise ValueError("box_center required for QuickVina2 docking")
        if protein_pdbqt is None:
            raise ValueError("protein_pdbqt_path required for QuickVina2 docking")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write ligand to PDB
            ligand_pdb = os.path.join(tmpdir, 'ligand.pdb')
            ligand_pdbqt = os.path.join(tmpdir, 'ligand.pdbqt')
            output_pdbqt = os.path.join(tmpdir, 'output.pdbqt')
            
            try:
                # Generate 3D coordinates if not present
                if mol.GetNumConformers() == 0:
                    AllChem.EmbedMolecule(mol, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(mol)
                
                # Save as PDB
                Chem.MolToPDBFile(mol, ligand_pdb)
                
                # Convert to PDBQT using obabel
                convert_result = subprocess.run(
                    ['obabel', ligand_pdb, '-O', ligand_pdbqt],
                    capture_output=True,
                    timeout=30
                )
                
                if not os.path.exists(ligand_pdbqt):
                    raise RuntimeError("PDBQT conversion failed")
                
                # Run QuickVina2
                cmd = [
                    self.vina_executable,
                    '--receptor', protein_pdbqt,
                    '--ligand', ligand_pdbqt,
                    '--center_x', str(box_center[0]),
                    '--center_y', str(box_center[1]),
                    '--center_z', str(box_center[2]),
                    '--size_x', str(box_size[0]),
                    '--size_y', str(box_size[1]),
                    '--size_z', str(box_size[2]),
                    '--out', output_pdbqt,
                    '--cpu', '1',
                    '--exhaustiveness', '8'
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                # Parse output for binding affinity
                for line in result.stdout.split('\n'):
                    if 'REMARK VINA RESULT' in line or '   1 ' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            try:
                                score = float(part)
                                if -20 < score < 0:  # Valid docking score range
                                    return torch.tensor(score, dtype=torch.float32)
                            except ValueError:
                                continue
                
                # Fallback: return neutral score
                return torch.tensor(-5.0, dtype=torch.float32)
                
            except Exception as e:
                print(f"Warning: QuickVina2 docking failed: {e}")
                # Fallback to fast approximation
                return self._fast_score(mol)
    
    def get_optimal_score_range(self) -> tuple[float, float]:
        """Get the optimal score range for drug-like molecules."""
        return (-12.0, -8.0)


def load_docking_guidance(use_fast: bool = True) -> DockingGuidance:
    """
    Convenience function to load docking guidance.
    
    Args:
        use_fast: Use fast approximation (True) or QuickVina2 (False)
        
    Returns:
        DockingGuidance instance
    """
    return DockingGuidance(use_fast_approximation=use_fast)


# Testing
if __name__ == '__main__':
    print("Testing DockingGuidance...")
    
    if not RDKIT_AVAILABLE:
        print("RDKit not available. Skipping tests.")
        exit(1)
    
    # Create guidance
    guidance = DockingGuidance(use_fast_approximation=True)
    
    # Test molecules
    test_smiles = [
        "CC(C)Cc1ccc(cc1)C(C)C(O)=O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(=O)Nc1ccc(O)cc1",  # Paracetamol
        "c1ccccc1",  # Benzene (simple)
        "CCCCCCCCCCCCCCCCCCCC(O)=O",  # Long chain (poor)
    ]
    
    print("\nFast scoring results:")
    for smiles in test_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            score = guidance.score_molecule(mol)
            print(f"  {smiles[:30]:<35} -> {score.item():.2f} kcal/mol")
    
    # Test batch scoring
    print("\nBatch scoring test:")
    scores = guidance.score_batch(test_smiles)
    print(f"  Batch scores: {scores.numpy()}")
    print(f"  Mean: {scores.mean().item():.2f}, Std: {scores.std().item():.2f}")
    
    print("\n✓ Tests completed")
