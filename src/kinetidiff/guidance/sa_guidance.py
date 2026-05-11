"""
Synthetic Accessibility (SA) scoring guidance.

Uses RDKit's SA score implementation to penalize difficult-to-synthesize molecules.

SA Score range: 1 (easy) to 10 (difficult)
Target for drug discovery: < 3.5 for practical synthesis

The SA score is critical for drug development:
- Score 1-3: Easy to synthesize
- Score 3-4: Moderate difficulty
- Score 4-5: Challenging but feasible
- Score > 5: Difficult, may require specialized chemistry

Usage:
    from src.guidance.sa_guidance import SAGuidance
    
    guidance = SAGuidance(target_sa=3.0, max_sa=3.5)
    score = guidance.score_molecule(mol)
    penalty = guidance.compute_penalty(score)
"""

import os
import sys
from typing import Union

import numpy as np
import torch

# RDKit imports
try:
    from rdkit import Chem, RDConfig
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. SA guidance will be limited.")

# Try to import SA_Score from RDKit contrib
try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    SASCORER_AVAILABLE = True
except (ImportError, AttributeError):
    SASCORER_AVAILABLE = False
    print("Warning: RDKit SA_Score not available. Using fallback implementation.")


class SAGuidance:
    """
    Synthetic Accessibility scoring for molecules.
    
    Uses RDKit's SA_Score when available, with fallback to
    empirical complexity-based scoring.
    
    SA Score interpretation:
    - 1: Very easy to synthesize (simple fragments)
    - 2-3: Easy (typical drug-like)
    - 3-4: Moderate (complex drug-like)
    - 4-5: Challenging (natural product-like)
    - 5+: Difficult (highly complex)
    
    Attributes:
        target_sa: Ideal SA score to aim for
        max_sa: Maximum acceptable SA score
        penalty_multiplier: Multiplier for SA violations
    """
    
    def __init__(
        self,
        target_sa: float = 3.0,
        max_sa: float = 3.5,
        penalty_multiplier: float = 10.0
    ):
        """
        Initialize SA guidance.
        
        Args:
            target_sa: Ideal SA score (default: 3.0)
            max_sa: Maximum acceptable SA score (default: 3.5)
            penalty_multiplier: Penalty multiplier for SA > max_sa
        """
        self.target_sa = target_sa
        self.max_sa = max_sa
        self.penalty_multiplier = penalty_multiplier
        
        self.use_rdkit_sascorer = SASCORER_AVAILABLE
        
        if self.use_rdkit_sascorer:
            print("SA Guidance: Using RDKit SA_Score")
        else:
            print("SA Guidance: Using fallback complexity scoring")
        
        print(f"  Target SA: {target_sa}, Max SA: {max_sa}")
    
    def score_molecule(
        self,
        mol: Union['Chem.Mol', str]
    ) -> torch.Tensor:
        """
        Compute SA score for a molecule.
        
        Args:
            mol: RDKit molecule or SMILES string
            
        Returns:
            sa_score: Float in range [1, 10]
        """
        # Convert SMILES to mol if needed
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        
        if mol is None:
            return torch.tensor(10.0, dtype=torch.float32)  # Maximum penalty
        
        try:
            if self.use_rdkit_sascorer:
                sa = sascorer.calculateScore(mol)
            else:
                sa = self._fallback_sa_score(mol)
            
            return torch.tensor(sa, dtype=torch.float32)
            
        except Exception as e:
            print(f"Warning: SA calculation failed: {e}")
            return torch.tensor(5.0, dtype=torch.float32)  # Moderate penalty
    
    def score_batch(
        self,
        molecules: list[Union['Chem.Mol', str]]
    ) -> torch.Tensor:
        """
        Score a batch of molecules.
        
        Args:
            molecules: List of RDKit molecules or SMILES strings
            
        Returns:
            scores: (batch_size,) tensor of SA scores
        """
        scores = []
        for mol in molecules:
            score = self.score_molecule(mol)
            scores.append(score)
        
        return torch.stack(scores)
    
    def compute_penalty(
        self,
        sa_score: float | torch.Tensor
    ) -> torch.Tensor:
        """
        Compute penalty based on SA score.
        
        Returns strong exponential penalty if SA > max_sa.
        
        Args:
            sa_score: SA score (scalar or tensor)
            
        Returns:
            penalty: Penalty value (0 if SA within range)
        """
        if isinstance(sa_score, (int, float)):
            sa_score = torch.tensor(sa_score, dtype=torch.float32)
        
        # Below max: quadratic penalty toward target
        below_max_penalty = (sa_score - self.target_sa) ** 2
        
        # Above max: exponential penalty
        above_max_penalty = self.penalty_multiplier * (
            torch.exp(sa_score - self.max_sa) - 1
        )
        
        # Use above_max penalty only when sa_score > max_sa
        penalty = torch.where(
            sa_score > self.max_sa,
            below_max_penalty + above_max_penalty,
            below_max_penalty
        )
        
        return penalty
    
    def _fallback_sa_score(self, mol: 'Chem.Mol') -> float:
        """
        Fallback SA score calculation using molecular complexity.
        
        Based on:
        - Molecular complexity (rings, stereocenters)
        - Fragment novelty (unusual substructures)
        - Synthetic feasibility heuristics
        
        Args:
            mol: RDKit molecule
            
        Returns:
            sa_score: Estimated SA score [1, 10]
        """
        # Start with base score
        score = 2.0
        
        # Molecular properties
        n_atoms = mol.GetNumHeavyAtoms()
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        n_rot = Descriptors.NumRotatableBonds(mol)
        n_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
        
        # Complexity from size
        if n_atoms > 30:
            score += (n_atoms - 30) * 0.05
        
        # Ring complexity
        score += n_rings * 0.3
        if n_rings > 4:
            score += (n_rings - 4) * 0.5  # Extra penalty for many rings
        
        # Stereocenters (each adds complexity)
        score += n_stereo * 0.4
        
        # Fused ring systems
        ring_info = mol.GetRingInfo()
        n_fused = 0
        rings = [set(r) for r in ring_info.AtomRings()]
        for i, r1 in enumerate(rings):
            for r2 in rings[i+1:]:
                if len(r1 & r2) >= 2:  # Fused rings share 2+ atoms
                    n_fused += 1
        score += n_fused * 0.3
        
        # Heteroatom complexity
        n_hetero = 0
        unusual_hetero = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in ['C', 'H']:
                n_hetero += 1
                if atom.GetSymbol() in ['B', 'P', 'Si', 'Se', 'Te']:
                    unusual_hetero += 1
        
        if n_hetero > 8:
            score += (n_hetero - 8) * 0.1
        score += unusual_hetero * 0.5
        
        # Spiro centers
        # (atoms in multiple rings that aren't shared)
        for atom in mol.GetAtoms():
            if ring_info.NumAtomRings(atom.GetIdx()) > 1:
                atom_rings = [set(r) for r in ring_info.AtomRings() 
                             if atom.GetIdx() in r]
                # Check if it's a spiro (in multiple rings but rings don't share other atoms)
                is_spiro = True
                for i, r1 in enumerate(atom_rings):
                    for r2 in atom_rings[i+1:]:
                        if len(r1 & r2) > 1:  # Share more than just this atom
                            is_spiro = False
                            break
                if is_spiro and len(atom_rings) > 1:
                    score += 0.5
        
        # Aromatic bonus (easier than complex aliphatic systems)
        if n_aromatic > 0 and n_aromatic <= n_rings:
            score -= min(n_aromatic * 0.2, 0.5)
        
        # Normalize to [1, 10] range
        score = np.clip(score, 1.0, 10.0)
        
        return score
    
    def is_synthesizable(self, mol: Union['Chem.Mol', str]) -> bool:
        """
        Check if molecule is considered synthesizable.
        
        Args:
            mol: RDKit molecule or SMILES
            
        Returns:
            True if SA score <= max_sa
        """
        score = self.score_molecule(mol)
        return score.item() <= self.max_sa
    
    def get_synthesis_difficulty(
        self,
        mol: Union['Chem.Mol', str]
    ) -> tuple[str, float]:
        """
        Get human-readable synthesis difficulty.
        
        Args:
            mol: RDKit molecule or SMILES
            
        Returns:
            difficulty: String description
            score: SA score
        """
        score = self.score_molecule(mol).item()
        
        if score <= 2.0:
            difficulty = "Very Easy"
        elif score <= 3.0:
            difficulty = "Easy"
        elif score <= 3.5:
            difficulty = "Moderate"
        elif score <= 4.5:
            difficulty = "Challenging"
        elif score <= 5.5:
            difficulty = "Difficult"
        else:
            difficulty = "Very Difficult"
        
        return difficulty, score


def load_sa_guidance(
    target: float = 3.0,
    max_threshold: float = 3.5
) -> SAGuidance:
    """
    Convenience function to load SA guidance.
    
    Args:
        target: Target SA score
        max_threshold: Maximum acceptable SA score
        
    Returns:
        SAGuidance instance
    """
    return SAGuidance(target_sa=target, max_sa=max_threshold)


# Testing
if __name__ == '__main__':
    print("Testing SAGuidance...")
    
    if not RDKIT_AVAILABLE:
        print("RDKit not available. Skipping tests.")
        exit(1)
    
    # Create guidance
    guidance = SAGuidance(target_sa=3.0, max_sa=3.5)
    
    # Test molecules (varying complexity)
    test_molecules = [
        ("Benzene", "c1ccccc1"),
        ("Ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(O)=O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(O)=O"),
        ("Paclitaxel-like", "CC1=C2[C@@]([C@]([C@H]([C@@H]3[C@]4([C@H](OC4)C[C@@H]([C@]3(C(=O)[C@@H]2OC(=O)C)C)O)OC(=O)C)(C[C@@H]1OC(=O)C6=CC=CC=C6)O)(C)C"),
        ("Simple amine", "CCN"),
        ("Complex fused", "C1CC2CCC3C(C2C1)CCC4C3CCC5CCCCC45"),
    ]
    
    print("\nSA Score results:")
    print("-" * 60)
    
    for name, smiles in test_molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            difficulty, score = guidance.get_synthesis_difficulty(mol)
            synthesizable = "✓" if guidance.is_synthesizable(mol) else "✗"
            print(f"  {name:<20} SA={score:.2f} ({difficulty:<15}) {synthesizable}")
    
    # Test batch scoring
    print("\nBatch scoring test:")
    smiles_list = [s for _, s in test_molecules if Chem.MolFromSmiles(s)]
    scores = guidance.score_batch(smiles_list)
    print(f"  Scores: {scores.numpy().round(2)}")
    print(f"  Mean: {scores.mean().item():.2f}")
    print(f"  Synthesizable fraction: {(scores <= 3.5).float().mean().item():.1%}")
    
    # Test penalty computation
    print("\nPenalty computation:")
    for sa in [2.5, 3.0, 3.5, 4.0, 5.0]:
        penalty = guidance.compute_penalty(torch.tensor(sa))
        print(f"  SA={sa:.1f} -> penalty={penalty.item():.3f}")
    
    print("\n✓ Tests completed")
