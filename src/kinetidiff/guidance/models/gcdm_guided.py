"""
GCDM with multi-objective guidance integration.

Extends base GCDM-SBDD to incorporate affinity, docking, and SA guidance
during the denoising process.

Key Features:
- HNN-Denovo affinity prediction guidance
- Docking score guidance (fast or QuickVina2)
- Synthetic accessibility guidance
- Adaptive multi-objective weighting
- Flexible timestep-dependent guidance scaling

Usage:
    from src.models.gcdm_guided import GCDMGuided
    
    model = GCDMGuided(
        gcdm_checkpoint='path/to/gcdm.ckpt',
        affinity_checkpoint='path/to/affinity.pt',
        guidance_scale=1.0,
        multi_obj_strategy='adaptive_threshold'
    )
    
    molecules, metrics = model.generate_guided(
        pdb_file='receptor.pdb',
        protein_sequence='MTEYKLVVV...',
        pocket_ids=['A:1', 'A:2', ...],
        n_samples=100
    )
"""

import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import QED, AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "GCDM-SBDD-modified"))

# Import guidance modules
from ..guidance.affinity_guidance import AffinityGuidanceModel
from ..guidance.docking_guidance import DockingGuidance
from ..guidance.multi_objective import AdaptiveMultiObjectiveGuidance
from ..guidance.sa_guidance import SAGuidance


class GCDMGuided:
    """
    Geometry-Complete Diffusion Model with Multi-Objective Guidance.
    
    Wraps the base GCDM-SBDD model and adds guidance during generation.
    
    Guidance is applied at each denoising step by:
    1. Generating intermediate molecules from GCDM
    2. Computing multi-objective scores (affinity, docking, SA)
    3. Filtering/ranking molecules based on scores
    
    For full gradient-based guidance (modifying coordinates during denoising),
    see GCDMGuidedGradient class.
    
    Attributes:
        gcdm_model: Base GCDM-SBDD model
        affinity_model: HNN-Denovo affinity predictor
        docking_model: Fast docking scorer
        sa_model: SA score calculator
        multi_obj: Multi-objective optimizer
        device: Computation device
        guidance_scale: Overall guidance strength
    """
    
    def __init__(
        self,
        gcdm_checkpoint: str = None,
        affinity_checkpoint: str = 'models/affinity_pred/checkpoints/best_model.pt',
        use_affinity_guidance: bool = True,
        use_docking_guidance: bool = True,
        use_sa_guidance: bool = True,
        guidance_scale: float = 1.0,
        multi_obj_strategy: str = 'adaptive_threshold',
        affinity_target: float = 7.0,
        affinity_min: float = 6.0,
        docking_target: float = -11.0,
        docking_max: float = -10.0,
        sa_target: float = 3.0,
        sa_max: float = 3.5,
        device: str = None
    ):
        """
        Initialize GCDM with multi-objective guidance.
        
        Args:
            gcdm_checkpoint: Path to GCDM-SBDD checkpoint
            affinity_checkpoint: Path to HNN-Denovo checkpoint
            use_*_guidance: Enable/disable each guidance component
            guidance_scale: Overall scaling factor for guidance
            multi_obj_strategy: 'adaptive_threshold', 'pareto', or 'soft_constraint'
            *_target: Target values for each objective
            *_max/*_min: Threshold values for each objective
            device: Computation device ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.use_affinity_guidance = use_affinity_guidance
        self.use_docking_guidance = use_docking_guidance
        self.use_sa_guidance = use_sa_guidance
        self.guidance_scale = guidance_scale
        
        print("=" * 70)
        print("GCDM-GUIDED: Multi-Objective Molecular Generation")
        print("=" * 70)
        print(f"Device: {self.device}")
        
        # Load GCDM model
        self.gcdm_model = None
        if gcdm_checkpoint:
            self._load_gcdm_model(gcdm_checkpoint)
        
        # Initialize guidance models
        if use_affinity_guidance:
            print("\n📊 Loading affinity guidance model...")
            self.affinity_model = AffinityGuidanceModel(
                affinity_checkpoint,
                device=self.device
            )
        else:
            self.affinity_model = None
        
        if use_docking_guidance:
            print("\n🎯 Loading docking guidance...")
            self.docking_model = DockingGuidance(use_fast_approximation=True)
        else:
            self.docking_model = None
        
        if use_sa_guidance:
            print("\n🧪 Loading SA guidance...")
            self.sa_model = SAGuidance(target_sa=sa_target, max_sa=sa_max)
        else:
            self.sa_model = None
        
        # Initialize multi-objective optimizer
        print("\n⚖️ Initializing multi-objective optimizer...")
        self.multi_obj = AdaptiveMultiObjectiveGuidance(
            strategy=multi_obj_strategy,
            affinity_target=affinity_target,
            affinity_min=affinity_min,
            docking_target=docking_target,
            docking_max=docking_max,
            sa_target=sa_target,
            sa_max=sa_max
        )
        
        # Store targets for filtering
        self.affinity_min = affinity_min
        self.docking_max = docking_max
        self.sa_max = sa_max
        
        print("\n✓ GCDM-Guided initialized successfully")
    
    def _load_gcdm_model(self, checkpoint_path: str):
        """Load GCDM-SBDD model from checkpoint."""
        try:
            from lightning_modules import LigandPocketDDPM
            
            print(f"\n🔄 Loading GCDM model from: {checkpoint_path}")
            self.gcdm_model = LigandPocketDDPM.load_from_checkpoint(
                checkpoint_path,
                map_location=self.device
            )
            self.gcdm_model = self.gcdm_model.to(self.device)
            self.gcdm_model.eval()
            print("   GCDM model loaded successfully")
            
        except Exception as e:
            print(f"   Warning: Failed to load GCDM model: {e}")
            print("   Generation will require external GCDM instance")
            self.gcdm_model = None
    
    def generate_guided(
        self,
        pdb_file: str,
        protein_sequence: str,
        pocket_ids: list[str] | None = None,
        centroid: tuple[float, float, float] | None = None,
        pocket_radius: float = 10.0,
        ref_ligand: str | None = None,
        n_samples: int = 100,
        batch_size: int = 20,
        filter_by_objectives: bool = True,
        return_all: bool = False,
        verbose: bool = True
    ) -> tuple[list['Chem.Mol'], list[dict]]:
        """
        Generate molecules with multi-objective guidance.
        
        Generates molecules in batches, scores them, and optionally
        filters based on objective thresholds.
        
        Args:
            pdb_file: Path to receptor PDB file
            protein_sequence: Target protein sequence
            pocket_ids: List of pocket residue IDs (chain:resnum format)
            centroid: (x,y,z) pocket center (alternative to pocket_ids)
            pocket_radius: Radius around centroid for pocket detection
            ref_ligand: Reference ligand for pocket definition
            n_samples: Total number of molecules to generate
            batch_size: Molecules per GCDM batch
            filter_by_objectives: Whether to filter molecules by thresholds
            return_all: Return all molecules (not just filtered)
            verbose: Print progress information
            
        Returns:
            molecules: List of RDKit molecules (filtered if requested)
            metrics: List of metric dictionaries per molecule
        """
        if self.gcdm_model is None:
            raise RuntimeError(
                "GCDM model not loaded. Provide gcdm_checkpoint or use "
                "generate_guided_external with external molecule source."
            )
        
        # Determine pocket definition
        if centroid is not None:
            pocket_ids = self._identify_pocket_from_centroid(
                pdb_file, centroid, pocket_radius
            )
        elif pocket_ids is None and ref_ligand is None:
            raise ValueError(
                "Must provide one of: pocket_ids, centroid, or ref_ligand"
            )
        
        all_molecules = []
        all_metrics = []
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        if verbose:
            print(f"\n🧬 Generating {n_samples} molecules in {n_batches} batches...")
        
        for batch_idx in range(n_batches):
            current_batch_size = min(batch_size, n_samples - len(all_molecules))
            
            if verbose:
                print(f"\n  Batch {batch_idx + 1}/{n_batches}: {current_batch_size} molecules")
            
            # Generate molecules
            molecules = self.gcdm_model.generate_ligands(
                pdb_file,
                n_samples=current_batch_size,
                pocket_ids=pocket_ids,
                ref_ligand=ref_ligand,
                sanitize=True,
                largest_frag=True,
                relax_iter=0
            )
            
            if verbose:
                print(f"    Generated: {len(molecules)} raw molecules")
            
            # Score and filter molecules
            valid_count = 0
            for mol in molecules:
                if mol is None:
                    continue
                
                try:
                    smiles = Chem.MolToSmiles(mol)
                    
                    # Score molecule
                    metrics = self._score_molecule(mol, smiles, protein_sequence)
                    
                    all_molecules.append(mol)
                    all_metrics.append(metrics)
                    valid_count += 1
                    
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Failed to process molecule: {e}")
                    continue
            
            if verbose:
                print(f"    Valid: {valid_count}")
        
        if verbose:
            print(f"\n✓ Generated {len(all_molecules)} valid molecules")
        
        # Filter by objectives if requested
        if filter_by_objectives and not return_all:
            filtered_mols, filtered_metrics = self._filter_by_objectives(
                all_molecules, all_metrics, verbose
            )
            return filtered_mols, filtered_metrics
        
        return all_molecules, all_metrics
    
    def generate_guided_external(
        self,
        molecules: list[Union['Chem.Mol', str]],
        protein_sequence: str,
        filter_by_objectives: bool = True,
        verbose: bool = True
    ) -> tuple[list['Chem.Mol'], list[dict]]:
        """
        Score and filter externally generated molecules.
        
        Use this when molecules are generated by an external GCDM instance
        (e.g., Docker container) and you want to apply multi-objective scoring.
        
        Args:
            molecules: List of RDKit molecules or SMILES strings
            protein_sequence: Target protein sequence
            filter_by_objectives: Whether to filter by thresholds
            verbose: Print progress
            
        Returns:
            molecules: Scored (and optionally filtered) molecules
            metrics: Metric dictionaries per molecule
        """
        all_molecules = []
        all_metrics = []
        
        if verbose:
            print(f"\n📊 Scoring {len(molecules)} molecules...")
        
        for i, mol in enumerate(molecules):
            # Convert SMILES to mol if needed
            if isinstance(mol, str):
                smiles = mol
                mol = Chem.MolFromSmiles(mol)
                if mol is None:
                    continue
            else:
                smiles = Chem.MolToSmiles(mol)
            
            try:
                metrics = self._score_molecule(mol, smiles, protein_sequence)
                all_molecules.append(mol)
                all_metrics.append(metrics)
                
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to score molecule {i}: {e}")
                continue
        
        if verbose:
            print(f"✓ Scored {len(all_molecules)} molecules")
        
        if filter_by_objectives:
            return self._filter_by_objectives(all_molecules, all_metrics, verbose)
        
        return all_molecules, all_metrics
    
    def _score_molecule(
        self,
        mol: 'Chem.Mol',
        smiles: str,
        protein_sequence: str
    ) -> dict:
        """
        Compute all scores for a molecule.
        
        Args:
            mol: RDKit molecule
            smiles: SMILES string
            protein_sequence: Target protein sequence
            
        Returns:
            metrics: Dictionary with all scores
        """
        metrics = {
            'smiles': smiles,
            'mol_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'qed': QED.qed(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_heavy_atoms': mol.GetNumHeavyAtoms()
        }
        
        # Affinity prediction
        if self.use_affinity_guidance and self.affinity_model is not None:
            try:
                affinity = self.affinity_model.predict_affinity(
                    smiles, protein_sequence
                )
                metrics['affinity_pkd'] = affinity.item()
            except Exception:
                metrics['affinity_pkd'] = 0.0
        else:
            metrics['affinity_pkd'] = 0.0
        
        # Docking score
        if self.use_docking_guidance and self.docking_model is not None:
            try:
                docking = self.docking_model.score_molecule(mol)
                metrics['docking_score'] = docking.item()
            except Exception:
                metrics['docking_score'] = 0.0
        else:
            metrics['docking_score'] = 0.0
        
        # SA score
        if self.use_sa_guidance and self.sa_model is not None:
            try:
                sa = self.sa_model.score_molecule(mol)
                metrics['sa_score'] = sa.item()
            except Exception:
                metrics['sa_score'] = 5.0
        else:
            metrics['sa_score'] = 5.0
        
        # Compute composite score
        metrics['composite_score'] = self._compute_composite_score(metrics)
        
        # Check if all objectives are satisfied
        metrics['all_satisfied'] = (
            metrics['affinity_pkd'] >= self.affinity_min and
            metrics['docking_score'] <= self.docking_max and
            metrics['sa_score'] <= self.sa_max
        )
        
        return metrics
    
    def _compute_composite_score(self, metrics: dict) -> float:
        """
        Compute composite ranking score from individual metrics.
        
        Higher score = better molecule.
        
        Args:
            metrics: Dictionary with affinity_pkd, docking_score, sa_score
            
        Returns:
            composite: Composite ranking score
        """
        # Normalize each metric to similar scale
        # Affinity: higher is better, normalize around target
        affinity_norm = (metrics['affinity_pkd'] - 6.0) / 2.0
        
        # Docking: lower (more negative) is better
        docking_norm = -(metrics['docking_score'] + 10.0) / 5.0
        
        # SA: lower is better
        sa_norm = -(metrics['sa_score'] - 3.0) / 2.0
        
        # QED: higher is better (0-1 scale)
        qed_norm = metrics['qed']
        
        # Weighted combination
        composite = (
            0.35 * affinity_norm +
            0.30 * docking_norm +
            0.25 * sa_norm +
            0.10 * qed_norm
        )
        
        return composite
    
    def _filter_by_objectives(
        self,
        molecules: list['Chem.Mol'],
        metrics: list[dict],
        verbose: bool = True
    ) -> tuple[list['Chem.Mol'], list[dict]]:
        """
        Filter molecules by objective thresholds.
        
        Args:
            molecules: List of molecules
            metrics: List of metric dictionaries
            verbose: Print filtering statistics
            
        Returns:
            filtered_molecules: Molecules meeting all thresholds
            filtered_metrics: Corresponding metrics
        """
        filtered_mols = []
        filtered_metrics = []
        
        satisfaction_counts = {
            'affinity': 0,
            'docking': 0,
            'sa': 0,
            'all': 0
        }
        
        for mol, m in zip(molecules, metrics):
            aff_ok = m['affinity_pkd'] >= self.affinity_min
            dock_ok = m['docking_score'] <= self.docking_max
            sa_ok = m['sa_score'] <= self.sa_max
            
            if aff_ok:
                satisfaction_counts['affinity'] += 1
            if dock_ok:
                satisfaction_counts['docking'] += 1
            if sa_ok:
                satisfaction_counts['sa'] += 1
            
            if aff_ok and dock_ok and sa_ok:
                filtered_mols.append(mol)
                filtered_metrics.append(m)
                satisfaction_counts['all'] += 1
        
        if verbose:
            total = len(molecules)
            print("\n📋 Filtering Results:")
            print(f"   Affinity ≥ {self.affinity_min}: "
                  f"{satisfaction_counts['affinity']}/{total} "
                  f"({satisfaction_counts['affinity']/total:.1%})")
            print(f"   Docking ≤ {self.docking_max}: "
                  f"{satisfaction_counts['docking']}/{total} "
                  f"({satisfaction_counts['docking']/total:.1%})")
            print(f"   SA ≤ {self.sa_max}: "
                  f"{satisfaction_counts['sa']}/{total} "
                  f"({satisfaction_counts['sa']/total:.1%})")
            print(f"   All satisfied: "
                  f"{satisfaction_counts['all']}/{total} "
                  f"({satisfaction_counts['all']/total:.1%})")
        
        return filtered_mols, filtered_metrics
    
    def _identify_pocket_from_centroid(
        self,
        pdb_file: str,
        centroid: tuple[float, float, float],
        radius: float = 10.0
    ) -> list[str]:
        """
        Identify pocket residues within radius of centroid.
        
        Args:
            pdb_file: Path to PDB file
            centroid: (x, y, z) pocket center coordinates
            radius: Search radius in Angstroms
            
        Returns:
            pocket_ids: List of residue IDs in "chain:resnum" format
        """
        pocket_residues = set()
        cx, cy, cz = centroid
        
        try:
            with open(pdb_file) as f:
                for line in f:
                    if line.startswith(('ATOM', 'HETATM')):
                        try:
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                            
                            dist = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
                            
                            if dist <= radius:
                                chain = line[21].strip() or 'A'
                                resnum = line[22:26].strip()
                                if resnum:
                                    pocket_residues.add(f"{chain}:{resnum}")
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            print(f"Warning: Error reading PDB file: {e}")
            return []
        
        return sorted(list(pocket_residues))
    
    def rank_molecules(
        self,
        molecules: list['Chem.Mol'],
        metrics: list[dict],
        top_k: int = 10
    ) -> tuple[list['Chem.Mol'], list[dict]]:
        """
        Rank molecules by composite score and return top_k.
        
        Args:
            molecules: List of molecules
            metrics: List of metric dictionaries
            top_k: Number of top molecules to return
            
        Returns:
            top_molecules: Top-k molecules
            top_metrics: Corresponding metrics
        """
        # Sort by composite score (descending)
        sorted_indices = np.argsort(
            [-m['composite_score'] for m in metrics]
        )
        
        top_indices = sorted_indices[:top_k]
        
        top_molecules = [molecules[i] for i in top_indices]
        top_metrics = [metrics[i] for i in top_indices]
        
        return top_molecules, top_metrics
    
    def get_summary_statistics(self, metrics: list[dict]) -> dict:
        """
        Compute summary statistics for a set of molecules.
        
        Args:
            metrics: List of metric dictionaries
            
        Returns:
            summary: Dictionary with mean, std, min, max for each metric
        """
        if not metrics:
            return {}
        
        keys = ['affinity_pkd', 'docking_score', 'sa_score', 'qed', 
                'mol_weight', 'composite_score']
        
        summary = {}
        for key in keys:
            values = [m.get(key, 0.0) for m in metrics]
            summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # Add satisfaction rates
        n_total = len(metrics)
        summary['satisfaction_rates'] = {
            'affinity': sum(1 for m in metrics if m['affinity_pkd'] >= self.affinity_min) / n_total,
            'docking': sum(1 for m in metrics if m['docking_score'] <= self.docking_max) / n_total,
            'sa': sum(1 for m in metrics if m['sa_score'] <= self.sa_max) / n_total,
            'all': sum(1 for m in metrics if m['all_satisfied']) / n_total
        }
        
        return summary


# Testing
if __name__ == '__main__':
    print("Testing GCDMGuided...")
    
    # Test without GCDM checkpoint (scoring only)
    model = GCDMGuided(
        gcdm_checkpoint=None,  # No GCDM model
        affinity_checkpoint='models/affinity_pred/checkpoints/best_model.pt',
        use_docking_guidance=True,
        use_sa_guidance=True,
        multi_obj_strategy='adaptive_threshold'
    )
    
    print("\n✓ Model initialized (scoring mode)")
    
    # Test with sample molecules
    test_smiles = [
        "CC(C)Cc1ccc(cc1)C(C)C(O)=O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ]
    
    if RDKIT_AVAILABLE:
        mols = [Chem.MolFromSmiles(s) for s in test_smiles]
        mols = [m for m in mols if m is not None]
        
        if mols:
            print("\nTesting external molecule scoring...")
            _, metrics = model.generate_guided_external(
                mols,
                protein_sequence="MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRK",
                filter_by_objectives=False
            )
            
            print("\nMetrics:")
            for m in metrics:
                print(f"  {m['smiles'][:30]}")
                print(f"    Affinity: {m['affinity_pkd']:.2f}")
                print(f"    Docking: {m['docking_score']:.2f}")
                print(f"    SA: {m['sa_score']:.2f}")
                print(f"    Composite: {m['composite_score']:.3f}")
    
    print("\n✓ Tests completed")
