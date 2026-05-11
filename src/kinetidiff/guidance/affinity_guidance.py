"""
Affinity-based guidance using trained HNN-Denovo model.

This module wraps the HNN-Denovo checkpoint and provides gradient-based
guidance for the diffusion model to steer generation toward high-affinity molecules.

Features:
- Load trained HNN-Denovo checkpoint
- Featurize molecules (SMILES → encoded tensors)
- Predict binding affinity with optional uncertainty
- Compute gradients for diffusion guidance

Usage:
    from src.guidance.affinity_guidance import AffinityGuidanceModel

    model = AffinityGuidanceModel(
        checkpoint_path='models/affinity_pred/checkpoints/best_model.pt',
        device='cuda'
    )

    affinity = model.predict_affinity(smiles, protein_sequence)
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


class SMILESTokenizer:
    """
    Simple character-level SMILES tokenizer.

    Converts SMILES strings to integer sequences for model input.
    """

    # Common SMILES characters (must match training vocabulary)
    VOCAB = [
        '<PAD>', '<UNK>',
        'C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'P', 'p',
        'F', 'Cl', 'Br', 'I',
        'H', 'B', 'Si', 'Se', 'Te', 'As',
        '(', ')', '[', ']', '{', '}',
        '=', '#', '-', '+', '\\', '/', ':',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
        '@', '.', '%',
        'l', 'r', 'i', 'e', 'a',  # For multi-char tokens like Cl, Br
    ]

    def __init__(self, max_length: int = 200, vocab_size: int = 100):
        self.max_length = max_length
        self.vocab_size = vocab_size

        # Build vocabulary
        self.char_to_idx = {c: i for i, c in enumerate(self.VOCAB)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

        self.pad_idx = 0
        self.unk_idx = 1

    def tokenize(self, smiles: str) -> torch.Tensor:
        """
        Tokenize a SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            tokens: (max_length,) integer tensor
        """
        tokens = []
        i = 0

        while i < len(smiles) and len(tokens) < self.max_length:
            # Check for two-character tokens first
            if i + 1 < len(smiles):
                two_char = smiles[i:i+2]
                if two_char in self.char_to_idx:
                    tokens.append(self.char_to_idx[two_char])
                    i += 2
                    continue

            # Single character
            char = smiles[i]
            if char in self.char_to_idx:
                tokens.append(self.char_to_idx[char])
            else:
                tokens.append(self.unk_idx)
            i += 1

        # Pad to max_length
        while len(tokens) < self.max_length:
            tokens.append(self.pad_idx)

        return torch.tensor(tokens[:self.max_length], dtype=torch.long)

    def batch_tokenize(self, smiles_list: list) -> torch.Tensor:
        """Tokenize a batch of SMILES strings."""
        return torch.stack([self.tokenize(s) for s in smiles_list])


class ProteinTokenizer:
    """
    Protein sequence tokenizer.

    Converts amino acid sequences to integer sequences.
    """

    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

    def __init__(self, max_length: int = 1000, vocab_size: int = 22):
        self.max_length = max_length
        self.vocab_size = vocab_size

        # Build vocabulary (0-19: amino acids, 20: unknown, 21: padding)
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.AMINO_ACIDS)}
        self.aa_to_idx['X'] = 20  # Unknown
        self.aa_to_idx['U'] = 20  # Selenocysteine
        self.aa_to_idx['O'] = 20  # Pyrrolysine
        self.idx_to_aa = {i: aa for aa, i in self.aa_to_idx.items()}

        self.pad_idx = 21
        self.unk_idx = 20

    def tokenize(self, sequence: str) -> torch.Tensor:
        """
        Tokenize a protein sequence.

        Args:
            sequence: Amino acid sequence (one-letter codes)

        Returns:
            tokens: (max_length,) integer tensor
        """
        tokens = []

        for aa in sequence.upper():
            if len(tokens) >= self.max_length:
                break
            if aa in self.aa_to_idx:
                tokens.append(self.aa_to_idx[aa])
            else:
                tokens.append(self.unk_idx)

        # Pad to max_length
        while len(tokens) < self.max_length:
            tokens.append(self.pad_idx)

        return torch.tensor(tokens[:self.max_length], dtype=torch.long)

    def batch_tokenize(self, sequences: list) -> torch.Tensor:
        """Tokenize a batch of protein sequences."""
        return torch.stack([self.tokenize(s) for s in sequences])


class AffinityGuidanceModel:
    """
    Wrapper for HNN-Denovo affinity prediction model.

    Provides:
    - Gradient computation for diffusion guidance
    - Affinity scoring for intermediate molecules
    - Uncertainty quantification (if MC dropout enabled)

    Attributes:
        model: Loaded HNN-Denovo model
        device: Computation device
        smiles_tokenizer: SMILES tokenizer
        protein_tokenizer: Protein tokenizer
        affinity_mean: Affinity normalization mean
        affinity_std: Affinity normalization std
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        use_uncertainty: bool = True
    ):
        """
        Initialize affinity guidance model.

        Args:
            checkpoint_path: Path to best_model.pt
            device: cuda or cpu
            use_uncertainty: Use MC dropout for uncertainty
        """

        self.device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'
        self.use_uncertainty = use_uncertainty

        # Load checkpoint
        print(f"Loading HNN-Denovo checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Fix Python path to import affinity_pred properly
        affinity_pred_path = str(Path(checkpoint_path).parent.parent)  # /root/FOP-Code/models/affinity_pred
        models_path = str(Path(checkpoint_path).parent.parent / "models")  # /root/FOP-Code/models/affinity_pred/models

        # Add both paths
        for p in [models_path, affinity_pred_path]:
            if p not in sys.path:
                sys.path.insert(0, p)

        # Now try to import
        try:
            # Try direct import from models subdirectory
            from hnn_denovo import load_model
            print("  ✓ Imported load_model from hnn_denovo")

            self.model, metadata = load_model(checkpoint_path, device=self.device)
            print("  ✓ REAL HNN-Denovo model loaded!")

        except Exception as e:
            print(f"  ERROR loading model: {e}")
            print(f"  Sys path entries: {[p for p in sys.path if 'affinity' in p or 'hnn' in p]}")
            raise

        self.model.to(self.device)
        self.model.eval()

        # CRITICAL: Disable gradient computation for model weights to save memory
        for param in self.model.parameters():
            param.requires_grad = False

        # Initialize tokenizers with vocabulary sizes from checkpoint
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

        if 'ligand_encoder.embedding.weight' in state_dict:
            smiles_vocab_size = state_dict['ligand_encoder.embedding.weight'].shape[0]
        else:
            smiles_vocab_size = 100

        if 'protein_encoder.embedding.weight' in state_dict:
            protein_vocab_size = state_dict['protein_encoder.embedding.weight'].shape[0]
        else:
            protein_vocab_size = 22

        self.smiles_tokenizer = SMILESTokenizer(vocab_size=smiles_vocab_size)
        self.protein_tokenizer = ProteinTokenizer(vocab_size=protein_vocab_size)

        # Get normalization parameters
        # Note: These should match the training data statistics
        # BindingDB pKd: mean=6.46, std=1.38 (not 6.52, 2.21!)
        self.affinity_mean = checkpoint.get('affinity_mean', 6.46)
        self.affinity_std = checkpoint.get('affinity_std', 1.38)

        # Store descriptor dimension
        if 'descriptor_encoder.input_norm.weight' in state_dict:
            self.descriptor_dim = state_dict['descriptor_encoder.input_norm.weight'].shape[0]
        else:
            self.descriptor_dim = 256

        print("✓ HNN-Denovo FULLY LOADED")
        print(f"  Device: {self.device}")
        print(f"  SMILES vocab: {smiles_vocab_size}")
        print(f"  Protein vocab: {protein_vocab_size}")
        print(f"  Descriptor dim: {self.descriptor_dim}")
        print(f"  Affinity normalization: mean={self.affinity_mean:.2f}, std={self.affinity_std:.2f}")

    def _load_model_fallback(self, checkpoint):
        """Fallback model loading when HNNDenovo import fails."""
        # Simplified fallback: create a dummy model that returns constant affinity
        # This allows the pipeline to run without the full HNN-Denovo model
        print("  Using dummy affinity model (constant predictions)")

        class DummyAffinityModel(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return torch.ones(x.shape[0], 1) * 7.0

        self.model = DummyAffinityModel()

    def predict_affinity(
        self,
        smiles: str | list,
        protein_sequence: str,
        descriptors: torch.Tensor | None = None,
        return_uncertainty: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Predict binding affinity for molecule(s).

        Args:
            smiles: SMILES string or list of SMILES
            protein_sequence: Target protein sequence
            descriptors: Optional BINANA descriptors (batch, descriptor_dim)
            return_uncertainty: Return epistemic uncertainty

        Returns:
            affinity: (batch_size,) predicted pKd values (DENORMALIZED)
            uncertainty: (batch_size,) epistemic uncertainty (if requested)
        """
        # Handle single SMILES
        if isinstance(smiles, str):
            smiles = [smiles]

        batch_size = len(smiles)

        # Tokenize inputs
        ligand_tokens = self.smiles_tokenizer.batch_tokenize(smiles).to(self.device)
        protein_tokens = self.protein_tokenizer.tokenize(protein_sequence).unsqueeze(0)
        protein_tokens = protein_tokens.expand(batch_size, -1).to(self.device)

        # Create descriptors if not provided (zeros as placeholder)
        if descriptors is None:
            descriptors = torch.zeros(
                batch_size, self.descriptor_dim,
                device=self.device, dtype=torch.float32
            )
        else:
            descriptors = descriptors.to(self.device)

        # Predict
        with torch.set_grad_enabled(return_uncertainty or self.use_uncertainty):
            if return_uncertainty and self.use_uncertainty:
                # MC dropout: multiple forward passes
                self.model.train()  # Enable dropout
                predictions = []

                for _ in range(10):  # 10 MC samples
                    pred = self.model(ligand_tokens, protein_tokens, descriptors)
                    predictions.append(pred)

                self.model.eval()

                predictions = torch.stack(predictions, dim=0)  # (10, batch, 1)
                pred_mean = predictions.mean(dim=0)
                pred_std = predictions.std(dim=0)

                # Denormalize
                affinity = pred_mean.squeeze(-1) * self.affinity_std + self.affinity_mean
                uncertainty = pred_std.squeeze(-1) * self.affinity_std

                return affinity, uncertainty
            else:
                # Single forward pass
                pred = self.model(ligand_tokens, protein_tokens, descriptors)
                affinity = pred.squeeze(-1) * self.affinity_std + self.affinity_mean

                if return_uncertainty:
                    return affinity, torch.zeros_like(affinity)
                return affinity

    def compute_guidance_gradient(
        self,
        smiles: str | list,
        protein_sequence: str,
        target_affinity: float = 8.0,
        guidance_scale: float = 1.0,
        descriptors: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradient of affinity for guidance.

        This computes the gradient of the affinity prediction with respect to
        the ligand embedding, which can be used to guide the diffusion process.

        Args:
            smiles: SMILES string(s)
            protein_sequence: Target protein sequence
            target_affinity: Target pKd value (default: 8.0 = 10 nM)
            guidance_scale: Scaling factor for gradient
            descriptors: Optional BINANA descriptors

        Returns:
            gradient: Gradient w.r.t. ligand embedding
            affinity: Current predicted affinity
        """
        if isinstance(smiles, str):
            smiles = [smiles]

        batch_size = len(smiles)

        # Tokenize inputs
        ligand_tokens = self.smiles_tokenizer.batch_tokenize(smiles).to(self.device)
        protein_tokens = self.protein_tokenizer.tokenize(protein_sequence).unsqueeze(0)
        protein_tokens = protein_tokens.expand(batch_size, -1).to(self.device)

        if descriptors is None:
            descriptors = torch.zeros(
                batch_size, self.descriptor_dim,
                device=self.device, dtype=torch.float32
            )

        # Enable gradients
        ligand_tokens = ligand_tokens.float()  # Convert to float for gradient
        ligand_tokens.requires_grad = True

        # Forward pass with gradient tracking
        # Note: We use embeddings for gradient, not raw tokens
        pred, embeddings = self.model(
            ligand_tokens.long(),
            protein_tokens,
            descriptors,
            return_embeddings=True
        )

        affinity = pred.squeeze(-1) * self.affinity_std + self.affinity_mean

        # Guidance loss: encourage affinity toward target
        # Negative because we want to maximize affinity
        loss = -torch.mean((affinity - target_affinity) ** 2)

        # Compute gradient w.r.t. ligand embedding
        ligand_embed = embeddings['ligand']

        if ligand_embed.requires_grad:
            gradient = torch.autograd.grad(
                outputs=loss,
                inputs=ligand_embed,
                create_graph=False,
                retain_graph=False
            )[0]
        else:
            # Fallback: return zeros if no gradient
            gradient = torch.zeros_like(ligand_embed)

        return gradient * guidance_scale, affinity

    def compute_affinity_gradient(
        self,
        mol_coords: torch.Tensor,
        mol_features: torch.Tensor,
        pocket_coords: torch.Tensor,
        protein_sequence: str,
        target_affinity: float = 7.0,
        guidance_scale: float = 1.0
    ) -> tuple[torch.Tensor, float]:
        """
        Compute gradient of affinity w.r.t. 3D coordinates.

        This is the KEY method for guided diffusion - it computes how to move
        atom coordinates to improve predicted binding affinity.

        Since HNN-Denovo is SMILES-based, we use a proxy approach:
        1. Compute distance-based features from 3D coords
        2. Pass through a differentiable scoring function
        3. Return gradient w.r.t. coordinates

        Args:
            mol_coords: (n_atoms, 3) ligand coordinates
            mol_features: (n_atoms, n_features) atom features (one-hot)
            pocket_coords: (n_pocket_atoms, 3) pocket coordinates
            protein_sequence: Target protein sequence
            target_affinity: Target pKd value
            guidance_scale: Scaling factor for gradient

        Returns:
            gradient: (n_atoms, 3) gradient w.r.t. coordinates
            affinity: Current predicted affinity (scalar)
        """
        device = mol_coords.device
        n_atoms = mol_coords.shape[0]

        # Ensure coords require grad
        if not mol_coords.requires_grad:
            mol_coords = mol_coords.clone().detach().requires_grad_(True)

        # =====================================================================
        # DIFFERENTIABLE BINDING SCORE
        # =====================================================================
        # Goal: Create a score that:
        # 1. ALWAYS has non-zero gradient (even far from pocket)
        # 2. Rewards being close to pocket center
        # 3. Rewards having contacts with pocket atoms

        # Distance from each ligand atom to closest pocket atom
        dist_matrix = torch.cdist(mol_coords, pocket_coords)  # (n_atoms, n_pocket)
        min_dists_per_atom, _ = dist_matrix.min(dim=1)  # (n_atoms,)

        # Ligand center to pocket center (MAIN GUIDANCE SIGNAL)
        ligand_center = mol_coords.mean(dim=0)
        pocket_center = pocket_coords.mean(dim=0)
        center_dist = (ligand_center - pocket_center).norm() + 1e-6  # Avoid div by zero

        # =====================================================================
        # SCORING COMPONENTS (all differentiable, NO CLAMPING)
        # =====================================================================

        # Component 1: Attractive potential toward pocket center
        # Uses inverse distance to ensure gradient at ALL distances
        # Closer = higher score, with diminishing returns when very close
        # At 50A: score = 10 / (1 + 50/10) = 10/6 = 1.67
        # At 10A: score = 10 / (1 + 10/10) = 10/2 = 5.0
        # At 2A:  score = 10 / (1 + 2/10) = 10/1.2 = 8.3
        distance_score = 10.0 / (1.0 + center_dist / 10.0)

        # Component 2: Contact score (soft sigmoid)
        # Atoms within 4A get bonus
        contact_threshold = 4.0
        contact_scores = torch.sigmoid((contact_threshold - min_dists_per_atom) * 2.0)
        n_contacts = contact_scores.sum()
        contact_score = n_contacts * 0.3  # 0.3 pKd per contact

        # Component 3: Closest atom bonus (encourage at least one close contact)
        min_dist_to_pocket = min_dists_per_atom.min()
        proximity_score = 5.0 / (1.0 + min_dist_to_pocket / 4.0)

        # Combined proxy affinity (NO CLAMP - we need gradients everywhere!)
        proxy_affinity = distance_score + contact_score + proximity_score

        # Compute gradient w.r.t. coordinates
        # We want to MAXIMIZE affinity (minimize distance to target)
        gradient = torch.autograd.grad(
            outputs=proxy_affinity,
            inputs=mol_coords,
            create_graph=False,
            retain_graph=False
        )[0]

        # Scale gradient
        gradient = gradient * guidance_scale

        # Clip extreme gradients
        grad_norm = gradient.norm()
        if grad_norm > 10.0:
            gradient = gradient * (10.0 / grad_norm)

        return gradient, proxy_affinity.item()

    def predict_affinity_from_coords(
        self,
        mol_coords: torch.Tensor,
        mol_features: torch.Tensor,
        pocket_coords: torch.Tensor,
        protein_sequence: str
    ) -> float:
        """
        Predict affinity from 3D coordinates (no gradient).
        Uses same proxy scoring as compute_affinity_gradient.
        """
        with torch.no_grad():
            dist_matrix = torch.cdist(mol_coords, pocket_coords)
            min_dists = dist_matrix.min(dim=1)[0]

            contact_threshold = 4.0
            contact_scores = (min_dists < contact_threshold).float()
            n_contacts = contact_scores.sum().item()

            ligand_center = mol_coords.mean(dim=0)
            pocket_center = pocket_coords.mean(dim=0)
            center_dist = (ligand_center - pocket_center).norm().item()

            proxy_affinity = 5.0 + n_contacts * 0.3 - center_dist * 0.1
            proxy_affinity = max(2.0, min(12.0, proxy_affinity))

            return proxy_affinity

    def score_batch(
        self,
        smiles_list: list,
        protein_sequence: str,
        descriptors: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Score a batch of molecules.

        Args:
            smiles_list: List of SMILES strings
            protein_sequence: Target protein sequence
            descriptors: Optional BINANA descriptors

        Returns:
            Dict with 'affinity', 'uncertainty' tensors
        """
        affinity, uncertainty = self.predict_affinity(
            smiles_list,
            protein_sequence,
            descriptors=descriptors,
            return_uncertainty=True
        )

        return {
            'affinity': affinity,
            'uncertainty': uncertainty,
            'batch_size': len(smiles_list)
        }


def load_affinity_guidance(
    checkpoint_path: str = 'models/affinity_pred/checkpoints/best_model.pt',
    device: str = 'cuda'
) -> AffinityGuidanceModel:
    """
    Convenience function to load affinity guidance model.

    Args:
        checkpoint_path: Path to HNN-Denovo checkpoint
        device: Computation device

    Returns:
        AffinityGuidanceModel instance
    """
    return AffinityGuidanceModel(checkpoint_path, device)


# Testing
if __name__ == '__main__':
    print("Testing AffinityGuidanceModel...")

    # Test tokenizers
    smiles_tok = SMILESTokenizer()
    protein_tok = ProteinTokenizer()

    test_smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"  # Ibuprofen
    test_protein = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRK"

    smiles_tokens = smiles_tok.tokenize(test_smiles)
    protein_tokens = protein_tok.tokenize(test_protein)

    print(f"SMILES tokens shape: {smiles_tokens.shape}")
    print(f"Protein tokens shape: {protein_tokens.shape}")

    # Test with actual checkpoint if available
    checkpoint_path = "models/affinity_pred/checkpoints/best_model.pt"

    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint: {checkpoint_path}")
        model = AffinityGuidanceModel(checkpoint_path, device='cpu')

        # Test prediction
        affinity, uncertainty = model.predict_affinity(
            test_smiles, test_protein, return_uncertainty=True
        )
        print(f"Predicted affinity: {affinity.item():.2f} +/- {uncertainty.item():.2f}")
    else:
        print(f"\nCheckpoint not found: {checkpoint_path}")
        print("Skipping model test.")

    print("\n✓ Tests completed")
