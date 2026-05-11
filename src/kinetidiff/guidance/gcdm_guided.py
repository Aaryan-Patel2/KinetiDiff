"""
GCDM with REAL-TIME affinity guidance during denoising

This module injects gradient-based guidance into the GCDM sampling loop
to steer generation toward high-affinity molecules.

Key insight: We modify `sample_given_pocket` to add affinity gradients
after computing the denoised mean but before sampling the next step.
"""


import numpy as np
import torch
import torch.nn.functional as F

# Import HNNDenovo early
try:
    from kinetidiff.affinity_pred.models.hnn_denovo import HNNDenovo
except ImportError:
    HNNDenovo = None  # Will raise error later if guidance is used

from kinetidiff.gcdm.equivariant_diffusion.conditional_model import ConditionalDDPM

# Try to import from torch_scatter, fall back to our impl
try:
    from torch_scatter import scatter_add, scatter_mean
except ImportError:
    from kinetidiff.gcdm.torch_scatter_impl import scatter_add, scatter_mean

import kinetidiff.gcdm.utils as utils


class SoftBINANAComputation:
    """
    Differentiable BINANA descriptor computation.

    Approximates BINANA features using soft thresholds for gradient flow.
    """

    def __init__(self, device='cuda', descriptor_dim=256):
        self.device = device
        self.descriptor_dim = descriptor_dim

        # Interaction distance thresholds
        self.contact_cutoff = 4.5  # A
        self.hbond_cutoff = 3.5
        self.hydrophobic_cutoff = 4.0

        # GCDM atom type mapping (from dataset_info)
        # Typical: C=0, N=1, O=2, F=3, P=4, S=5, Cl=6, Br=7
        self.hydrophobic_atoms = [0, 3, 6, 7]  # C, F, Cl, Br
        self.hbond_donors = [1, 2]  # N, O
        self.hbond_acceptors = [1, 2, 5]  # N, O, S

    def compute_soft_binana(
        self,
        mol_coords: torch.Tensor,      # (n_atoms, 3)
        mol_atom_types: torch.Tensor,  # (n_atoms,) or (n_atoms, n_types)
        pocket_coords: torch.Tensor,   # (n_pocket, 3)
    ) -> torch.Tensor:
        """
        Compute soft BINANA descriptors (differentiable).

        Returns:
            descriptors: (descriptor_dim,)
        """
        descriptors = torch.zeros(self.descriptor_dim, device=mol_coords.device)

        n_atoms = mol_coords.shape[0]
        n_pocket = pocket_coords.shape[0]

        if n_atoms == 0 or n_pocket == 0:
            return descriptors

        # Extract atom type indices
        if mol_atom_types.dim() == 2:
            # One-hot or logits - use softmax probabilities
            atom_probs = F.softmax(mol_atom_types, dim=-1)
            n_types = mol_atom_types.shape[-1]
        else:
            # Convert indices to one-hot
            atom_indices = mol_atom_types.long()
            n_types = 10  # Assume max 10 atom types
            atom_probs = F.one_hot(atom_indices.clamp(0, n_types-1), n_types).float()

        # 1. Pairwise distances (differentiable)
        dist_matrix = torch.cdist(mol_coords, pocket_coords)  # (n_atoms, n_pocket)

        # 2. Soft contact counts (differentiable sigmoid threshold)
        soft_contacts = torch.sigmoid((self.contact_cutoff - dist_matrix) * 2.0)

        # 3. Per-pocket contact density
        per_pocket_contacts = soft_contacts.sum(dim=0)  # (n_pocket,)
        n_desc = min(64, n_pocket)
        descriptors[0:n_desc] = per_pocket_contacts[:n_desc]

        # 4. Hydrophobic contacts (soft)
        hydrophobic_prob = sum(atom_probs[:, i] if i < n_types else 0
                               for i in self.hydrophobic_atoms)
        hydrophobic_contacts = (soft_contacts.T * hydrophobic_prob).sum(dim=1)  # (n_pocket,)
        descriptors[64:64+n_desc] = hydrophobic_contacts[:n_desc]

        # 5. Hydrogen bond approximation
        hbond_donor_prob = sum(atom_probs[:, i] if i < n_types else 0
                               for i in self.hbond_donors)
        hbond_acceptor_prob = sum(atom_probs[:, i] if i < n_types else 0
                                  for i in self.hbond_acceptors)

        soft_hbonds = torch.sigmoid((self.hbond_cutoff - dist_matrix) * 3.0)
        donor_contacts = (soft_hbonds.T * hbond_donor_prob).sum(dim=1)
        acceptor_contacts = (soft_hbonds.T * hbond_acceptor_prob).sum(dim=1)

        descriptors[128:128+n_desc] = donor_contacts[:n_desc]
        descriptors[192:192+n_desc] = acceptor_contacts[:n_desc]

        # 6. Global distance statistics
        min_distances = dist_matrix.min(dim=1)[0]  # (n_atoms,)
        if len(min_distances) > 0:
            descriptors[n_desc] = min_distances.mean()
            descriptors[n_desc+1] = min_distances.std() if len(min_distances) > 1 else 0
            descriptors[n_desc+2] = torch.sigmoid((3.0 - min_distances) * 2.0).sum()

        return descriptors


class AffinityGuidanceForDiffusion:
    """
    Affinity guidance model optimized for diffusion-time gradient computation.

    Key differences from post-hoc scoring:
    1. Accepts 3D coordinates directly (no SMILES conversion)
    2. Uses soft BINANA descriptors (differentiable)
    3. Computes gradients w.r.t. ligand coordinates
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        descriptor_dim: int = 256
    ):
        self.device = device
        self.descriptor_dim = descriptor_dim

        # Load HNN-Denovo model
        print(f"Loading affinity guidance model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Get state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Use pre-imported HNNDenovo
        if HNNDenovo is None:
            raise ImportError("HNNDenovo model not available - affinity guidance disabled")

        # Infer config from checkpoint
        smiles_vocab = state_dict.get('ligand_encoder.embedding.weight',
                                      torch.zeros(100, 64)).shape[0]
        protein_vocab = state_dict.get('protein_encoder.embedding.weight',
                                       torch.zeros(22, 64)).shape[0]

        config = {
            'smiles_vocab_size': smiles_vocab,
            'protein_vocab_size': protein_vocab,
            'embed_dim': 64,
            'hidden_dim': 256,
            'n_layers': 3,
            'descriptor_dim': descriptor_dim,
            'dropout': 0.1,
        }

        self.model = HNNDenovo(config)

        # Load weights (handle prefix differences)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('model.', '').replace('module.', '')
            cleaned_state_dict[new_key] = v

        self.model.load_state_dict(cleaned_state_dict, strict=False)
        self.model.to(device)
        self.model.eval()

        # CORRECTED normalization (from BindingDB statistics)
        self.affinity_mean = 6.46
        self.affinity_std = 1.38

        # Soft BINANA computer
        self.binana = SoftBINANAComputation(device=device, descriptor_dim=descriptor_dim)

        # Simple SMILES tokenizer (for embedding lookup)
        self.smiles_vocab = self._build_smiles_vocab()
        self.max_smiles_len = 100

        # Protein tokenizer
        self.aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        self.aa_to_idx['X'] = 20
        self.max_protein_len = 500

        print("  ✓ Affinity guidance loaded")
        print(f"  Normalization: mean={self.affinity_mean:.2f}, std={self.affinity_std:.2f}")

    def _build_smiles_vocab(self):
        """Build SMILES vocabulary for tokenization."""
        chars = ['<PAD>', '<UNK>'] + list('CNOSFPClBrI=#@+-.()[]0123456789')
        return {c: i for i, c in enumerate(chars)}

    def compute_affinity_gradient(
        self,
        ligand_coords: torch.Tensor,      # (n_atoms, 3)
        ligand_atom_types: torch.Tensor,  # (n_atoms, n_types) - one-hot logits
        pocket_coords: torch.Tensor,      # (n_pocket, 3)
        pocket_sequence: str,             # Protein sequence
        target_affinity: float = 7.0,     # Target pKd
    ) -> tuple[torch.Tensor, float]:
        """
        Compute gradient of affinity loss w.r.t. ligand coordinates.

        Returns:
            gradient: (n_atoms, 3) - direction to move atoms for higher affinity
            affinity: float - current predicted pKd
        """
        # Enable gradients
        coords = ligand_coords.clone().detach().requires_grad_(True)

        # Compute soft BINANA descriptors (differentiable)
        descriptors = self.binana.compute_soft_binana(
            coords, ligand_atom_types, pocket_coords
        )

        # Encode protein sequence
        protein_tokens = self._encode_protein(pocket_sequence)

        # Create dummy SMILES encoding based on atom types
        # (We can't convert 3D coords to SMILES differentiably)
        ligand_tokens = self._encode_atoms_as_smiles(ligand_atom_types)

        # Forward through model
        # Shape: (1, smiles_len), (1, protein_len), (1, descriptor_dim)
        ligand_batch = ligand_tokens.unsqueeze(0).to(self.device)
        protein_batch = protein_tokens.unsqueeze(0).to(self.device)
        descriptor_batch = descriptors.unsqueeze(0)

        # Model forward
        pred_normalized = self.model(ligand_batch, protein_batch, descriptor_batch)

        # Denormalize
        affinity = pred_normalized.squeeze() * self.affinity_std + self.affinity_mean

        # Loss: maximize affinity (minimize negative)
        # Also add term to push toward target
        loss = -affinity  # + 0.1 * (affinity - target_affinity)**2

        # Compute gradient
        gradient = torch.autograd.grad(
            outputs=loss,
            inputs=coords,
            create_graph=False,
            retain_graph=False
        )[0]

        return gradient, affinity.item()

    def _encode_protein(self, sequence: str) -> torch.Tensor:
        """Encode protein sequence to tokens."""
        tokens = []
        for aa in sequence[:self.max_protein_len]:
            tokens.append(self.aa_to_idx.get(aa.upper(), 20))

        # Pad
        while len(tokens) < self.max_protein_len:
            tokens.append(21)  # PAD token

        return torch.tensor(tokens[:self.max_protein_len], dtype=torch.long)

    def _encode_atoms_as_smiles(self, atom_types: torch.Tensor) -> torch.Tensor:
        """
        Create a pseudo-SMILES encoding from atom types.

        Since we can't differentiably convert 3D to SMILES, we create
        a simple atom-list representation.
        """
        # GCDM atom types: C=0, N=1, O=2, F=3, P=4, S=5, Cl=6, Br=7
        atom_chars = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'C']

        if atom_types.dim() == 2:
            # Logits - get argmax
            atom_indices = atom_types.argmax(dim=-1)
        else:
            atom_indices = atom_types

        # Build token sequence
        tokens = []
        for idx in atom_indices[:self.max_smiles_len]:
            char = atom_chars[idx.item() % len(atom_chars)]
            token = self.smiles_vocab.get(char, 1)  # 1 = UNK
            tokens.append(token)

        # Pad
        while len(tokens) < self.max_smiles_len:
            tokens.append(0)  # PAD

        return torch.tensor(tokens[:self.max_smiles_len], dtype=torch.long)


class GuidedConditionalDDPM(ConditionalDDPM):
    """
    ConditionalDDPM with real-time affinity guidance during sampling.

    Injects gradient-based guidance into the denoising loop to steer
    generation toward high-affinity molecules.
    """

    def __init__(
        self,
        *args,
        affinity_checkpoint: str = None,
        protein_sequence: str = None,
        guidance_scale: float = 1.0,
        guidance_start_fraction: float = 0.0,  # When to start (0.0 = immediately)
        guidance_end_fraction: float = 0.8,    # When to stop (0.8 = last 20% unguided)
        guidance_interval: int = 10,           # Apply every N steps
        target_affinity: float = 7.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.guidance_scale = guidance_scale
        self.guidance_start_fraction = guidance_start_fraction
        self.guidance_end_fraction = guidance_end_fraction
        self.guidance_interval = guidance_interval
        self.target_affinity = target_affinity
        self.protein_sequence = protein_sequence

        # Initialize affinity guidance if checkpoint provided
        self.affinity_guidance = None
        if affinity_checkpoint:
            try:
                self.affinity_guidance = AffinityGuidanceForDiffusion(
                    checkpoint_path=affinity_checkpoint,
                    device=str(next(self.parameters()).device)
                )
                print("✓ Real-time affinity guidance enabled")
                print(f"  Scale: {guidance_scale}")
                print(f"  Steps: {guidance_start_fraction*100:.0f}% → {guidance_end_fraction*100:.0f}%")
                print(f"  Target: {target_affinity} pKd")
            except Exception as e:
                print(f"Failed to load affinity guidance: {e}")
                self.affinity_guidance = None

    @torch.no_grad()
    def sample_given_pocket_guided(
        self,
        pocket,
        num_nodes_lig,
        return_frames=1,
        timesteps=None,
        verbose=True
    ):
        """
        Sample molecules with real-time affinity guidance.

        This is the core method that injects gradients during denoising.
        """
        timesteps = self.T if timesteps is None else timesteps
        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0

        n_samples = len(pocket['size'])
        device = pocket['x'].device

        _, pocket = self.normalize(pocket=pocket)

        xh0_pocket = torch.cat([pocket['x'], pocket['one_hot']], dim=1)

        lig_mask = utils.num_nodes_to_batch_mask(
            n_samples, num_nodes_lig, device)

        # Sample from Normal distribution in the pocket center
        mu_lig_x = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        mu_lig_h = torch.zeros((n_samples, self.atom_nf), device=device)
        mu_lig = torch.cat((mu_lig_x, mu_lig_h), dim=1)[lig_mask]
        sigma = torch.ones_like(pocket['size']).unsqueeze(1)

        z_lig, xh_pocket = self.sample_normal_zero_com(
            mu_lig, xh0_pocket, sigma, lig_mask, pocket['mask'])

        self.assert_mean_zero_with_mask(z_lig[:, :self.n_dims], lig_mask)

        out_lig = torch.zeros((return_frames,) + z_lig.size(),
                              device=z_lig.device)
        out_pocket = torch.zeros((return_frames,) + xh_pocket.size(),
                                 device=device)

        # Track guidance metrics
        guidance_history = []

        # Compute guidance step range
        guidance_start_step = int(timesteps * (1 - self.guidance_end_fraction))
        guidance_end_step = int(timesteps * (1 - self.guidance_start_fraction))

        if verbose:
            print(f"\nGuided generation: {n_samples} molecules")
            print(f"   Guidance steps: {guidance_start_step} → {guidance_end_step}")

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T
        for s in reversed(range(0, timesteps)):
            s_array = torch.full((n_samples, 1), fill_value=s,
                                 device=z_lig.device)
            t_array = s_array + 1
            s_array = s_array / timesteps
            t_array = t_array / timesteps

            # Standard denoising step
            z_lig, xh_pocket = self.sample_p_zs_given_zt(
                s_array, t_array, z_lig, xh_pocket, lig_mask, pocket['mask'])

            # ========== REAL-TIME GUIDANCE ==========
            if (self.affinity_guidance is not None and
                guidance_start_step <= s < guidance_end_step and
                s % self.guidance_interval == 0):

                z_lig, step_metrics = self._apply_guidance(
                    z_lig, lig_mask, pocket, s, timesteps
                )
                guidance_history.append(step_metrics)

                if verbose and s % 100 == 0:
                    print(f"   Step {s}: pKd={step_metrics['affinity']:.2f}, "
                          f"grad_norm={step_metrics['grad_norm']:.4f}")
            # ========================================

            # Save frame
            if (s * return_frames) % timesteps == 0:
                idx = (s * return_frames) // timesteps
                out_lig[idx], out_pocket[idx] = \
                    self.unnormalize_z(z_lig, xh_pocket)

        # Finally sample p(x, h | z_0)
        x_lig, h_lig, x_pocket, h_pocket = self.sample_p_xh_given_z0(
            z_lig, xh_pocket, lig_mask, pocket['mask'], n_samples)

        self.assert_mean_zero_with_mask(x_lig, lig_mask)

        # Correct CoM drift
        if return_frames == 1:
            max_cog = scatter_add(x_lig, lig_mask, dim=0).abs().max().item()
            if max_cog > 5e-2:
                x_lig, x_pocket = self.remove_mean_batch(
                    x_lig, x_pocket, lig_mask, pocket['mask'])

        out_lig[0] = torch.cat([x_lig, h_lig], dim=1)
        out_pocket[0] = torch.cat([x_pocket, h_pocket], dim=1)

        # Log final guidance stats
        if guidance_history and verbose:
            mean_aff = np.mean([h['affinity'] for h in guidance_history])
            print(f"   Mean guided affinity: {mean_aff:.2f} pKd")

        return out_lig.squeeze(0), out_pocket.squeeze(0), lig_mask, \
               pocket['mask'], guidance_history

    def _apply_guidance(
        self,
        z_lig: torch.Tensor,
        lig_mask: torch.Tensor,
        pocket: dict,
        step: int,
        total_steps: int
    ) -> tuple[torch.Tensor, dict]:
        """
        Apply affinity gradient guidance to ligand representation.

        Args:
            z_lig: (n_atoms_total, x_dims + atom_nf) - ligand [coords, features]
            lig_mask: (n_atoms_total,) - batch indices
            pocket: pocket dict with 'x' coordinates
            step: current timestep
            total_steps: total timesteps

        Returns:
            z_lig_guided: guided ligand representation
            metrics: dict with affinity, gradient info
        """
        device = z_lig.device
        n_samples = int(lig_mask.max().item()) + 1

        all_gradients = []
        all_affinities = []

        # Process each molecule in batch
        for mol_idx in range(n_samples):
            mol_mask = (lig_mask == mol_idx)

            # Extract this molecule's coords and features
            mol_z = z_lig[mol_mask]  # (n_atoms, x_dims + atom_nf)
            mol_coords = mol_z[:, :self.n_dims]  # (n_atoms, 3)
            mol_features = mol_z[:, self.n_dims:]  # (n_atoms, atom_nf)

            # Get pocket coords for this molecule
            pocket_mask = (pocket['mask'] == mol_idx)
            pocket_coords = pocket['x'][pocket_mask]

            try:
                # Compute gradient
                with torch.enable_grad():
                    gradient, affinity = self.affinity_guidance.compute_affinity_gradient(
                        mol_coords,
                        mol_features,
                        pocket_coords,
                        self.protein_sequence or "ACDEFGHIKLMNPQRSTVWY" * 10,  # Dummy if not provided
                        self.target_affinity
                    )

                all_gradients.append(gradient)
                all_affinities.append(affinity)

            except Exception:
                # Fallback: zero gradient
                all_gradients.append(torch.zeros_like(mol_coords))
                all_affinities.append(0.0)

        # Apply gradients with timestep-dependent scaling
        # Stronger guidance early (high t), weaker late (low t)
        t_fraction = step / total_steps
        timestep_scale = t_fraction  # Linear decay

        for mol_idx in range(n_samples):
            mol_mask = (lig_mask == mol_idx)

            # Scale gradient
            grad = all_gradients[mol_idx]
            scaled_grad = grad * self.guidance_scale * timestep_scale

            # Clip gradient to prevent instability
            grad_norm = scaled_grad.norm()
            if grad_norm > 1.0:
                scaled_grad = scaled_grad / grad_norm

            # Apply to coordinates only (not features)
            z_lig[mol_mask, :self.n_dims] = z_lig[mol_mask, :self.n_dims] + scaled_grad

        # Metrics
        metrics = {
            'affinity': np.mean(all_affinities) if all_affinities else 0.0,
            'grad_norm': np.mean([g.norm().item() for g in all_gradients]) if all_gradients else 0.0,
            'timestep': step,
            'scale': timestep_scale * self.guidance_scale
        }

        return z_lig, metrics


def create_guided_model(
    base_model,
    affinity_checkpoint: str,
    protein_sequence: str,
    guidance_scale: float = 1.0,
    guidance_start_fraction: float = 0.0,
    guidance_end_fraction: float = 0.8,
    guidance_interval: int = 10,
    target_affinity: float = 7.0,
) -> GuidedConditionalDDPM:
    """
    Wrap an existing GCDM model with guided sampling.

    This adds real-time affinity guidance to an already-loaded model.
    """
    # Copy model state
    guided_model = GuidedConditionalDDPM.__new__(GuidedConditionalDDPM)
    guided_model.__dict__.update(base_model.__dict__)

    # Add guidance components
    guided_model.guidance_scale = guidance_scale
    guided_model.guidance_start_fraction = guidance_start_fraction
    guided_model.guidance_end_fraction = guidance_end_fraction
    guided_model.guidance_interval = guidance_interval
    guided_model.target_affinity = target_affinity
    guided_model.protein_sequence = protein_sequence

    # Initialize affinity guidance
    try:
        guided_model.affinity_guidance = AffinityGuidanceForDiffusion(
            checkpoint_path=affinity_checkpoint,
            device=str(next(base_model.parameters()).device)
        )
        print("✓ Real-time affinity guidance enabled")
        print(f"  Scale: {guidance_scale}, Target: {target_affinity} pKd")
    except Exception as e:
        print(f"Failed to load affinity guidance: {e}")
        guided_model.affinity_guidance = None

    return guided_model
