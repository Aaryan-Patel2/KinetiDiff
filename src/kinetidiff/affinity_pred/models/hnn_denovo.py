"""
HNN-Denovo: Hybrid Neural Network for De Novo Molecular Design

This model combines:
1. CNN encoder for ligand SMILES
2. CNN encoder for protein sequences
3. FFNN encoder for BINANA descriptors
4. Fusion network for affinity prediction

Architecture matches validated model achieving:
- PCC: 0.72
- RMSE: 0.70
- MAE: 0.53

Key features:
- Differentiable end-to-end for diffusion guidance
- Fast inference (<10ms per sample)
- Exposes latent embeddings for downstream use
- Non-Bayesian for training stability
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HNNDenovoConfig:
    """Configuration for HNN-Denovo model."""
    
    # Vocabulary sizes
    smiles_vocab_size: int = 100    # SMILES character vocabulary
    protein_vocab_size: int = 22    # 20 AA + unknown + padding
    
    # Embedding dimensions
    smiles_embed_dim: int = 128
    protein_embed_dim: int = 128
    
    # CNN architecture - Ligand encoder
    ligand_channels: list[int] = field(default_factory=lambda: [128, 128, 256])
    ligand_kernel_sizes: list[int] = field(default_factory=lambda: [3, 3, 3])
    
    # CNN architecture - Protein encoder
    protein_channels: list[int] = field(default_factory=lambda: [64, 128, 256])
    protein_kernel_sizes: list[int] = field(default_factory=lambda: [3, 5, 7])
    
    # FFNN architecture - Descriptor encoder
    descriptor_dim: int = 256       # Input BINANA descriptor dimension
    descriptor_hidden: list[int] = field(default_factory=lambda: [512, 256, 128])
    
    # Fusion architecture
    fusion_hidden: list[int] = field(default_factory=lambda: [1024, 512, 256, 128])
    
    # Regularization
    cnn_dropout: float = 0.1
    ffnn_dropout: float = 0.2
    fusion_dropout: float = 0.2
    
    # Sequence lengths
    max_smiles_length: int = 200
    max_protein_length: int = 1000


class LigandCNNEncoder(nn.Module):
    """
    CNN encoder for ligand SMILES strings.
    
    Architecture:
        Embedding → Conv1D layers → BatchNorm → ReLU → Dropout → GlobalMaxPool
    
    Input: (batch, smiles_length) integer-encoded SMILES
    Output: (batch, ligand_channels[-1]) = (batch, 256)
    """
    
    def __init__(self, config: HNNDenovoConfig):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(
            config.smiles_vocab_size,
            config.smiles_embed_dim,
            padding_idx=0
        )
        
        # CNN layers
        channels = [config.smiles_embed_dim] + config.ligand_channels
        kernels = config.ligand_kernel_sizes
        
        layers = []
        for i in range(len(config.ligand_channels)):
            layers.extend([
                nn.Conv1d(channels[i], channels[i+1], kernels[i], padding=kernels[i]//2),
                nn.BatchNorm1d(channels[i+1]),
                nn.ReLU(inplace=True),
                nn.Dropout(config.cnn_dropout),
            ])
        
        self.cnn = nn.Sequential(*layers)
        
        # Output normalization
        self.output_norm = nn.LayerNorm(config.ligand_channels[-1])
        
        self.output_dim = config.ligand_channels[-1]
    
    def forward(self, smiles: torch.Tensor) -> torch.Tensor:
        """
        Encode SMILES sequences.
        
        Args:
            smiles: (batch, smiles_length) integer tensor
            
        Returns:
            embedding: (batch, output_dim) ligand embedding
        """
        # Embed: (batch, length) -> (batch, length, embed_dim)
        x = self.embedding(smiles)
        
        # Transpose for Conv1d: (batch, embed_dim, length)
        x = x.transpose(1, 2)
        
        # CNN: (batch, channels[-1], length)
        x = self.cnn(x)
        
        # Global max pooling: (batch, channels[-1])
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        
        # Normalize
        x = self.output_norm(x)
        
        return x


class ProteinCNNEncoder(nn.Module):
    """
    CNN encoder for protein sequences.
    
    Architecture:
        Embedding → Conv1D layers (variable kernel sizes) → BatchNorm → ReLU → Dropout → GlobalMaxPool
    
    Input: (batch, protein_length) integer-encoded sequence
    Output: (batch, protein_channels[-1]) = (batch, 256)
    
    Note: No output LayerNorm (matches trained checkpoint)
    """
    
    def __init__(self, config: HNNDenovoConfig):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(
            config.protein_vocab_size,
            config.protein_embed_dim,
            padding_idx=21  # Padding token
        )
        
        # CNN layers with varying kernel sizes for multi-scale features
        channels = [config.protein_embed_dim] + config.protein_channels
        kernels = config.protein_kernel_sizes
        
        layers = []
        for i in range(len(config.protein_channels)):
            layers.extend([
                nn.Conv1d(channels[i], channels[i+1], kernels[i], padding=kernels[i]//2),
                nn.BatchNorm1d(channels[i+1]),
                nn.ReLU(inplace=True),
                nn.Dropout(config.cnn_dropout),
            ])
        
        self.cnn = nn.Sequential(*layers)
        
        # No output_norm for protein encoder (matches trained checkpoint)
        
        self.output_dim = config.protein_channels[-1]
    
    def forward(self, protein_seq: torch.Tensor) -> torch.Tensor:
        """
        Encode protein sequences.
        
        Args:
            protein_seq: (batch, protein_length) integer tensor
            
        Returns:
            embedding: (batch, output_dim) protein embedding
        """
        # Embed: (batch, length) -> (batch, length, embed_dim)
        x = self.embedding(protein_seq)
        
        # Transpose for Conv1d: (batch, embed_dim, length)
        x = x.transpose(1, 2)
        
        # CNN: (batch, channels[-1], length)
        x = self.cnn(x)
        
        # Global max pooling: (batch, channels[-1])
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        
        return x


class DescriptorFFNN(nn.Module):
    """
    Feed-Forward Neural Network for BINANA descriptors.
    
    Non-Bayesian version for stable training.
    Uses LayerNorm for input normalization.
    
    Architecture:
        LayerNorm → Linear layers → LayerNorm → ReLU → Dropout
    
    Input: (batch, descriptor_dim) BINANA features
    Output: (batch, descriptor_hidden[-1]) = (batch, 128)
    """
    
    def __init__(self, config: HNNDenovoConfig):
        super().__init__()
        
        # Input normalization (crucial for stability)
        self.input_norm = nn.LayerNorm(config.descriptor_dim)
        
        # Build FFNN layers
        dims = [config.descriptor_dim] + config.descriptor_hidden
        
        layers = []
        for i in range(len(config.descriptor_hidden)):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.ReLU(inplace=True),
                nn.Dropout(config.ffnn_dropout),
            ])
        
        self.network = nn.Sequential(*layers)
        
        self.output_dim = config.descriptor_hidden[-1]
    
    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        """
        Encode BINANA descriptors.
        
        Args:
            descriptors: (batch, descriptor_dim) feature tensor
            
        Returns:
            embedding: (batch, output_dim) descriptor embedding
        """
        # Normalize input
        x = self.input_norm(descriptors)
        
        # FFNN
        x = self.network(x)
        
        return x


class HNNDenovo(nn.Module):
    """
    Hybrid Neural Network for De Novo Molecular Design.
    
    Combines CNN encoders for ligand/protein with FFNN for BINANA descriptors,
    then fuses representations for affinity prediction.
    
    Key features:
    - Differentiable for gradient-based guidance
    - Exposes latent embeddings for diffusion integration
    - Non-Bayesian for stable training
    
    Total parameters: ~10M
    """
    
    def __init__(self, config: HNNDenovoConfig):
        super().__init__()
        
        self.config = config
        
        # Encoders
        self.ligand_encoder = LigandCNNEncoder(config)
        self.protein_encoder = ProteinCNNEncoder(config)
        self.descriptor_encoder = DescriptorFFNN(config)
        
        # Compute fusion input dimension
        # ligand (256) + protein (256) + descriptor (128) = 640
        fusion_input_dim = (
            self.ligand_encoder.output_dim +
            self.protein_encoder.output_dim +
            self.descriptor_encoder.output_dim
        )
        
        # Fusion network with LayerNorm for stability
        dims = [fusion_input_dim] + config.fusion_hidden
        
        fusion_layers = []
        for i in range(len(config.fusion_hidden)):
            fusion_layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.ReLU(inplace=True),
                nn.Dropout(config.fusion_dropout),
            ])
        
        self.fusion = nn.Sequential(*fusion_layers)
        
        # Output head (simple linear, no activation for regression)
        self.output_head = nn.Linear(config.fusion_hidden[-1], 1)
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
        # Store dimensions for external access
        self.ligand_embed_dim = self.ligand_encoder.output_dim
        self.protein_embed_dim = self.protein_encoder.output_dim
        self.descriptor_embed_dim = self.descriptor_encoder.output_dim
        self.fusion_dim = config.fusion_hidden[-1]
    
    def _init_weights(self, module):
        """Initialize weights with Xavier/Kaiming initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
    
    def encode(
        self,
        ligand_smiles: torch.Tensor,
        protein_seq: torch.Tensor,
        descriptors: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Encode inputs into latent representations.
        
        Useful for diffusion guidance - access intermediate embeddings.
        
        Returns:
            Dict with 'ligand', 'protein', 'descriptor' embeddings
        """
        ligand_embed = self.ligand_encoder(ligand_smiles)
        protein_embed = self.protein_encoder(protein_seq)
        descriptor_embed = self.descriptor_encoder(descriptors)
        
        return {
            'ligand': ligand_embed,
            'protein': protein_embed,
            'descriptor': descriptor_embed,
        }
    
    def forward(
        self,
        ligand_smiles: torch.Tensor,
        protein_seq: torch.Tensor,
        descriptors: torch.Tensor,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass for affinity prediction.
        
        Args:
            ligand_smiles: (batch, smiles_len) integer-encoded SMILES
            protein_seq: (batch, seq_len) integer-encoded amino acids
            descriptors: (batch, descriptor_dim) BINANA features
            return_embeddings: If True, also return intermediate embeddings
            
        Returns:
            prediction: (batch, 1) predicted affinity (normalized)
            embeddings: Dict of intermediate embeddings (if return_embeddings=True)
        """
        # Encode
        embeddings = self.encode(ligand_smiles, protein_seq, descriptors)
        
        # Concatenate embeddings: (batch, 640)
        combined = torch.cat([
            embeddings['ligand'],
            embeddings['protein'],
            embeddings['descriptor'],
        ], dim=1)
        
        # Fusion
        fused = self.fusion(combined)
        embeddings['fused'] = fused
        
        # Predict (no activation - regression output)
        prediction = self.output_head(fused)
        
        if return_embeddings:
            return prediction, embeddings
        
        return prediction
    
    def score_ligand(
        self,
        ligand_embedding: torch.Tensor,
        protein_seq: torch.Tensor,
        descriptors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score a ligand embedding directly (for diffusion guidance).
        
        Used during denoising to guide toward high-affinity molecules.
        
        Args:
            ligand_embedding: (batch, ligand_embed_dim) pre-computed embedding
            protein_seq: (batch, seq_len) protein sequence
            descriptors: (batch, descriptor_dim) BINANA features
            
        Returns:
            score: (batch, 1) predicted affinity
        """
        # Get protein and descriptor embeddings
        protein_embed = self.protein_encoder(protein_seq)
        descriptor_embed = self.descriptor_encoder(descriptors)
        
        # Combine with provided ligand embedding
        combined = torch.cat([
            ligand_embedding,
            protein_embed,
            descriptor_embed,
        ], dim=1)
        
        # Fusion and predict
        fused = self.fusion(combined)
        score = self.output_head(fused)
        
        return score
    
    def get_embedding_dims(self) -> dict[str, int]:
        """Get dimensions of all embeddings."""
        return {
            'ligand': self.ligand_embed_dim,
            'protein': self.protein_embed_dim,
            'descriptor': self.descriptor_embed_dim,
            'fused': self.fusion_dim,
        }


def create_model(config: HNNDenovoConfig | None = None) -> HNNDenovo:
    """
    Create HNN-Denovo model with default or custom config.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized HNNDenovo model
    """
    if config is None:
        config = HNNDenovoConfig()
    
    model = HNNDenovo(config)
    
    # Print model summary
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Created HNN-Denovo model:")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")
    print(f"  Embedding dims: {model.get_embedding_dims()}")
    
    return model


def load_model(checkpoint_path: str, device: str = 'cpu') -> tuple[HNNDenovo, dict]:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        
    Returns:
        model: Loaded HNNDenovo model
        checkpoint: Full checkpoint dict with metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Infer config from state dict
    state_dict = checkpoint['model_state_dict']
    
    # Extract dimensions from weights
    smiles_vocab_size = state_dict['ligand_encoder.embedding.weight'].shape[0]
    protein_vocab_size = state_dict['protein_encoder.embedding.weight'].shape[0]
    descriptor_dim = state_dict['descriptor_encoder.input_norm.weight'].shape[0]
    
    config = HNNDenovoConfig(
        smiles_vocab_size=smiles_vocab_size,
        protein_vocab_size=protein_vocab_size,
        descriptor_dim=descriptor_dim,
    )
    
    model = HNNDenovo(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, checkpoint
