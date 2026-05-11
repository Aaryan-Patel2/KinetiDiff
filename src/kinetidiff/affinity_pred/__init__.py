"""
Affinity Prediction Module for FOP Drug Discovery Pipeline

This module implements the HNN-Denovo architecture for protein-ligand
binding affinity prediction, designed to serve as a guidance model
for diffusion-based generative approaches.

Key Components:
- Data preprocessing with PDBBind support
- BINANA descriptor extraction
- CNN encoders for ligand SMILES and protein sequences
- FFNN encoder for BINANA descriptors (non-Bayesian for stability)
- Fusion network for affinity prediction

Architecture follows the HNN-Denovo design achieving:
- PCC: 0.72
- RMSE: 0.70
- MAE: 0.53
"""

__version__ = "1.0.0"
__author__ = "FOP Drug Discovery Team"

from .data.featurizers import BINANADescriptor
from .data.preprocessing import PDBBindConfig, PDBBindDataset
from .models.hnn_denovo import HNNDenovo, HNNDenovoConfig
