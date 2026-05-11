"""
Guidance modules for multi-objective molecular optimization.

This package provides guidance functions for steering GCDM generation toward
high-quality drug candidates that satisfy multiple objectives:
- Binding affinity (pKd)
- Docking score (QuickVina2)
- Synthetic accessibility (SA score)

Key classes:
- AffinityGuidanceModel: HNN-Denovo wrapper for affinity prediction
- DockingGuidance: Fast docking score estimation
- SAGuidance: Synthetic accessibility scoring
- AdaptiveMultiObjectiveGuidance: Adaptive weighting for multi-objective optimization
"""

from .affinity_guidance import AffinityGuidanceModel, load_affinity_guidance
from .docking_guidance import DockingGuidance, load_docking_guidance
from .multi_objective import AdaptiveMultiObjectiveGuidance
from .sa_guidance import SAGuidance, load_sa_guidance

__all__ = [
    'AdaptiveMultiObjectiveGuidance',
    'AffinityGuidanceModel',
    'DockingGuidance',
    'SAGuidance',
    'load_affinity_guidance',
    'load_docking_guidance',
    'load_sa_guidance',
]
