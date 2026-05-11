"""
Adaptive multi-objective guidance for drug design.

Implements dynamic weighting strategies to balance:
1. Affinity (pKd) - maximize
2. Docking score - minimize (more negative = better)
3. Synthetic Accessibility (SA) - minimize (lower = easier to synthesize)

Key Innovation: Pareto-aware adaptive weighting that prevents
one objective from dominating at the expense of others.

The problem with static weighting:
- Static weights (e.g., w_affinity=0.4, w_docking=0.4, w_sa=0.2) resulted in
  molecules with good affinity/docking but poor SA (~5.5+)
- SA was under-weighted, so the model prioritized other objectives

Solution: Adaptive strategies that monitor objective satisfaction and
dynamically adjust weights in real-time.

Usage:
    from src.guidance.multi_objective import AdaptiveMultiObjectiveGuidance
    
    guidance = AdaptiveMultiObjectiveGuidance(
        strategy='adaptive_threshold',
        affinity_target=7.0,
        sa_max=3.5
    )
    
    loss, metrics = guidance.compute_combined_loss(
        affinity=torch.tensor([7.2]),
        docking=torch.tensor([-10.5]),
        sa=torch.tensor([3.8]),
        step=100
    )
"""

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""
    
    # Target values (ideal goals)
    affinity_target: float = 7.0      # Target pKd
    docking_target: float = -11.0     # Target docking score (kcal/mol)
    sa_target: float = 3.0            # Target SA score
    
    # Threshold values (hard constraints)
    affinity_min: float = 6.0         # Minimum acceptable pKd
    docking_max: float = -10.0        # Maximum acceptable docking (most negative)
    sa_max: float = 3.5               # Maximum acceptable SA
    
    # Adaptation parameters
    adaptation_rate: float = 0.1      # How fast weights adapt
    initial_weights: dict[str, float] = field(default_factory=lambda: {
        'affinity': 0.33,
        'docking': 0.33,
        'sa': 0.34
    })
    
    # SA-specific parameters (to fix the underweighting problem)
    sa_penalty_multiplier: float = 10.0  # Extra penalty for SA violations
    sa_min_weight: float = 0.25          # Minimum SA weight to ensure it's not ignored


class AdaptiveMultiObjectiveGuidance:
    """
    Adaptive weighting for multi-objective molecular generation.
    
    Strategies implemented:
    1. adaptive_threshold: Adjusts weights based on objective satisfaction
    2. pareto: Pareto-front tracking with Tchebycheff scalarization
    3. soft_constraint: Smooth barrier functions with exponential penalties
    
    Attributes:
        strategy: Optimization strategy name
        weights: Current objective weights
        satisfaction_history: Tracks objective satisfaction over time
        config: MultiObjectiveConfig instance
    """
    
    def __init__(
        self,
        strategy: str = 'adaptive_threshold',
        affinity_target: float = 7.0,
        affinity_min: float = 6.0,
        docking_target: float = -11.0,
        docking_max: float = -10.0,
        sa_target: float = 3.0,
        sa_max: float = 3.5,
        adaptation_rate: float = 0.1
    ):
        """
        Initialize adaptive multi-objective guidance.
        
        Args:
            strategy: 'adaptive_threshold', 'pareto', or 'soft_constraint'
            affinity_target: Ideal affinity to aim for
            affinity_min: Hard minimum threshold
            docking_target: Ideal docking score
            docking_max: Hard maximum threshold
            sa_target: Ideal SA score
            sa_max: Hard maximum SA threshold
            adaptation_rate: Learning rate for weight adaptation
        """
        if strategy not in ['adaptive_threshold', 'pareto', 'soft_constraint']:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.strategy = strategy
        
        # Create configuration
        self.config = MultiObjectiveConfig(
            affinity_target=affinity_target,
            affinity_min=affinity_min,
            docking_target=docking_target,
            docking_max=docking_max,
            sa_target=sa_target,
            sa_max=sa_max,
            adaptation_rate=adaptation_rate
        )
        
        # Initialize weights
        self.weights = self.config.initial_weights.copy()
        
        # Track objective satisfaction over time
        self.satisfaction_history = {
            'affinity': [],
            'docking': [],
            'sa': []
        }
        
        # Track weight history for analysis
        self.weight_history = {
            'affinity': [],
            'docking': [],
            'sa': []
        }
        
        print("Multi-objective guidance initialized:")
        print(f"  Strategy: {strategy}")
        print(f"  Targets: pKd≥{affinity_target}, Docking≤{docking_target}, SA≤{sa_target}")
        print(f"  Thresholds: pKd≥{affinity_min}, Docking≤{docking_max}, SA≤{sa_max}")
        print(f"  Initial weights: {self.weights}")
    
    def compute_combined_loss(
        self,
        affinity: torch.Tensor,
        docking: torch.Tensor,
        sa: torch.Tensor,
        step: int = 0
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute combined multi-objective loss with adaptive weighting.
        
        Args:
            affinity: Predicted pKd values (batch_size,)
            docking: Predicted docking scores (batch_size,)
            sa: Predicted SA scores (batch_size,)
            step: Current diffusion step (for adaptive strategies)
            
        Returns:
            combined_loss: Weighted combination
            metrics: Dict with individual losses and weights
        """
        if self.strategy == 'adaptive_threshold':
            return self._adaptive_threshold_loss(affinity, docking, sa, step)
        elif self.strategy == 'pareto':
            return self._pareto_aware_loss(affinity, docking, sa, step)
        elif self.strategy == 'soft_constraint':
            return self._soft_constraint_loss(affinity, docking, sa, step)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _adaptive_threshold_loss(
        self,
        affinity: torch.Tensor,
        docking: torch.Tensor,
        sa: torch.Tensor,
        step: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Adaptive threshold-based weighting.
        
        Key idea: Once an objective meets its threshold, REDUCE its weight
        and INCREASE weight on objectives that haven't met threshold yet.
        
        This prevents over-optimization of already-satisfied objectives.
        """
        # Ensure tensors are on same device
        device = affinity.device
        
        # Compute how well each objective is satisfied (0=not satisfied, 1=satisfied)
        affinity_satisfaction = torch.mean(
            (affinity >= self.config.affinity_min).float()
        ).item()
        docking_satisfaction = torch.mean(
            (docking <= self.config.docking_max).float()
        ).item()
        sa_satisfaction = torch.mean(
            (sa <= self.config.sa_max).float()
        ).item()
        
        # Store history
        self.satisfaction_history['affinity'].append(affinity_satisfaction)
        self.satisfaction_history['docking'].append(docking_satisfaction)
        self.satisfaction_history['sa'].append(sa_satisfaction)
        
        # CRITICAL INNOVATION: Inverse satisfaction weighting
        # If affinity is satisfied (0.9), reduce its weight (becomes 0.1)
        # If SA is not satisfied (0.3), increase its weight (becomes 0.7)
        unsatisfaction = {
            'affinity': 1.0 - affinity_satisfaction,
            'docking': 1.0 - docking_satisfaction,
            'sa': 1.0 - sa_satisfaction
        }
        
        # Add small epsilon to prevent division by zero
        total_unsatisfaction = sum(unsatisfaction.values()) + 1e-6
        
        # Compute target weights based on unsatisfaction
        target_weights = {
            k: v / total_unsatisfaction for k, v in unsatisfaction.items()
        }
        
        # Ensure SA maintains minimum weight (key fix for SA underweighting)
        target_weights['sa'] = max(target_weights['sa'], self.config.sa_min_weight)
        
        # Renormalize
        total = sum(target_weights.values())
        target_weights = {k: v / total for k, v in target_weights.items()}
        
        # Smoothly adapt weights (don't change too abruptly)
        for key in self.weights:
            self.weights[key] = (
                (1 - self.config.adaptation_rate) * self.weights[key] +
                self.config.adaptation_rate * target_weights[key]
            )
        
        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        # Store weight history
        for key in self.weights:
            self.weight_history[key].append(self.weights[key])
        
        # Compute individual losses
        # Affinity: want to MAXIMIZE (so loss is negative of affinity)
        affinity_loss = torch.mean((self.config.affinity_target - affinity) ** 2)
        
        # Docking: want to MINIMIZE (already negative, so minimize means make more negative)
        docking_loss = torch.mean((docking - self.config.docking_target) ** 2)
        
        # SA: want to MINIMIZE
        sa_loss = torch.mean((sa - self.config.sa_target) ** 2)
        
        # Combined loss with adaptive weights
        weights_tensor = torch.tensor(
            [self.weights['affinity'], self.weights['docking'], self.weights['sa']],
            device=device
        )
        losses_tensor = torch.stack([affinity_loss, docking_loss, sa_loss])
        
        combined_loss = torch.sum(weights_tensor * losses_tensor)
        
        metrics = {
            'affinity_loss': affinity_loss.item(),
            'docking_loss': docking_loss.item(),
            'sa_loss': sa_loss.item(),
            'combined_loss': combined_loss.item(),
            'w_affinity': self.weights['affinity'],
            'w_docking': self.weights['docking'],
            'w_sa': self.weights['sa'],
            'affinity_sat': affinity_satisfaction,
            'docking_sat': docking_satisfaction,
            'sa_sat': sa_satisfaction,
            'affinity_mean': affinity.mean().item(),
            'docking_mean': docking.mean().item(),
            'sa_mean': sa.mean().item()
        }
        
        return combined_loss, metrics
    
    def _soft_constraint_loss(
        self,
        affinity: torch.Tensor,
        docking: torch.Tensor,
        sa: torch.Tensor,
        step: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Soft constraint approach with barrier functions.
        
        Uses smooth penalty functions that become steep near thresholds.
        SA violations get extra-strong penalties (10x multiplier).
        """
        device = affinity.device
        
        # Affinity: exponential penalty if below threshold
        affinity_violation = torch.clamp(self.config.affinity_min - affinity, min=0)
        affinity_penalty = torch.exp(affinity_violation) - 1
        affinity_loss = torch.mean(affinity_penalty) + torch.mean(
            (self.config.affinity_target - affinity) ** 2
        )
        
        # Docking: penalty if above threshold (less negative)
        docking_violation = torch.clamp(docking - self.config.docking_max, min=0)
        docking_penalty = torch.exp(docking_violation) - 1
        docking_loss = torch.mean(docking_penalty) + torch.mean(
            (docking - self.config.docking_target) ** 2
        )
        
        # SA: STRONG penalty if above threshold (THIS IS KEY!)
        sa_violation = torch.clamp(sa - self.config.sa_max, min=0)
        # 10x stronger penalty for SA violations
        sa_penalty = self.config.sa_penalty_multiplier * (torch.exp(sa_violation) - 1)
        sa_loss = torch.mean(sa_penalty) + torch.mean(
            (sa - self.config.sa_target) ** 2
        )
        
        # Equal base weighting, but penalties handle constraint satisfaction
        combined_loss = affinity_loss + docking_loss + sa_loss
        
        metrics = {
            'affinity_loss': affinity_loss.item(),
            'docking_loss': docking_loss.item(),
            'sa_loss': sa_loss.item(),
            'combined_loss': combined_loss.item(),
            'affinity_penalty': torch.mean(affinity_penalty).item(),
            'docking_penalty': torch.mean(docking_penalty).item(),
            'sa_penalty': torch.mean(sa_penalty).item(),
            'affinity_mean': affinity.mean().item(),
            'docking_mean': docking.mean().item(),
            'sa_mean': sa.mean().item()
        }
        
        return combined_loss, metrics
    
    def _pareto_aware_loss(
        self,
        affinity: torch.Tensor,
        docking: torch.Tensor,
        sa: torch.Tensor,
        step: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Pareto-optimal multi-objective optimization.
        
        Uses scalarization with dynamically adjusted weights to explore
        the Pareto front and find non-dominated solutions.
        """
        device = affinity.device
        
        # Normalize objectives to [0, 1] range for fair comparison
        # Affinity: higher is better, so (current - min) / (target - min)
        affinity_norm = (affinity - self.config.affinity_min) / (
            self.config.affinity_target - self.config.affinity_min + 1e-6
        )
        affinity_norm = torch.clamp(affinity_norm, 0, 1)
        
        # Docking: lower is better, so (max - current) / (max - target)
        docking_norm = (self.config.docking_max - docking) / (
            self.config.docking_max - self.config.docking_target + 1e-6
        )
        docking_norm = torch.clamp(docking_norm, 0, 1)
        
        # SA: lower is better, so (max - current) / (max - target)
        sa_norm = (self.config.sa_max - sa) / (
            self.config.sa_max - self.config.sa_target + 1e-6
        )
        sa_norm = torch.clamp(sa_norm, 0, 1)
        
        # Compute achievement of each objective (1 = fully achieved)
        achievements = {
            'affinity': torch.mean(affinity_norm).item(),
            'docking': torch.mean(docking_norm).item(),
            'sa': torch.mean(sa_norm).item()
        }
        
        # Store satisfaction
        self.satisfaction_history['affinity'].append(achievements['affinity'])
        self.satisfaction_history['docking'].append(achievements['docking'])
        self.satisfaction_history['sa'].append(achievements['sa'])
        
        # Adjust weights to favor under-performing objectives
        min_achievement = min(achievements.values())
        
        for key in self.weights:
            if achievements[key] < min_achievement + 0.1:  # Within 10% of worst
                self.weights[key] = min(
                    self.weights[key] + self.config.adaptation_rate, 0.6
                )
            else:
                self.weights[key] = max(
                    self.weights[key] - self.config.adaptation_rate, 0.1
                )
        
        # Ensure SA minimum weight
        self.weights['sa'] = max(self.weights['sa'], self.config.sa_min_weight)
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        
        # Store weight history
        for key in self.weights:
            self.weight_history[key].append(self.weights[key])
        
        # Weighted Tchebycheff scalarization (Pareto method)
        # Convert normalized achievements to losses (1 - norm)
        objectives = torch.stack([
            1 - affinity_norm,
            1 - docking_norm,
            1 - sa_norm
        ], dim=-1)  # (batch, 3)
        
        weights_tensor = torch.tensor(
            [self.weights['affinity'], self.weights['docking'], self.weights['sa']],
            device=device
        )
        
        # Tchebycheff: minimize the worst weighted objective
        # This finds solutions on the Pareto front
        weighted_objectives = weights_tensor.unsqueeze(0) * objectives
        combined_loss = torch.max(weighted_objectives, dim=-1)[0].mean()
        
        metrics = {
            'combined_loss': combined_loss.item(),
            'w_affinity': self.weights['affinity'],
            'w_docking': self.weights['docking'],
            'w_sa': self.weights['sa'],
            'achievement_affinity': achievements['affinity'],
            'achievement_docking': achievements['docking'],
            'achievement_sa': achievements['sa'],
            'affinity_mean': affinity.mean().item(),
            'docking_mean': docking.mean().item(),
            'sa_mean': sa.mean().item()
        }
        
        return combined_loss, metrics
    
    def get_current_weights(self) -> dict[str, float]:
        """Return current weights for logging."""
        return self.weights.copy()
    
    def reset_weights(self):
        """Reset weights to default."""
        self.weights = self.config.initial_weights.copy()
        self.satisfaction_history = {k: [] for k in self.satisfaction_history}
        self.weight_history = {k: [] for k in self.weight_history}
    
    def get_summary(self) -> str:
        """Get summary of optimization state."""
        lines = [
            f"Strategy: {self.strategy}",
            f"Current weights: affinity={self.weights['affinity']:.3f}, "
            f"docking={self.weights['docking']:.3f}, sa={self.weights['sa']:.3f}",
        ]
        
        if self.satisfaction_history['affinity']:
            avg_sat = {
                k: np.mean(v[-100:]) if v else 0.0
                for k, v in self.satisfaction_history.items()
            }
            lines.append(
                f"Recent satisfaction: affinity={avg_sat['affinity']:.2f}, "
                f"docking={avg_sat['docking']:.2f}, sa={avg_sat['sa']:.2f}"
            )
        
        return '\n'.join(lines)
    
    def check_all_satisfied(
        self,
        affinity: float,
        docking: float,
        sa: float
    ) -> tuple[bool, dict[str, bool]]:
        """
        Check if all objectives are satisfied.
        
        Args:
            affinity: pKd value
            docking: Docking score
            sa: SA score
            
        Returns:
            all_satisfied: True if all objectives met
            individual: Dict with individual satisfaction
        """
        individual = {
            'affinity': affinity >= self.config.affinity_min,
            'docking': docking <= self.config.docking_max,
            'sa': sa <= self.config.sa_max
        }
        
        all_satisfied = all(individual.values())
        
        return all_satisfied, individual


class StaticMultiObjectiveGuidance:
    """
    Static weighting baseline for comparison.
    
    This is the approach that led to poor SA scores.
    Included for ablation studies.
    """
    
    def __init__(
        self,
        w_affinity: float = 0.4,
        w_docking: float = 0.4,
        w_sa: float = 0.2,
        affinity_target: float = 7.0,
        docking_target: float = -11.0,
        sa_target: float = 3.0
    ):
        self.weights = {
            'affinity': w_affinity,
            'docking': w_docking,
            'sa': w_sa
        }
        self.affinity_target = affinity_target
        self.docking_target = docking_target
        self.sa_target = sa_target
    
    def compute_combined_loss(
        self,
        affinity: torch.Tensor,
        docking: torch.Tensor,
        sa: torch.Tensor,
        step: int = 0
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss with static weights."""
        affinity_loss = torch.mean((self.affinity_target - affinity) ** 2)
        docking_loss = torch.mean((docking - self.docking_target) ** 2)
        sa_loss = torch.mean((sa - self.sa_target) ** 2)
        
        combined_loss = (
            self.weights['affinity'] * affinity_loss +
            self.weights['docking'] * docking_loss +
            self.weights['sa'] * sa_loss
        )
        
        return combined_loss, {
            'affinity_loss': affinity_loss.item(),
            'docking_loss': docking_loss.item(),
            'sa_loss': sa_loss.item(),
            'combined_loss': combined_loss.item()
        }


# Testing
if __name__ == '__main__':
    print("Testing AdaptiveMultiObjectiveGuidance...")
    
    # Create guidance
    guidance = AdaptiveMultiObjectiveGuidance(
        strategy='adaptive_threshold',
        affinity_target=7.0,
        sa_max=3.5
    )
    
    # Simulate optimization steps
    print("\nSimulating optimization with different satisfaction levels:")
    
    # Case 1: Good affinity/docking, poor SA (the problem case)
    print("\nCase 1: Good affinity/docking, poor SA")
    affinity = torch.tensor([7.5, 7.2, 6.8])
    docking = torch.tensor([-11.0, -10.5, -10.2])
    sa = torch.tensor([5.5, 5.2, 4.8])  # Too high!
    
    for step in range(5):
        loss, metrics = guidance.compute_combined_loss(affinity, docking, sa, step)
        print(f"  Step {step}: loss={loss.item():.3f}, "
              f"w_aff={metrics['w_affinity']:.3f}, "
              f"w_dock={metrics['w_docking']:.3f}, "
              f"w_sa={metrics['w_sa']:.3f}")
    
    # Case 2: Poor affinity, good docking/SA
    print("\nCase 2: Poor affinity, good docking/SA")
    guidance.reset_weights()
    affinity = torch.tensor([5.5, 5.2, 5.8])  # Below threshold
    docking = torch.tensor([-11.0, -11.5, -10.8])
    sa = torch.tensor([2.8, 3.0, 3.2])
    
    for step in range(5):
        loss, metrics = guidance.compute_combined_loss(affinity, docking, sa, step)
        print(f"  Step {step}: loss={loss.item():.3f}, "
              f"w_aff={metrics['w_affinity']:.3f}, "
              f"w_dock={metrics['w_docking']:.3f}, "
              f"w_sa={metrics['w_sa']:.3f}")
    
    # Test soft constraint strategy
    print("\nTesting soft_constraint strategy:")
    guidance_soft = AdaptiveMultiObjectiveGuidance(
        strategy='soft_constraint',
        affinity_target=7.0,
        sa_max=3.5
    )
    
    affinity = torch.tensor([7.5, 7.2, 6.8])
    docking = torch.tensor([-11.0, -10.5, -10.2])
    sa = torch.tensor([5.5, 5.2, 4.8])
    
    loss, metrics = guidance_soft.compute_combined_loss(affinity, docking, sa, 0)
    print(f"  SA penalty: {metrics['sa_penalty']:.3f}")
    print(f"  Combined loss: {loss.item():.3f}")
    
    print("\n✓ Tests completed")
