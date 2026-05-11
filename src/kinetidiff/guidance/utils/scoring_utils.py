"""
Scoring utilities for GCDM-Modified.

Provides functions for:
- FOP-specific scoring (affinity + kinetics)
- Multi-objective ranking
- Score normalization and combination
"""


import numpy as np


def compute_fop_score(
    affinity_pkd: float,
    docking_score: float,
    sa_score: float,
    koff: float | None = None,
    weights: dict[str, float] | None = None
) -> float:
    """
    Compute FOP-specific composite score.
    
    FOP (Fibrodysplasia Ossificans Progressiva) requires molecules with:
    - Moderate affinity (pKd 7-8)
    - Fast dissociation kinetics (k_off ~0.1-1 s^-1)
    - Good synthetic accessibility
    - Good docking scores
    
    Args:
        affinity_pkd: Predicted pKd value
        docking_score: Docking score (kcal/mol, negative is better)
        sa_score: Synthetic accessibility (1-10, lower is better)
        koff: Dissociation rate (s^-1), if available
        weights: Custom weights for each component
        
    Returns:
        fop_score: Composite FOP suitability score (higher is better)
    """
    if weights is None:
        weights = {
            'affinity': 0.30,
            'kinetics': 0.25,
            'docking': 0.25,
            'sa': 0.20
        }
    
    # Affinity component
    # Optimal range for FOP: pKd 7-8 (moderate affinity)
    # Penalize both too weak (<6) and too strong (>9)
    if 7.0 <= affinity_pkd <= 8.0:
        affinity_score = 1.0
    elif affinity_pkd < 6.0:
        affinity_score = affinity_pkd / 6.0
    elif affinity_pkd < 7.0:
        affinity_score = 0.8 + 0.2 * (affinity_pkd - 6.0)
    elif affinity_pkd > 9.0:
        affinity_score = max(0.5, 1.0 - 0.1 * (affinity_pkd - 9.0))
    else:
        affinity_score = 1.0 - 0.1 * (affinity_pkd - 8.0)
    
    # Kinetics component
    # Optimal k_off: 0.1-1.0 s^-1 (fast dissociation)
    if koff is not None:
        if 0.1 <= koff <= 1.0:
            kinetics_score = 1.0
        elif koff < 0.1:
            kinetics_score = max(0.3, koff / 0.1)
        else:  # koff > 1.0
            kinetics_score = max(0.5, 1.0 - 0.5 * np.log10(koff))
    else:
        # Estimate k_off from pKd using empirical correlation
        # log(k_off) ≈ -0.5 * pKd + 3.0
        estimated_log_koff = -0.5 * affinity_pkd + 3.0
        estimated_koff = 10 ** estimated_log_koff
        
        if 0.1 <= estimated_koff <= 1.0:
            kinetics_score = 1.0
        elif estimated_koff < 0.1:
            kinetics_score = max(0.3, estimated_koff / 0.1)
        else:
            kinetics_score = max(0.5, 1.0 - 0.3 * np.log10(estimated_koff))
    
    # Docking component
    # Good: < -10 kcal/mol
    if docking_score <= -12.0:
        docking_component = 1.0
    elif docking_score <= -10.0:
        docking_component = 0.8 + 0.2 * (-10.0 - docking_score) / 2.0
    elif docking_score <= -8.0:
        docking_component = 0.6 + 0.2 * (-8.0 - docking_score) / 2.0
    else:
        docking_component = max(0.0, 0.6 + 0.1 * (-8.0 - docking_score))
    
    # SA component
    # Good: < 3.5
    if sa_score <= 3.0:
        sa_component = 1.0
    elif sa_score <= 3.5:
        sa_component = 0.9 - 0.2 * (sa_score - 3.0)
    elif sa_score <= 4.5:
        sa_component = 0.7 - 0.3 * (sa_score - 3.5)
    else:
        sa_component = max(0.0, 0.4 - 0.1 * (sa_score - 4.5))
    
    # Combine scores
    fop_score = (
        weights['affinity'] * affinity_score +
        weights['kinetics'] * kinetics_score +
        weights['docking'] * docking_component +
        weights['sa'] * sa_component
    )
    
    return fop_score


def rank_molecules_multi_objective(
    molecules_data: list[dict],
    strategy: str = 'weighted_sum',
    weights: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None
) -> list[tuple[int, float]]:
    """
    Rank molecules by multi-objective criteria.
    
    Args:
        molecules_data: List of dicts with 'affinity_pkd', 'docking_score', 'sa_score'
        strategy: 'weighted_sum', 'pareto', or 'threshold_filter'
        weights: Weights for each objective
        thresholds: Thresholds for filtering
        
    Returns:
        rankings: List of (index, score) tuples, sorted by score descending
    """
    if weights is None:
        weights = {'affinity': 0.4, 'docking': 0.35, 'sa': 0.25}
    
    if thresholds is None:
        thresholds = {'affinity': 6.0, 'docking': -10.0, 'sa': 3.5}
    
    if strategy == 'weighted_sum':
        return _rank_weighted_sum(molecules_data, weights)
    elif strategy == 'pareto':
        return _rank_pareto(molecules_data)
    elif strategy == 'threshold_filter':
        return _rank_threshold_filter(molecules_data, weights, thresholds)
    else:
        raise ValueError(f"Unknown ranking strategy: {strategy}")


def _rank_weighted_sum(
    molecules_data: list[dict],
    weights: dict[str, float]
) -> list[tuple[int, float]]:
    """Rank by weighted sum of normalized objectives."""
    scores = []
    
    for i, data in enumerate(molecules_data):
        # Normalize each objective
        aff_norm = (data.get('affinity_pkd', 0) - 6.0) / 2.0  # Center around 7
        dock_norm = -(data.get('docking_score', 0) + 10.0) / 5.0  # Negative is better
        sa_norm = -(data.get('sa_score', 5.0) - 3.0) / 2.0  # Lower is better
        
        score = (
            weights['affinity'] * aff_norm +
            weights['docking'] * dock_norm +
            weights['sa'] * sa_norm
        )
        
        scores.append((i, score))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores


def _rank_pareto(molecules_data: list[dict]) -> list[tuple[int, float]]:
    """Rank by Pareto dominance."""
    n = len(molecules_data)
    dominance_count = [0] * n
    
    # Count how many solutions dominate each solution
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            # Check if j dominates i
            data_i = molecules_data[i]
            data_j = molecules_data[j]
            
            better_or_equal = (
                data_j.get('affinity_pkd', 0) >= data_i.get('affinity_pkd', 0) and
                data_j.get('docking_score', 0) <= data_i.get('docking_score', 0) and
                data_j.get('sa_score', 5.0) <= data_i.get('sa_score', 5.0)
            )
            
            strictly_better = (
                data_j.get('affinity_pkd', 0) > data_i.get('affinity_pkd', 0) or
                data_j.get('docking_score', 0) < data_i.get('docking_score', 0) or
                data_j.get('sa_score', 5.0) < data_i.get('sa_score', 5.0)
            )
            
            if better_or_equal and strictly_better:
                dominance_count[i] += 1
    
    # Lower dominance count = better (Pareto front has count 0)
    scores = [(i, -dominance_count[i]) for i in range(n)]
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores


def _rank_threshold_filter(
    molecules_data: list[dict],
    weights: dict[str, float],
    thresholds: dict[str, float]
) -> list[tuple[int, float]]:
    """Filter by thresholds, then rank by weighted sum."""
    filtered_indices = []
    
    for i, data in enumerate(molecules_data):
        if (data.get('affinity_pkd', 0) >= thresholds['affinity'] and
            data.get('docking_score', 0) <= thresholds['docking'] and
            data.get('sa_score', 10.0) <= thresholds['sa']):
            filtered_indices.append(i)
    
    if not filtered_indices:
        # No molecules pass all thresholds, fall back to weighted sum
        return _rank_weighted_sum(molecules_data, weights)
    
    # Rank filtered molecules
    filtered_data = [molecules_data[i] for i in filtered_indices]
    rankings = _rank_weighted_sum(filtered_data, weights)
    
    # Map back to original indices
    return [(filtered_indices[r[0]], r[1]) for r in rankings]


def compute_objective_statistics(molecules_data: list[dict]) -> dict:
    """
    Compute statistics for each objective.
    
    Args:
        molecules_data: List of molecule data dicts
        
    Returns:
        statistics: Dict with mean, std, min, max for each objective
    """
    if not molecules_data:
        return {}
    
    stats = {}
    
    for key in ['affinity_pkd', 'docking_score', 'sa_score', 'qed']:
        values = [d.get(key, np.nan) for d in molecules_data]
        values = [v for v in values if not np.isnan(v)]
        
        if values:
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
    
    return stats


def compute_satisfaction_rates(
    molecules_data: list[dict],
    thresholds: dict[str, float] | None = None
) -> dict[str, float]:
    """
    Compute fraction of molecules satisfying each threshold.
    
    Args:
        molecules_data: List of molecule data dicts
        thresholds: Dict with threshold values
        
    Returns:
        rates: Dict with satisfaction rate for each objective
    """
    if not molecules_data:
        return {}
    
    if thresholds is None:
        thresholds = {'affinity': 6.0, 'docking': -10.0, 'sa': 3.5}
    
    n = len(molecules_data)
    
    rates = {
        'affinity': sum(1 for d in molecules_data 
                       if d.get('affinity_pkd', 0) >= thresholds['affinity']) / n,
        'docking': sum(1 for d in molecules_data 
                      if d.get('docking_score', 0) <= thresholds['docking']) / n,
        'sa': sum(1 for d in molecules_data 
                 if d.get('sa_score', 10.0) <= thresholds['sa']) / n,
    }
    
    rates['all'] = sum(
        1 for d in molecules_data
        if (d.get('affinity_pkd', 0) >= thresholds['affinity'] and
            d.get('docking_score', 0) <= thresholds['docking'] and
            d.get('sa_score', 10.0) <= thresholds['sa'])
    ) / n
    
    return rates


# Testing
if __name__ == '__main__':
    print("Testing scoring_utils...")
    
    # Test FOP score
    test_cases = [
        {'affinity': 7.5, 'docking': -11.0, 'sa': 3.0, 'expected': 'high'},
        {'affinity': 5.5, 'docking': -8.0, 'sa': 5.0, 'expected': 'low'},
        {'affinity': 9.5, 'docking': -12.0, 'sa': 2.5, 'expected': 'medium'},
    ]
    
    print("\nFOP Score Tests:")
    for case in test_cases:
        score = compute_fop_score(
            case['affinity'],
            case['docking'],
            case['sa']
        )
        print(f"  pKd={case['affinity']}, dock={case['docking']}, SA={case['sa']}")
        print(f"    -> FOP score: {score:.3f} (expected: {case['expected']})")
    
    # Test multi-objective ranking
    molecules = [
        {'affinity_pkd': 7.5, 'docking_score': -11.0, 'sa_score': 3.0},
        {'affinity_pkd': 8.0, 'docking_score': -10.5, 'sa_score': 4.0},
        {'affinity_pkd': 6.5, 'docking_score': -12.0, 'sa_score': 3.5},
        {'affinity_pkd': 5.5, 'docking_score': -8.0, 'sa_score': 5.0},
    ]
    
    print("\nMulti-objective Ranking:")
    rankings = rank_molecules_multi_objective(molecules, strategy='weighted_sum')
    print("  Weighted sum ranking:")
    for idx, score in rankings:
        print(f"    Mol {idx}: score={score:.3f}")
    
    rankings = rank_molecules_multi_objective(molecules, strategy='pareto')
    print("  Pareto ranking:")
    for idx, score in rankings:
        print(f"    Mol {idx}: dominance_score={score}")
    
    # Test statistics
    print("\nObjective Statistics:")
    stats = compute_objective_statistics(molecules)
    for key, s in stats.items():
        print(f"  {key}: mean={s['mean']:.2f}, std={s['std']:.2f}")
    
    # Test satisfaction rates
    print("\nSatisfaction Rates:")
    rates = compute_satisfaction_rates(molecules)
    for key, rate in rates.items():
        print(f"  {key}: {rate:.1%}")
    
    print("\n✓ Tests completed")
