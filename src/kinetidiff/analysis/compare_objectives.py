#!/usr/bin/env python3
"""
Multi-Objective Analysis for Generated Molecules.

Analyzes the trade-offs between objectives and visualizes Pareto fronts.

Usage:
    python compare_objectives.py \\
        --input results/molecular_metrics.csv \\
        --output results/objective_analysis/
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

# Optional: matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_metrics(filepath: str) -> pd.DataFrame:
    """Load molecular metrics from CSV."""
    return pd.read_csv(filepath)


def identify_pareto_front(
    df: pd.DataFrame,
    objectives: list[tuple[str, str]]
) -> np.ndarray:
    """
    Identify Pareto-optimal molecules.
    
    Args:
        df: DataFrame with objective columns
        objectives: List of (column_name, direction) where direction is 'max' or 'min'
        
    Returns:
        Boolean array indicating Pareto-optimal rows
    """
    n = len(df)
    is_pareto = np.ones(n, dtype=bool)
    
    # Extract objective values (convert to maximize format)
    obj_values = np.zeros((n, len(objectives)))
    for i, (col, direction) in enumerate(objectives):
        values = df[col].values
        if direction == 'min':
            values = -values
        obj_values[:, i] = values
    
    # Check Pareto dominance
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # Check if j dominates i
            if (all(obj_values[j] >= obj_values[i]) and
                any(obj_values[j] > obj_values[i])):
                is_pareto[i] = False
                break
    
    return is_pareto


def compute_hypervolume(
    df: pd.DataFrame,
    objectives: list[tuple[str, str]],
    reference_point: list[float]
) -> float:
    """
    Compute hypervolume indicator for multi-objective assessment.
    
    Simplified 2D/3D hypervolume computation.
    """
    # Extract Pareto front
    pareto_mask = identify_pareto_front(df, objectives)
    pareto_df = df[pareto_mask]
    
    if len(pareto_df) == 0:
        return 0.0
    
    # For simplicity, compute dominated hypervolume for 2D case
    if len(objectives) == 2:
        col1, dir1 = objectives[0]
        col2, dir2 = objectives[1]
        
        # Sort by first objective
        values1 = pareto_df[col1].values
        values2 = pareto_df[col2].values
        
        if dir1 == 'min':
            values1 = -values1
        if dir2 == 'min':
            values2 = -values2
        
        # Sort and compute area
        sorted_idx = np.argsort(values1)
        values1 = values1[sorted_idx]
        values2 = values2[sorted_idx]
        
        # Simple hypervolume approximation
        hv = 0.0
        prev_v2 = reference_point[1]
        for v1, v2 in zip(values1, values2):
            hv += (reference_point[0] - v1) * (prev_v2 - v2)
            prev_v2 = v2
        
        return max(hv, 0.0)
    
    return 0.0  # 3D case not implemented


def analyze_objectives(df: pd.DataFrame) -> dict:
    """
    Comprehensive multi-objective analysis.
    
    Args:
        df: DataFrame with objective columns
        
    Returns:
        analysis: Dict with analysis results
    """
    objectives = [
        ('affinity_pkd', 'max'),
        ('docking_score', 'min'),
        ('sa_score', 'min')
    ]
    
    analysis = {
        'n_molecules': len(df),
        'pareto_front': {},
        'correlations': {},
        'objective_statistics': {},
        'trade_offs': {}
    }
    
    # Pareto front analysis
    pareto_mask = identify_pareto_front(df, objectives)
    analysis['pareto_front'] = {
        'n_pareto_optimal': int(pareto_mask.sum()),
        'fraction_pareto': float(pareto_mask.mean()),
        'indices': df.index[pareto_mask].tolist()
    }
    
    # Correlation analysis
    obj_cols = [o[0] for o in objectives]
    for i, col1 in enumerate(obj_cols):
        for col2 in obj_cols[i+1:]:
            if col1 in df.columns and col2 in df.columns:
                corr = df[col1].corr(df[col2])
                analysis['correlations'][f'{col1}_vs_{col2}'] = float(corr)
    
    # Per-objective statistics
    for col, direction in objectives:
        if col in df.columns:
            values = df[col].values
            analysis['objective_statistics'][col] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'direction': direction
            }
    
    # Trade-off analysis
    # Identify molecules that excel in one objective but fail in another
    thresholds = {
        'affinity_pkd': (6.0, 'above'),
        'docking_score': (-10.0, 'below'),
        'sa_score': (3.5, 'below')
    }
    
    trade_off_counts = {}
    for obj1, _ in objectives:
        for obj2, _ in objectives:
            if obj1 >= obj2:
                continue
            
            t1, dir1 = thresholds[obj1]
            t2, dir2 = thresholds[obj2]
            
            if dir1 == 'above':
                pass1 = df[obj1] >= t1
            else:
                pass1 = df[obj1] <= t1
            
            if dir2 == 'above':
                pass2 = df[obj2] >= t2
            else:
                pass2 = df[obj2] <= t2
            
            # Count: pass obj1 but fail obj2
            conflict_count = int((pass1 & ~pass2).sum())
            key = f'{obj1}_good_{obj2}_bad'
            trade_off_counts[key] = conflict_count
    
    analysis['trade_offs'] = trade_off_counts
    
    return analysis


def plot_pareto_front(
    df: pd.DataFrame,
    output_path: str
) -> None:
    """Plot Pareto front visualizations."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plots")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Affinity vs Docking
    ax = axes[0]
    ax.scatter(df['affinity_pkd'], df['docking_score'], alpha=0.6, s=20)
    ax.set_xlabel('Affinity (pKd)')
    ax.set_ylabel('Docking Score (kcal/mol)')
    ax.axhline(-10.0, color='r', linestyle='--', label='Threshold')
    ax.axvline(6.0, color='g', linestyle='--', label='Threshold')
    ax.legend()
    ax.set_title('Affinity vs Docking')
    
    # Affinity vs SA
    ax = axes[1]
    ax.scatter(df['affinity_pkd'], df['sa_score'], alpha=0.6, s=20)
    ax.set_xlabel('Affinity (pKd)')
    ax.set_ylabel('SA Score')
    ax.axhline(3.5, color='r', linestyle='--', label='Threshold')
    ax.axvline(6.0, color='g', linestyle='--', label='Threshold')
    ax.legend()
    ax.set_title('Affinity vs SA')
    
    # Docking vs SA
    ax = axes[2]
    ax.scatter(df['docking_score'], df['sa_score'], alpha=0.6, s=20)
    ax.set_xlabel('Docking Score (kcal/mol)')
    ax.set_ylabel('SA Score')
    ax.axhline(3.5, color='r', linestyle='--', label='SA Threshold')
    ax.axvline(-10.0, color='g', linestyle='--', label='Docking Threshold')
    ax.legend()
    ax.set_title('Docking vs SA')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze multi-objective optimization results'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV with molecular metrics')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading metrics from: {args.input}")
    df = load_metrics(args.input)
    print(f"Loaded {len(df)} molecules")
    
    # Check required columns
    required = ['affinity_pkd', 'docking_score', 'sa_score']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns: {missing}")
        print(f"Available columns: {df.columns.tolist()}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Analyze
    print("Analyzing objectives...")
    analysis = analyze_objectives(df)
    
    # Save analysis
    analysis_path = os.path.join(args.output, 'objective_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis to: {analysis_path}")
    
    # Plot
    if MATPLOTLIB_AVAILABLE and all(c in df.columns for c in required):
        plot_path = os.path.join(args.output, 'pareto_plots.png')
        plot_pareto_front(df, plot_path)
    
    # Print summary
    print("\n" + "=" * 50)
    print("MULTI-OBJECTIVE ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total molecules: {analysis['n_molecules']}")
    print(f"Pareto-optimal: {analysis['pareto_front']['n_pareto_optimal']} "
          f"({analysis['pareto_front']['fraction_pareto']:.1%})")
    
    print("\nObjective Statistics:")
    for obj, stats in analysis['objective_statistics'].items():
        print(f"  {obj}: {stats['mean']:.2f} ± {stats['std']:.2f}")
    
    print("\nCorrelations:")
    for pair, corr in analysis['correlations'].items():
        print(f"  {pair}: {corr:.3f}")
    
    print("\nTrade-offs (count):")
    for trade_off, count in analysis['trade_offs'].items():
        print(f"  {trade_off}: {count}")


if __name__ == '__main__':
    main()
