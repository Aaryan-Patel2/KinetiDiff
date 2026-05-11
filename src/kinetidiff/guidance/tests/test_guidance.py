#!/usr/bin/env python3
"""
Unit tests for GCDM-Modified guidance modules.

Run with:
    python -m pytest tests/test_guidance.py -v
    
Or standalone:
    python tests/test_guidance.py
"""

import sys
from pathlib import Path

import torch

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_multi_objective_adaptive_threshold():
    """Test adaptive threshold multi-objective optimizer."""
    from src.guidance.multi_objective import AdaptiveMultiObjectiveGuidance
    
    guidance = AdaptiveMultiObjectiveGuidance(
        strategy='adaptive_threshold',
        affinity_target=7.0,
        affinity_min=6.0,
        docking_target=-11.0,
        docking_max=-10.0,
        sa_target=3.0,
        sa_max=3.5,
        adaptation_rate=0.1
    )
    
    # Test case 1: Good affinity/docking, poor SA
    affinity = torch.tensor([7.5, 7.2, 6.8])
    docking = torch.tensor([-11.0, -10.5, -10.2])
    sa = torch.tensor([5.5, 5.2, 4.8])  # Above threshold
    
    loss, metrics = guidance.compute_combined_loss(affinity, docking, sa, step=0)
    
    # SA weight should increase since SA is unsatisfied
    assert metrics['w_sa'] > 0.25, "SA weight should be at least min_weight"
    print(f"✓ Adaptive threshold test 1: w_sa={metrics['w_sa']:.3f}")
    
    # Test case 2: All satisfied
    guidance.reset_weights()
    affinity = torch.tensor([7.5, 7.2, 7.0])
    docking = torch.tensor([-11.0, -10.5, -10.2])
    sa = torch.tensor([3.0, 3.2, 3.4])
    
    loss, metrics = guidance.compute_combined_loss(affinity, docking, sa, step=0)
    
    assert metrics['sa_sat'] > 0.8, "SA satisfaction should be high"
    print(f"✓ Adaptive threshold test 2: sa_sat={metrics['sa_sat']:.3f}")
    
    return True


def test_multi_objective_soft_constraint():
    """Test soft constraint multi-objective optimizer."""
    from src.guidance.multi_objective import AdaptiveMultiObjectiveGuidance
    
    guidance = AdaptiveMultiObjectiveGuidance(
        strategy='soft_constraint',
        sa_max=3.5
    )
    
    # Test SA penalty
    affinity = torch.tensor([7.5])
    docking = torch.tensor([-11.0])
    sa_good = torch.tensor([3.0])
    sa_bad = torch.tensor([5.5])
    
    loss_good, metrics_good = guidance.compute_combined_loss(affinity, docking, sa_good, 0)
    loss_bad, metrics_bad = guidance.compute_combined_loss(affinity, docking, sa_bad, 0)
    
    assert metrics_bad['sa_penalty'] > metrics_good['sa_penalty'], \
        "Bad SA should have higher penalty"
    print(f"✓ Soft constraint test: penalty_good={metrics_good['sa_penalty']:.3f}, "
          f"penalty_bad={metrics_bad['sa_penalty']:.3f}")
    
    return True


def test_multi_objective_pareto():
    """Test Pareto-aware multi-objective optimizer."""
    from src.guidance.multi_objective import AdaptiveMultiObjectiveGuidance
    
    guidance = AdaptiveMultiObjectiveGuidance(
        strategy='pareto',
        affinity_target=7.0,
        sa_max=3.5
    )
    
    affinity = torch.tensor([7.5, 6.5, 5.5])
    docking = torch.tensor([-11.0, -10.5, -9.0])
    sa = torch.tensor([3.0, 3.5, 4.5])
    
    loss, metrics = guidance.compute_combined_loss(affinity, docking, sa, 0)
    
    # Should have computed achievements
    assert 'achievement_affinity' in metrics
    assert 'achievement_docking' in metrics
    assert 'achievement_sa' in metrics
    print(f"✓ Pareto test: achievements = aff:{metrics['achievement_affinity']:.3f}, "
          f"dock:{metrics['achievement_docking']:.3f}, sa:{metrics['achievement_sa']:.3f}")
    
    return True


def test_sa_guidance():
    """Test SA scoring guidance."""
    try:
        from rdkit import Chem
    except ImportError:
        print("⚠ RDKit not available, skipping SA test")
        return True
    
    from src.guidance.sa_guidance import SAGuidance
    
    guidance = SAGuidance(target_sa=3.0, max_sa=3.5)
    
    # Test simple vs complex molecules
    simple_smiles = "c1ccccc1"  # Benzene
    complex_smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"  # Ibuprofen
    
    mol_simple = Chem.MolFromSmiles(simple_smiles)
    mol_complex = Chem.MolFromSmiles(complex_smiles)
    
    sa_simple = guidance.score_molecule(mol_simple)
    sa_complex = guidance.score_molecule(mol_complex)
    
    print(f"✓ SA test: benzene={sa_simple.item():.2f}, ibuprofen={sa_complex.item():.2f}")
    
    # Simple should have lower SA than complex
    assert sa_simple.item() < sa_complex.item(), "Benzene should be easier to synthesize"
    
    return True


def test_docking_guidance():
    """Test docking score guidance."""
    try:
        from rdkit import Chem
    except ImportError:
        print("⚠ RDKit not available, skipping docking test")
        return True
    
    from src.guidance.docking_guidance import DockingGuidance
    
    guidance = DockingGuidance(use_fast_approximation=True)
    
    # Test scoring
    test_smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"  # Ibuprofen
    mol = Chem.MolFromSmiles(test_smiles)
    mol = Chem.AddHs(mol)
    
    score = guidance.score_molecule(mol)
    
    print(f"✓ Docking test: ibuprofen fast score = {score.item():.2f}")
    
    # Should be in reasonable range
    assert -15.0 < score.item() < 0.0, "Score should be in typical docking range"
    
    return True


def test_scoring_utils():
    """Test scoring utility functions."""
    from src.utils.scoring_utils import (
        compute_fop_score,
        compute_satisfaction_rates,
        rank_molecules_multi_objective,
    )
    
    # Test FOP score
    score_good = compute_fop_score(
        affinity_pkd=7.5,
        docking_score=-11.0,
        sa_score=3.0
    )
    
    score_bad = compute_fop_score(
        affinity_pkd=5.5,
        docking_score=-8.0,
        sa_score=5.5
    )
    
    assert score_good > score_bad, "Good molecule should have higher FOP score"
    print(f"✓ FOP score test: good={score_good:.3f}, bad={score_bad:.3f}")
    
    # Test ranking
    molecules_data = [
        {'affinity_pkd': 7.5, 'docking_score': -11.0, 'sa_score': 3.0},
        {'affinity_pkd': 5.5, 'docking_score': -8.0, 'sa_score': 5.5},
        {'affinity_pkd': 6.5, 'docking_score': -10.5, 'sa_score': 3.5},
    ]
    
    rankings = rank_molecules_multi_objective(molecules_data, strategy='weighted_sum')
    assert rankings[0][0] == 0, "First molecule should rank highest"
    print(f"✓ Ranking test: top molecule index = {rankings[0][0]}")
    
    # Test satisfaction rates
    rates = compute_satisfaction_rates(molecules_data)
    print(f"✓ Satisfaction rates: affinity={rates['affinity']:.1%}, "
          f"docking={rates['docking']:.1%}, sa={rates['sa']:.1%}")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("GCDM-MODIFIED GUIDANCE TESTS")
    print("=" * 60)
    
    tests = [
        ("Multi-objective (adaptive_threshold)", test_multi_objective_adaptive_threshold),
        ("Multi-objective (soft_constraint)", test_multi_objective_soft_constraint),
        ("Multi-objective (pareto)", test_multi_objective_pareto),
        ("SA guidance", test_sa_guidance),
        ("Docking guidance", test_docking_guidance),
        ("Scoring utilities", test_scoring_utils),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        print(f"\nRunning: {name}")
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
                print("  ✗ FAILED")
        except Exception as e:
            failed += 1
            print(f"  ✗ ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
