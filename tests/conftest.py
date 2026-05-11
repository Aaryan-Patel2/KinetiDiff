"""Shared fixtures for the kinetidiff test suite."""
from pathlib import Path

import pytest
import torch

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def pocket_pdb_path():
    return FIXTURES / "mini_pocket.pdb"


@pytest.fixture
def pocket_center():
    return (24.87, -12.54, 38.40)


@pytest.fixture
def dummy_coords():
    """5-atom ligand coordinates near pocket center."""
    torch.manual_seed(0)
    return torch.randn(5, 3) + torch.tensor([24.87, -12.54, 38.40])


@pytest.fixture
def dummy_atom_types():
    """Discrete GCDM atom type indices: C, N, O, C, C."""
    return torch.tensor([0, 1, 2, 0, 0], dtype=torch.long)
