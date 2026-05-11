"""Unit tests for GCDM_ATOM_MAP completeness."""
from rdkit import Chem

from kinetidiff.gcdm.equivariant_diffusion.vina_gradient_guidance import (
    ATOMIC_NUM_TO_SYMBOL,
    GCDM_ATOM_MAP,
)

EXPECTED_INDICES = set(range(9))
EXPECTED_ATOMIC_NUMS = {6, 7, 8, 9, 15, 16, 17, 35, 53}  # C N O F P S Cl Br I
EXPECTED_ELEMENTS = {"C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


def test_atom_map_has_all_nine_types():
    assert set(GCDM_ATOM_MAP.keys()) == EXPECTED_INDICES, (
        f"Expected indices 0-8, got {sorted(GCDM_ATOM_MAP.keys())}"
    )


def test_atom_map_covers_all_elements():
    assert set(GCDM_ATOM_MAP.values()) == EXPECTED_ATOMIC_NUMS, (
        f"Unexpected atomic nums: {set(GCDM_ATOM_MAP.values())}"
    )


def test_atom_map_symbols_are_valid_rdkit():
    pt = Chem.GetPeriodicTable()
    for atomic_num in GCDM_ATOM_MAP.values():
        symbol = ATOMIC_NUM_TO_SYMBOL[atomic_num]
        result = pt.GetAtomicNumber(symbol)
        assert result == atomic_num, (
            f"RDKit atomic num mismatch for symbol '{symbol}': got {result}, expected {atomic_num}"
        )
