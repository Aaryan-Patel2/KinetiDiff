"""Unit tests for _coords_to_rdkit — no GPU, no Vina required."""
import pytest
import torch
from rdkit import Chem

from kinetidiff.gcdm.equivariant_diffusion.vina_gradient_guidance import (
    GCDM_ATOM_MAP,
    VinaGradientGuidance,
)


@pytest.fixture
def guidance_stub(tmp_path):
    dummy_pdbqt = tmp_path / "stub.pdbqt"
    dummy_pdbqt.write_text("")
    return VinaGradientGuidance(
        receptor_pdbqt=str(dummy_pdbqt),
        binding_site_config={
            "center_x": 0.0, "center_y": 0.0, "center_z": 0.0,
            "size_x": 20.0, "size_y": 20.0, "size_z": 20.0,
        },
    )


@pytest.mark.parametrize("atom_type_idx", list(GCDM_ATOM_MAP.keys()))
def test_single_atom_type_produces_valid_mol(atom_type_idx, guidance_stub):
    n = 4
    coords = torch.randn(n, 3)
    atom_types = torch.full((n,), atom_type_idx, dtype=torch.long)
    mol = guidance_stub._coords_to_rdkit(coords, atom_types)
    assert mol is not None, f"atom_type={atom_type_idx} produced None"
    assert mol.GetNumAtoms() == n


def test_molecule_is_connected(guidance_stub):
    coords = torch.randn(8, 3)
    atom_types = torch.tensor([0, 1, 2, 0, 0, 1, 2, 0], dtype=torch.long)
    mol = guidance_stub._coords_to_rdkit(coords, atom_types)
    assert mol is not None
    frags = Chem.rdmolops.GetMolFrags(mol)
    assert len(frags) == 1, f"Expected 1 fragment, got {len(frags)}"


def test_atom_count_preserved(guidance_stub, dummy_coords, dummy_atom_types):
    mol = guidance_stub._coords_to_rdkit(dummy_coords, dummy_atom_types)
    assert mol is not None
    assert mol.GetNumAtoms() == dummy_coords.shape[0]
