"""Unit tests for _distance_guidance — no GPU, no Vina required."""
import pytest
import torch

from kinetidiff.gcdm.equivariant_diffusion.vina_gradient_guidance import VinaGradientGuidance


@pytest.fixture
def guidance_stub(tmp_path):
    dummy_pdbqt = tmp_path / "stub.pdbqt"
    dummy_pdbqt.write_text("")
    return VinaGradientGuidance(
        receptor_pdbqt=str(dummy_pdbqt),
        binding_site_config={
            "center_x": 24.87, "center_y": -12.54, "center_z": 38.40,
            "size_x": 20.0, "size_y": 20.0, "size_z": 20.0,
        },
    )


def test_distance_guidance_points_toward_center(guidance_stub, pocket_center):
    center_tensor = torch.tensor(pocket_center)
    coords = torch.zeros(5, 3)
    coords[:, 0] = 100.0  # far along +x
    ligand_mask = torch.ones(5, dtype=torch.bool)

    grad = guidance_stub._distance_guidance(coords, center_tensor, ligand_mask)
    assert grad[:, 0].mean() > 0, "gradient should push toward pocket center in x"


def test_distance_guidance_output_shape(guidance_stub, dummy_coords, dummy_atom_types):
    pocket_center = torch.tensor([24.87, -12.54, 38.40])
    ligand_mask = torch.ones(dummy_coords.shape[0], dtype=torch.bool)
    grad = guidance_stub._distance_guidance(dummy_coords, pocket_center, ligand_mask)
    assert grad.shape == dummy_coords.shape
