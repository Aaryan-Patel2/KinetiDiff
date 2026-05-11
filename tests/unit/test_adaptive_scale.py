"""Unit tests for get_adaptive_scale — no GPU, no Vina required."""
import pytest

from kinetidiff.gcdm.equivariant_diffusion.vina_gradient_guidance import get_adaptive_scale


@pytest.mark.parametrize(
    "timestep, expected",
    [
        (800, 0.1),
        (650, 0.1),
        (601, 0.1),
        (600, 0.3),
        (500, 0.3),
        (401, 0.3),
        (400, 0.7),
        (300, 0.7),
        (201, 0.7),
        (200, 1.5),
        (100, 1.5),
        (0, 1.5),
    ],
)
def test_adaptive_scale_boundaries(timestep, expected):
    scale = get_adaptive_scale(timestep)
    assert scale == pytest.approx(expected), (
        f"timestep={timestep}: expected {expected}, got {scale}"
    )
