"""Unit tests for GradientProcessor — no GPU, no Vina required."""
import torch

from kinetidiff.gcdm.equivariant_diffusion.vina_gradient_guidance import GradientProcessor


def test_gradient_processor_output_shape():
    proc = GradientProcessor()
    grad = torch.randn(5, 3)
    out = proc.process(grad)
    assert out.shape == grad.shape


def test_gradient_processor_tanh_bounds():
    proc = GradientProcessor()
    grad = torch.full((10, 3), 1000.0)
    out = proc.process(grad)
    assert (out.abs() < 1.0).all(), "tanh should bound output to (-1, 1)"


def test_gradient_processor_running_stats_converge():
    proc = GradientProcessor(momentum=0.1)
    constant_grad = torch.ones(5, 3) * 2.0
    for _ in range(200):
        proc.process(constant_grad)
    mean_val = proc.running_mean.mean().item()
    assert abs(mean_val - 2.0) < 0.1, (
        f"running_mean mean={mean_val:.4f}, expected ~2.0"
    )


def test_gradient_processor_zero_grad():
    proc = GradientProcessor()
    grad = torch.zeros(5, 3)
    out = proc.process(grad)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)
