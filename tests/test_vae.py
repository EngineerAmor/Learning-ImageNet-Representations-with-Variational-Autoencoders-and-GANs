"""Unit tests for the Milestone 1 Convolutional VAE (Member B)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import torch
    from vae.model import VAE, vae_loss
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch is required to run VAE model unit tests")
class TestConvolutionalVAE(unittest.TestCase):
    def test_shapes_and_output_range(self) -> None:
        model = VAE(latent_dim=64)
        x = torch.rand(4, 3, 64, 64)
        x_hat, mu, logvar = model(x)
        self.assertEqual(tuple(mu.shape), (4, 64))
        self.assertEqual(tuple(logvar.shape), (4, 64))
        self.assertEqual(tuple(x_hat.shape), (4, 3, 64, 64))
        self.assertTrue(bool((x_hat >= 0.0).all() and (x_hat <= 1.0).all()))

    def test_reparameterisation_gradient_flow(self) -> None:
        model = VAE(latent_dim=32)
        x = torch.rand(2, 3, 64, 64)
        x_hat, mu, logvar = model(x)
        loss, _, _ = vae_loss(x, x_hat, mu, logvar, beta=1.0, sigma2=0.5)
        loss.backward()
        self.assertIsNotNone(model.encoder.fc_mu.weight.grad)
        self.assertIsNotNone(model.encoder.fc_logvar.weight.grad)
        self.assertGreater(float(model.encoder.fc_mu.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.encoder.fc_logvar.weight.grad.abs().sum()), 0.0)

    def test_kl_is_zero_at_standard_normal_prior(self) -> None:
        x = torch.full((3, 3, 64, 64), 0.5)
        mu = torch.zeros(3, 16)
        logvar = torch.zeros(3, 16)  # sigma^2 = exp(0) = 1
        total, recon, kl = vae_loss(x, x, mu, logvar, beta=1.0, sigma2=0.5)
        self.assertAlmostEqual(float(recon.item()), 0.0, places=6)
        self.assertAlmostEqual(float(kl.item()), 0.0, places=6)
        self.assertAlmostEqual(float(total.item()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
