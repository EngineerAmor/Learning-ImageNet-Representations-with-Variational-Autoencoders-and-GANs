"""Convolutional Variational Autoencoder (VAE) from scratch for 64x64 RGB images.

No pretrained weights and no ready-made VAE classes are used.
Input/output images are float32 tensors in NCHW layout scaled to [0, 1].
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Team T12 official class labels (0..9) from the Assignment Card
CLASS_NAMES: list[str] = [
    "terrapin",                # 0: n01667778
    "red-breasted merganser",  # 1: n01855032
    "flat-coated retriever",   # 2: n02099267
    "curly-coated retriever",  # 3: n02099429
    "dhole",                   # 4: n02115913
    "chain saw",               # 5: n03000684
    "crash helmet",            # 6: n03127747
    "harvester",               # 7: n03496892
    "water bottle",            # 8: n04557648
    "butternut squash",        # 9: n07717556
]


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """4x4 stride-2 conv halves spatial resolution: H -> H/2."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.LeakyReLU(0.2, inplace=True),
    )


def deconv_block(c_in: int, c_out: int) -> nn.Sequential:
    """4x4 stride-2 transposed conv doubles spatial resolution: H -> 2H."""
    return nn.Sequential(
        nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    """Maps (B, 3, 64, 64) -> posterior parameters mu (B, D) and logvar (B, D)."""

    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # Spatial progression: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        flat_dim = 256 * 4 * 4  # 4096
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x).flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Maps latent vector z (B, D) -> reconstructed image x_hat (B, 3, 64, 64) in [0, 1]."""

    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        # Spatial progression: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.deconv = nn.Sequential(
            deconv_block(256, 128),
            deconv_block(128, 64),
            deconv_block(64, 32),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # constrains predicted Gaussian mean to [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.deconv(h)


class VAE(nn.Module):
    """Convolutional VAE combining Encoder, Reparameterisation Trick, and Decoder."""

    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    @staticmethod
    def reparameterise(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw z = mu + std * eps with eps ~ N(0, I) so gradients flow through mu and logvar."""
        # Clamp logvar moderately to guard against numerical overflow early in training
        logvar_clamped = torch.clamp(logvar, min=-20.0, max=10.0)
        std = torch.exp(0.5 * logvar_clamped)
        eps = torch.randn(std.shape, device=std.device, dtype=std.dtype, generator=generator)
        return mu + std * eps

    def forward(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar, generator=generator)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    sigma2: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (total_loss, recon_term, kl_term) averaged over batch, summed over dimensions.

    Likelihood assumption:
      Factorised Gaussian p(x | z) = N(x; x_hat(z), sigma2 * I) with fixed variance sigma2 = 0.5.
      Up to an additive constant independent of parameters:
        -log p(x | z) = sum_{c,h,w} (x - x_hat)^2 / (2 * sigma2).
      With sigma2 = 0.5, 2 * sigma2 = 1, so the reconstruction term is exact summed squared error
      over all 3 * 64 * 64 = 12,288 pixel channels (in nats).

    KL term:
      Closed-form KL(N(mu, diag(sigma^2)) || N(0, I)) summed over latent_dim (in nats):
        KL = -0.5 * sum_k (1 + logvar_k - mu_k^2 - exp(logvar_k)).

    With beta = 1.0, total_loss = recon_term + kl_term is the negative ELBO (up to constant).
    """
    logvar_clamped = torch.clamp(logvar, min=-20.0, max=10.0)
    recon_per_img = ((x - x_hat) ** 2).flatten(start_dim=1).sum(dim=1) / (2.0 * sigma2)
    kl_per_img = -0.5 * (1.0 + logvar_clamped - mu.pow(2) - logvar_clamped.exp()).sum(dim=1)
    total = (recon_per_img + beta * kl_per_img).mean()
    return total, recon_per_img.mean(), kl_per_img.mean()
