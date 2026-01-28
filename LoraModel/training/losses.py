"""
Loss functions for RVC-LoRA training

This module provides loss functions for training LoRA-enhanced RVC models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


def feature_loss(fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
    """Feature matching loss between real and generated feature maps.

    Compares intermediate features from the discriminator to encourage
    the generator to produce realistic features at multiple scales.

    Args:
        fmap_r: List of feature maps from real audio
        fmap_g: List of feature maps from generated audio

    Returns:
        Feature matching loss value
    """
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(
    disc_real_outputs: List[torch.Tensor],
    disc_generated_outputs: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """Discriminator loss using least squares GAN formulation.

    Args:
        disc_real_outputs: Discriminator outputs for real audio
        disc_generated_outputs: Discriminator outputs for generated audio

    Returns:
        Tuple of (total_loss, real_losses, generated_losses)
    """
    loss = 0
    r_losses = []
    g_losses = []

    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(
    disc_outputs: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Generator adversarial loss using least squares GAN formulation.

    Args:
        disc_outputs: Discriminator outputs for generated audio

    Returns:
        Tuple of (total_loss, individual_losses)
    """
    loss = 0
    gen_losses = []

    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor
) -> torch.Tensor:
    """KL divergence loss for VAE posterior encoder.

    Computes KL divergence between the posterior distribution q(z|x)
    and the prior distribution p(z).

    Args:
        z_p: Sampled latent from posterior
        logs_q: Log variance of posterior
        m_p: Mean of prior
        logs_p: Log variance of prior
        z_mask: Mask for valid positions

    Returns:
        KL divergence loss value
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)

    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)

    return l


def mel_spectrogram_loss(
    y_mel: torch.Tensor,
    y_g_hat_mel: torch.Tensor
) -> torch.Tensor:
    """L1 loss between mel spectrograms.

    Args:
        y_mel: Mel spectrogram of real audio
        y_g_hat_mel: Mel spectrogram of generated audio

    Returns:
        L1 loss value
    """
    return F.l1_loss(y_mel, y_g_hat_mel)


class GeneratorLoss(nn.Module):
    """Combined generator loss for RVC-LoRA training.

    Combines multiple loss components:
    - Adversarial loss (from discriminator)
    - Feature matching loss
    - Mel spectrogram reconstruction loss
    - KL divergence loss (optional, for VAE models)

    Args:
        c_mel: Coefficient for mel loss (default: 45)
        c_kl: Coefficient for KL loss (default: 1.0)
        c_fm: Coefficient for feature matching loss (default: 2.0)
    """

    def __init__(
        self,
        c_mel: float = 45.0,
        c_kl: float = 1.0,
        c_fm: float = 2.0,
    ):
        super().__init__()
        self.c_mel = c_mel
        self.c_kl = c_kl
        self.c_fm = c_fm

    def forward(
        self,
        disc_outputs: List[torch.Tensor],
        fmap_r: List[List[torch.Tensor]],
        fmap_g: List[List[torch.Tensor]],
        y_mel: torch.Tensor,
        y_g_hat_mel: torch.Tensor,
        z_p: Optional[torch.Tensor] = None,
        logs_q: Optional[torch.Tensor] = None,
        m_p: Optional[torch.Tensor] = None,
        logs_p: Optional[torch.Tensor] = None,
        z_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute combined generator loss.

        Args:
            disc_outputs: Discriminator outputs for generated audio
            fmap_r: Feature maps from real audio
            fmap_g: Feature maps from generated audio
            y_mel: Mel spectrogram of real audio
            y_g_hat_mel: Mel spectrogram of generated audio
            z_p, logs_q, m_p, logs_p, z_mask: VAE components (optional)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Adversarial loss
        loss_gen, _ = generator_loss(disc_outputs)

        # Feature matching loss
        loss_fm = feature_loss(fmap_r, fmap_g)

        # Mel spectrogram loss
        loss_mel = mel_spectrogram_loss(y_mel, y_g_hat_mel)

        # KL divergence loss (if VAE components provided)
        loss_kl = torch.tensor(0.0, device=loss_gen.device)
        if all(v is not None for v in [z_p, logs_q, m_p, logs_p, z_mask]):
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)

        # Combined loss
        total_loss = (
            loss_gen +
            self.c_fm * loss_fm +
            self.c_mel * loss_mel +
            self.c_kl * loss_kl
        )

        loss_dict = {
            'loss_gen': loss_gen.item(),
            'loss_fm': loss_fm.item(),
            'loss_mel': loss_mel.item(),
            'loss_kl': loss_kl.item(),
            'loss_total': total_loss.item(),
        }

        return total_loss, loss_dict


class DiscriminatorLoss(nn.Module):
    """Discriminator loss for RVC-LoRA training."""

    def forward(
        self,
        disc_real_outputs: List[torch.Tensor],
        disc_generated_outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, dict]:
        """Compute discriminator loss.

        Args:
            disc_real_outputs: Discriminator outputs for real audio
            disc_generated_outputs: Discriminator outputs for generated audio

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        loss, r_losses, g_losses = discriminator_loss(
            disc_real_outputs, disc_generated_outputs
        )

        loss_dict = {
            'loss_disc': loss.item(),
            'loss_disc_real': sum(r_losses) / len(r_losses),
            'loss_disc_fake': sum(g_losses) / len(g_losses),
        }

        return loss, loss_dict


class LoRAFineTuneLoss(nn.Module):
    """Simplified loss for LoRA fine-tuning.

    For LoRA fine-tuning, we typically use a simpler loss that focuses
    on mel spectrogram reconstruction, optionally with adversarial training.

    Args:
        use_adversarial: Whether to use adversarial loss (default: False)
        c_mel: Coefficient for mel loss (default: 45)
    """

    def __init__(
        self,
        use_adversarial: bool = False,
        c_mel: float = 45.0,
    ):
        super().__init__()
        self.use_adversarial = use_adversarial
        self.c_mel = c_mel

    def forward(
        self,
        y_mel: torch.Tensor,
        y_g_hat_mel: torch.Tensor,
        disc_outputs: Optional[List[torch.Tensor]] = None,
        fmap_r: Optional[List[List[torch.Tensor]]] = None,
        fmap_g: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute LoRA fine-tuning loss.

        Args:
            y_mel: Mel spectrogram of real audio
            y_g_hat_mel: Mel spectrogram of generated audio
            disc_outputs: Discriminator outputs (optional)
            fmap_r, fmap_g: Feature maps (optional)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Mel spectrogram loss (primary)
        loss_mel = mel_spectrogram_loss(y_mel, y_g_hat_mel)
        total_loss = self.c_mel * loss_mel

        loss_dict = {
            'loss_mel': loss_mel.item(),
        }

        # Optional adversarial loss
        if self.use_adversarial and disc_outputs is not None:
            loss_gen, _ = generator_loss(disc_outputs)
            total_loss = total_loss + loss_gen
            loss_dict['loss_gen'] = loss_gen.item()

            if fmap_r is not None and fmap_g is not None:
                loss_fm = feature_loss(fmap_r, fmap_g)
                total_loss = total_loss + loss_fm
                loss_dict['loss_fm'] = loss_fm.item()

        loss_dict['loss_total'] = total_loss.item()

        return total_loss, loss_dict
