"""
Synthesizer with LoRA support for RVC

This module provides LoRA-enhanced versions of RVC's Synthesizer models.
LoRA is primarily applied to the Generator (dec) component for efficient fine-tuning.
"""

import logging
from typing import Optional, Tuple, List

import torch
from torch import nn
from torch.nn import functional as F

# Import LoRA components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lora import LoRAConfig, inject_lora, print_lora_info, extract_lora_weights

logger = logging.getLogger(__name__)


class SynthesizerLoRA(nn.Module):
    """Synthesizer with LoRA support.

    This is a LoRA-enhanced wrapper for RVC's Synthesizer classes.
    LoRA is applied to the Generator (dec) component for efficient fine-tuning
    while keeping other components (enc_p, enc_q, flow) frozen.

    The class supports all four RVC Synthesizer variants:
    - SynthesizerTrnMs256NSFsid (v1 with F0)
    - SynthesizerTrnMs768NSFsid (v2 with F0)
    - SynthesizerTrnMs256NSFsid_nono (v1 without F0)
    - SynthesizerTrnMs768NSFsid_nono (v2 without F0)

    Args:
        base_synthesizer: Pre-loaded base synthesizer model
        lora_config: LoRA configuration
        freeze_non_lora: Whether to freeze non-LoRA parameters (default: True)
    """

    def __init__(
        self,
        base_synthesizer: nn.Module,
        lora_config: Optional[LoRAConfig] = None,
        freeze_non_lora: bool = True,
    ):
        super(SynthesizerLoRA, self).__init__()

        self.synthesizer = base_synthesizer
        self.lora_config = lora_config
        self.freeze_non_lora = freeze_non_lora
        self._lora_injected = False

        # Copy important attributes from base synthesizer
        self._copy_attributes()

        # Apply LoRA if config is provided
        if lora_config is not None:
            self._inject_lora()

    def _copy_attributes(self):
        """Copy important attributes from base synthesizer."""
        attrs_to_copy = [
            'spec_channels', 'inter_channels', 'hidden_channels',
            'filter_channels', 'n_heads', 'n_layers', 'kernel_size',
            'p_dropout', 'resblock', 'resblock_kernel_sizes',
            'resblock_dilation_sizes', 'upsample_rates',
            'upsample_initial_channel', 'upsample_kernel_sizes',
            'segment_size', 'gin_channels', 'spk_embed_dim'
        ]
        for attr in attrs_to_copy:
            if hasattr(self.synthesizer, attr):
                setattr(self, attr, getattr(self.synthesizer, attr))

    def _inject_lora(self):
        """Inject LoRA into the synthesizer's generator (dec)."""
        if self._lora_injected:
            logger.warning("LoRA already injected, skipping...")
            return

        logger.info("Injecting LoRA into Synthesizer...")

        # Remove weight_norm before injecting LoRA
        self.remove_weight_norm()

        # Inject LoRA into the decoder (Generator)
        if hasattr(self.synthesizer, 'dec'):
            inject_lora(
                self.synthesizer.dec,
                self.lora_config,
                target_modules=self.lora_config.target_modules
            )
            logger.info("LoRA injected into dec (Generator)")

        # Optionally inject into flow (for more expressiveness)
        # This is disabled by default as flow is less critical
        # if hasattr(self.synthesizer, 'flow') and 'flow' in (self.lora_config.target_modules or []):
        #     inject_lora(self.synthesizer.flow, self.lora_config)

        self._lora_injected = True

        # Freeze non-LoRA parameters if requested
        if self.freeze_non_lora:
            self._freeze_non_lora_params()

        # Print LoRA info
        print_lora_info(self.synthesizer)

    def _freeze_non_lora_params(self):
        """Freeze all parameters except LoRA parameters."""
        lora_params = 0
        frozen_params = 0

        for name, param in self.synthesizer.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
                lora_params += param.numel()
            else:
                param.requires_grad = False
                frozen_params += param.numel()

        logger.info(f"Frozen {frozen_params:,} parameters, {lora_params:,} LoRA parameters trainable")

    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.synthesizer.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen")

    def freeze_base_model(self):
        """Freeze base model parameters, keep only LoRA trainable."""
        self._freeze_non_lora_params()

    def remove_weight_norm(self):
        """Remove weight normalization from all layers."""
        if hasattr(self.synthesizer, 'remove_weight_norm'):
            self.synthesizer.remove_weight_norm()

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        ds: Optional[torch.Tensor] = None,
    ):
        """Forward pass for training.

        Args:
            phone: Phone features tensor
            phone_lengths: Phone sequence lengths
            pitch: Pitch tensor (quantized)
            pitchf: Pitch tensor (continuous, for F0 models)
            y: Target spectrogram
            y_lengths: Target sequence lengths
            ds: Speaker ID tensor

        Returns:
            Tuple of (output, ids_slice, x_mask, y_mask, latent_variables)
        """
        return self.synthesizer.forward(
            phone, phone_lengths, pitch, pitchf, y, y_lengths, ds
        )

    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        nsff0: torch.Tensor,
        sid: torch.Tensor,
        skip_head: Optional[torch.Tensor] = None,
        return_length: Optional[torch.Tensor] = None,
        return_length2: Optional[torch.Tensor] = None,
    ):
        """Inference pass.

        Args:
            phone: Phone features tensor
            phone_lengths: Phone sequence lengths
            pitch: Pitch tensor (quantized)
            nsff0: F0 tensor (continuous)
            sid: Speaker ID tensor
            skip_head: Number of frames to skip at the beginning
            return_length: Target output length
            return_length2: Target resolution for decoder

        Returns:
            Tuple of (output, x_mask, latent_variables)
        """
        return self.synthesizer.infer(
            phone, phone_lengths, pitch, nsff0, sid,
            skip_head, return_length, return_length2
        )

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Get all LoRA parameters for optimizer.

        Returns:
            List of LoRA parameters
        """
        lora_params = []
        for name, param in self.synthesizer.named_parameters():
            if 'lora_' in name and param.requires_grad:
                lora_params.append(param)
        return lora_params

    def get_lora_state_dict(self) -> dict:
        """Get state dict containing only LoRA weights.

        Returns:
            Dictionary of LoRA weights
        """
        return extract_lora_weights(self.synthesizer)

    def load_lora_weights(self, lora_state_dict: dict, strict: bool = True):
        """Load LoRA weights from state dict.

        Args:
            lora_state_dict: Dictionary of LoRA weights
            strict: Whether to strictly enforce that the keys match
        """
        current_state = self.synthesizer.state_dict()

        # Filter to only LoRA keys
        lora_keys = {k for k in current_state.keys() if 'lora_' in k}

        if strict:
            missing = lora_keys - set(lora_state_dict.keys())
            unexpected = set(lora_state_dict.keys()) - lora_keys
            if missing:
                raise RuntimeError(f"Missing LoRA keys: {missing}")
            if unexpected:
                raise RuntimeError(f"Unexpected keys: {unexpected}")

        # Update state dict
        for key, value in lora_state_dict.items():
            if key in current_state:
                current_state[key] = value

        self.synthesizer.load_state_dict(current_state, strict=False)
        logger.info(f"Loaded {len(lora_state_dict)} LoRA weights")

    def __prepare_scriptable__(self):
        """Prepare model for TorchScript compilation."""
        if hasattr(self.synthesizer, '__prepare_scriptable__'):
            return self.synthesizer.__prepare_scriptable__()
        return self


def load_synthesizer_with_lora(
    checkpoint_path: str,
    lora_config: LoRAConfig,
    device: str = "cpu",
    version: str = "v2",
    is_half: bool = False,
    f0: bool = True,
) -> SynthesizerLoRA:
    """Load a pre-trained Synthesizer and inject LoRA.

    Args:
        checkpoint_path: Path to the pre-trained model checkpoint
        lora_config: LoRA configuration
        device: Device to load the model on
        version: Model version ("v1" or "v2")
        is_half: Whether to use half precision
        f0: Whether the model supports F0

    Returns:
        Synthesizer model with LoRA injected
    """
    # Import RVC synthesizer classes
    try:
        from infer.lib.infer_pack.models import (
            SynthesizerTrnMs256NSFsid,
            SynthesizerTrnMs768NSFsid,
            SynthesizerTrnMs256NSFsid_nono,
            SynthesizerTrnMs768NSFsid_nono,
        )
    except ImportError:
        raise ImportError(
            "Could not import RVC models. Make sure you're running from the RVC root directory."
        )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config or use default for pretrained models
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        # Default configs for pretrained_v2 models (which don't have config key)
        # Determine sample rate from filename
        filename = os.path.basename(checkpoint_path).lower()
        if "48k" in filename:
            sr = 48000
            if version == "v1":
                config = [
                    1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                    [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 6, 2, 2, 2], 512, [16, 16, 4, 4, 4], 109, 256, 48000
                ]
            else:
                config = [
                    1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                    [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [12, 10, 2, 2], 512, [24, 20, 4, 4], 109, 256, 48000
                ]
        elif "32k" in filename:
            sr = 32000
            if version == "v1":
                config = [
                    513, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                    [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 4, 2, 2, 2], 512, [16, 16, 4, 4, 4], 109, 256, 32000
                ]
            else:
                config = [
                    513, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                    [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 8, 2, 2], 512, [20, 16, 4, 4], 109, 256, 32000
                ]
        else:  # Default to 40k
            sr = 40000
            config = [
                1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000
            ]
        logger.info(f"Using default config for pretrained model (sr={sr}, version={version})")

    # Select appropriate synthesizer class
    if version == "v1":
        if f0:
            SynthesizerClass = SynthesizerTrnMs256NSFsid
        else:
            SynthesizerClass = SynthesizerTrnMs256NSFsid_nono
    else:  # v2
        if f0:
            SynthesizerClass = SynthesizerTrnMs768NSFsid
        else:
            SynthesizerClass = SynthesizerTrnMs768NSFsid_nono

    # Create base synthesizer
    # Config format: [spec_channels, segment_size, inter_channels, hidden_channels,
    #                 filter_channels, n_heads, n_layers, kernel_size, p_dropout,
    #                 resblock, resblock_kernel_sizes, resblock_dilation_sizes,
    #                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
    #                 spk_embed_dim, gin_channels, sr]
    base_synthesizer = SynthesizerClass(
        spec_channels=config[0],
        segment_size=config[1],
        inter_channels=config[2],
        hidden_channels=config[3],
        filter_channels=config[4],
        n_heads=config[5],
        n_layers=config[6],
        kernel_size=config[7],
        p_dropout=config[8],
        resblock=config[9],
        resblock_kernel_sizes=config[10],
        resblock_dilation_sizes=config[11],
        upsample_rates=config[12],
        upsample_initial_channel=config[13],
        upsample_kernel_sizes=config[14],
        spk_embed_dim=config[15],
        gin_channels=config[16],
        sr=config[17] if len(config) > 17 else None,
        is_half=is_half,
    )

    # Load weights (pretrained models use "model" key, trained models use "weight" key)
    if "weight" in checkpoint:
        base_synthesizer.load_state_dict(checkpoint["weight"], strict=False)
    elif "model" in checkpoint:
        base_synthesizer.load_state_dict(checkpoint["model"], strict=False)
    else:
        logger.warning("No weights found in checkpoint, using random initialization")

    # Create LoRA wrapper
    model = SynthesizerLoRA(
        base_synthesizer=base_synthesizer,
        lora_config=lora_config,
        freeze_non_lora=True,
    )

    model.to(device)

    return model


def create_synthesizer_lora_from_pretrained(
    pretrained_model: nn.Module,
    lora_config: LoRAConfig,
    freeze_non_lora: bool = True,
) -> SynthesizerLoRA:
    """Create a SynthesizerLoRA from an already loaded pretrained model.

    Args:
        pretrained_model: Pre-loaded synthesizer model
        lora_config: LoRA configuration
        freeze_non_lora: Whether to freeze non-LoRA parameters

    Returns:
        SynthesizerLoRA wrapper
    """
    return SynthesizerLoRA(
        base_synthesizer=pretrained_model,
        lora_config=lora_config,
        freeze_non_lora=freeze_non_lora,
    )
