"""
Generator with LoRA support for RVC

This module provides LoRA-enhanced versions of RVC's Generator models.
"""

import math
import logging
from typing import Optional

import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

# Import LoRA components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lora import LoRAConfig, inject_lora, print_lora_info

logger = logging.getLogger(__name__)


class GeneratorLoRA(torch.nn.Module):
    """Generator with LoRA support.

    This is a LoRA-enhanced version of RVC's Generator class.
    LoRA is applied to upsampling layers and ResBlocks for efficient fine-tuning.

    Args:
        initial_channel: Number of input channels
        resblock: ResBlock type ("1" or "2")
        resblock_kernel_sizes: Kernel sizes for ResBlocks
        resblock_dilation_sizes: Dilation sizes for ResBlocks
        upsample_rates: Upsampling rates for each layer
        upsample_initial_channel: Initial channel count for upsampling
        upsample_kernel_sizes: Kernel sizes for upsampling layers
        gin_channels: Number of global conditioning channels (default: 0)
        lora_config: LoRA configuration (optional)
    """

    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
        lora_config: Optional[LoRAConfig] = None,
    ):
        super(GeneratorLoRA, self).__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.lora_config = lora_config

        # Pre-convolution layer
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )

        # Import ResBlock from modules
        # For now, we'll use a simplified version
        # In production, this should import from infer.lib.infer_pack.modules
        from .resblock import ResBlock1, ResBlock2
        resblock_class = ResBlock1 if resblock == "1" else ResBlock2

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        # ResBlocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock_class(ch, k, d))

        # Post-convolution layer
        ch = upsample_initial_channel // (2 ** len(self.ups))
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)

        # Initialize weights
        self.ups.apply(self._init_weights)

        # Conditional layer for speaker embedding
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        # Apply LoRA if config is provided
        if lora_config is not None:
            self._inject_lora()

    def _init_weights(self, m):
        """Initialize weights for upsampling layers."""
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _inject_lora(self):
        """Inject LoRA into the model."""
        print("Injecting LoRA into Generator...")

        # Remove weight_norm before injecting LoRA
        self.remove_weight_norm()

        # Inject LoRA
        inject_lora(self, self.lora_config, target_modules=self.lora_config.target_modules)

        # Print LoRA info
        print_lora_info(self)

    def forward(
        self,
        x: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        n_res: Optional[torch.Tensor] = None,
    ):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time)
            g: Global conditioning tensor (optional)
            n_res: Target resolution (optional)

        Returns:
            Output tensor of shape (batch, 1, time')
        """
        # Interpolate to target resolution if specified
        if n_res is not None:
            assert isinstance(n_res, torch.Tensor)
            n = int(n_res.item())
            if n != x.shape[-1]:
                x = F.interpolate(x, size=n, mode="linear")

        # Pre-convolution
        x = self.conv_pre(x)

        # Add global conditioning
        if g is not None:
            x = x + self.cond(g)

        # Upsampling with ResBlocks
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)

            # Apply ResBlocks and average
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Post-processing
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization from all layers."""
        print("Removing weight_norm from Generator...")

        # Remove from upsampling layers
        for l in self.ups:
            try:
                remove_weight_norm(l)
            except ValueError:
                pass  # Already removed

        # Remove from ResBlocks
        for l in self.resblocks:
            if hasattr(l, 'remove_weight_norm'):
                l.remove_weight_norm()

    def __prepare_scriptable__(self):
        """Prepare model for TorchScript compilation."""
        for l in self.ups:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(l)

        for l in self.resblocks:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    torch.nn.utils.remove_weight_norm(l)

        return self


def load_generator_with_lora(
    checkpoint_path: str,
    lora_config: LoRAConfig,
    device: str = "cpu",
) -> GeneratorLoRA:
    """Load a pre-trained Generator and inject LoRA.

    Args:
        checkpoint_path: Path to the pre-trained model checkpoint
        lora_config: LoRA configuration
        device: Device to load the model on

    Returns:
        Generator model with LoRA injected
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model config
    if "config" in checkpoint:
        config = checkpoint["config"]
        # config format: [spec_channels, segment_size, inter_channels, hidden_channels,
        #                 filter_channels, n_heads, n_layers, kernel_size, p_dropout,
        #                 resblock, resblock_kernel_sizes, resblock_dilation_sizes,
        #                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
        #                 spk_embed_dim, gin_channels, sr]

        initial_channel = config[2]  # inter_channels
        resblock = config[9]
        resblock_kernel_sizes = config[10]
        resblock_dilation_sizes = config[11]
        upsample_rates = config[12]
        upsample_initial_channel = config[13]
        upsample_kernel_sizes = config[14]
        gin_channels = config[16]
    else:
        raise ValueError("Checkpoint does not contain model config")

    # Create model
    model = GeneratorLoRA(
        initial_channel=initial_channel,
        resblock=resblock,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        upsample_rates=upsample_rates,
        upsample_initial_channel=upsample_initial_channel,
        upsample_kernel_sizes=upsample_kernel_sizes,
        gin_channels=gin_channels,
        lora_config=lora_config,
    )

    # Load weights (only base model weights, not LoRA)
    if "weight" in checkpoint:
        model_dict = checkpoint["weight"]
        # Filter out LoRA weights if any
        model_dict = {k: v for k, v in model_dict.items() if "lora_" not in k}
        model.load_state_dict(model_dict, strict=False)

    model.to(device)
    model.eval()

    return model
