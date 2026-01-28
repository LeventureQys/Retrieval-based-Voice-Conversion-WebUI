"""
LoRA Layer Implementation

Implements LoRA (Low-Rank Adaptation) layers for efficient fine-tuning.

Reference: https://arxiv.org/abs/2106.09685
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRALayer:
    """Base class for LoRA layers.

    LoRA decomposes weight updates into low-rank matrices:
        W = W0 + BA
    where B is (d × r) and A is (r × k), with r << min(d, k)
    """

    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        """Initialize LoRA layer.

        Args:
            r: Rank of LoRA matrices
            lora_alpha: Scaling factor
            lora_dropout: Dropout probability
            merge_weights: Whether to merge LoRA weights into base weights
        """
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else lambda x: x
        self.merged = False
        self.merge_weights = merge_weights

    @property
    def scaling(self):
        """Scaling factor for LoRA weights."""
        return self.lora_alpha / self.r


class LoRALinear(nn.Linear, LoRALayer):
    """LoRA-enhanced Linear layer.

    Adds low-rank adaptation to a standard linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
        **kwargs
    ):
        """Initialize LoRA Linear layer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            r: Rank of LoRA matrices
            lora_alpha: Scaling factor
            lora_dropout: Dropout probability
            merge_weights: Whether to merge LoRA weights
            **kwargs: Additional arguments for nn.Linear
        """
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        # LoRA matrices
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.reset_lora_parameters()

        # Freeze base weights
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def reset_lora_parameters(self):
        """Initialize LoRA parameters.

        A is initialized with Kaiming uniform, B is initialized to zero.
        This ensures that at initialization, the LoRA contribution is zero.
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        """Set training mode."""
        nn.Linear.train(self, mode)
        if mode and self.merge_weights and self.merged:
            # Unmerge weights during training
            if self.r > 0:
                self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
        elif not mode and self.merge_weights and not self.merged:
            # Merge weights during inference
            if self.r > 0:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, ..., in_features)

        Returns:
            Output tensor of shape (batch, ..., out_features)
        """
        if self.r > 0 and not self.merged:
            # Standard linear transformation
            result = F.linear(x, self.weight, bias=self.bias)

            # Add LoRA contribution
            lora_out = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            result += lora_out

            return result
        else:
            # Use merged weights or no LoRA
            return F.linear(x, self.weight, bias=self.bias)


class LoRAConv1d(nn.Conv1d, LoRALayer):
    """LoRA-enhanced Conv1d layer.

    Adds low-rank adaptation to a 1D convolutional layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
        **kwargs
    ):
        """Initialize LoRA Conv1d layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolving kernel
            r: Rank of LoRA matrices
            lora_alpha: Scaling factor
            lora_dropout: Dropout probability
            merge_weights: Whether to merge LoRA weights
            **kwargs: Additional arguments for nn.Conv1d
        """
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        # LoRA matrices for convolution
        if r > 0:
            # Flatten kernel dimensions for low-rank decomposition
            # kernel_size might be int or tuple
            k_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
            self.lora_A = nn.Parameter(torch.zeros(r, in_channels * k_size))
            self.lora_B = nn.Parameter(torch.zeros(out_channels, r))
            self.reset_lora_parameters()

        # Freeze base weights
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def reset_lora_parameters(self):
        """Initialize LoRA parameters."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        """Set training mode."""
        nn.Conv1d.train(self, mode)
        if mode and self.merge_weights and self.merged:
            # Unmerge weights during training
            if self.r > 0:
                lora_weight = (self.lora_B @ self.lora_A).view(self.weight.shape)
                self.weight.data -= lora_weight * self.scaling
            self.merged = False
        elif not mode and self.merge_weights and not self.merged:
            # Merge weights during inference
            if self.r > 0:
                lora_weight = (self.lora_B @ self.lora_A).view(self.weight.shape)
                self.weight.data += lora_weight * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, length)

        Returns:
            Output tensor of shape (batch, out_channels, length')
        """
        if self.r > 0 and not self.merged:
            # Standard convolution
            result = F.conv1d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

            # Add LoRA contribution
            # Reshape input for matrix multiplication
            x_unfold = F.unfold(
                x.unsqueeze(2),  # Add height dimension
                kernel_size=(self.kernel_size[0], 1),
                padding=(self.padding[0], 0),
                stride=(self.stride[0], 1),
                dilation=(self.dilation[0], 1),
            ).squeeze(2)  # Remove height dimension

            # Apply LoRA
            lora_out = (
                (self.lora_dropout(x_unfold.transpose(1, 2)) @ self.lora_A.T @ self.lora_B.T)
                .transpose(1, 2)
                * self.scaling
            )

            result += lora_out

            return result
        else:
            # Use merged weights or no LoRA
            return F.conv1d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )


class LoRAConvTranspose1d(nn.ConvTranspose1d, LoRALayer):
    """LoRA-enhanced ConvTranspose1d layer.

    Adds low-rank adaptation to a 1D transposed convolutional layer.
    This is particularly important for RVC's upsampling layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
        **kwargs
    ):
        """Initialize LoRA ConvTranspose1d layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolving kernel
            r: Rank of LoRA matrices
            lora_alpha: Scaling factor
            lora_dropout: Dropout probability
            merge_weights: Whether to merge LoRA weights
            **kwargs: Additional arguments for nn.ConvTranspose1d
        """
        nn.ConvTranspose1d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        # LoRA matrices for transposed convolution
        if r > 0:
            # Flatten kernel dimensions for low-rank decomposition
            # kernel_size might be int or tuple
            k_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
            self.lora_A = nn.Parameter(torch.zeros(r, in_channels * k_size))
            self.lora_B = nn.Parameter(torch.zeros(out_channels, r))
            self.reset_lora_parameters()

        # Freeze base weights
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def reset_lora_parameters(self):
        """Initialize LoRA parameters."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        """Set training mode."""
        nn.ConvTranspose1d.train(self, mode)
        if mode and self.merge_weights and self.merged:
            # Unmerge weights during training
            if self.r > 0:
                lora_weight = (self.lora_B @ self.lora_A).view(self.weight.shape)
                self.weight.data -= lora_weight * self.scaling
            self.merged = False
        elif not mode and self.merge_weights and not self.merged:
            # Merge weights during inference
            if self.r > 0:
                lora_weight = (self.lora_B @ self.lora_A).view(self.weight.shape)
                self.weight.data += lora_weight * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor, output_size: Optional[torch.Size] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, length)
            output_size: Optional output size

        Returns:
            Output tensor of shape (batch, out_channels, length')
        """
        if self.r > 0 and not self.merged:
            # Standard transposed convolution
            result = F.conv_transpose1d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.output_padding,
                self.groups,
                self.dilation,
            )

            # Add LoRA contribution
            # For transposed conv, we need to handle the upsampling
            # Simplified approach: apply LoRA as a linear transformation on flattened input
            batch_size, in_channels, length = x.shape

            # Get kernel size
            k_size = self.kernel_size[0] if isinstance(self.kernel_size, tuple) else self.kernel_size

            # Unfold input to match kernel size
            # This creates patches of size (in_channels * k_size)
            x_unfold = F.unfold(
                x.unsqueeze(2),  # Add height dimension: (B, C, 1, L)
                kernel_size=(1, k_size),
                padding=(0, self.padding[0] if isinstance(self.padding, tuple) else self.padding),
                stride=(1, self.stride[0] if isinstance(self.stride, tuple) else self.stride),
                dilation=(1, self.dilation[0] if isinstance(self.dilation, tuple) else self.dilation),
            ).squeeze(2)  # Remove height dimension: (B, C*k, L')

            # Apply LoRA: (B, C*k, L') -> (B, L', C*k) -> (B, L', out_channels) -> (B, out_channels, L')
            lora_contribution = (
                (self.lora_dropout(x_unfold.transpose(1, 2)) @ self.lora_A.T @ self.lora_B.T)
                .transpose(1, 2)
                * self.scaling
            )

            # Upsample LoRA contribution to match output size if needed
            if result.shape[-1] != lora_contribution.shape[-1]:
                lora_contribution = F.interpolate(
                    lora_contribution,
                    size=result.shape[-1],
                    mode='linear',
                    align_corners=False,
                )

            result += lora_contribution

            return result
        else:
            # Use merged weights or no LoRA
            return F.conv_transpose1d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.output_padding,
                self.groups,
                self.dilation,
            )
