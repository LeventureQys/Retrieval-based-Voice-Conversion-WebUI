"""
LoRA Configuration

Defines configuration for LoRA layers and training.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation.

    Attributes:
        r: Rank of LoRA matrices (lower = fewer parameters)
        lora_alpha: Scaling factor for LoRA weights
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
        merge_weights: Whether to merge LoRA weights into base model
        bias: Whether to train bias parameters ("none", "all", "lora_only")
    """

    # LoRA hyperparameters
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Target modules
    target_modules: Optional[List[str]] = None

    # Training options
    merge_weights: bool = False
    bias: str = "none"  # "none", "all", "lora_only"

    # Module-specific ranks (optional)
    ups_rank: Optional[int] = None  # Rank for upsampling layers
    resblock_rank: Optional[int] = None  # Rank for ResBlock layers
    attention_rank: Optional[int] = None  # Rank for attention layers

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.r <= 0:
            raise ValueError(f"r must be positive, got {self.r}")

        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")

        if not 0 <= self.lora_dropout < 1:
            raise ValueError(f"lora_dropout must be in [0, 1), got {self.lora_dropout}")

        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError(f"bias must be 'none', 'all', or 'lora_only', got {self.bias}")

        # Set module-specific ranks if not provided
        if self.ups_rank is None:
            self.ups_rank = self.r
        if self.resblock_rank is None:
            self.resblock_rank = max(4, self.r // 2)  # Use smaller rank for ResBlocks
        if self.attention_rank is None:
            self.attention_rank = max(4, self.r // 2)

    def get_rank_for_module(self, module_name: str) -> int:
        """Get the appropriate rank for a given module.

        Args:
            module_name: Name of the module

        Returns:
            Rank to use for this module
        """
        if "ups" in module_name or "upsample" in module_name:
            return self.ups_rank
        elif "resblock" in module_name or "res_block" in module_name:
            return self.resblock_rank
        elif "attn" in module_name or "attention" in module_name:
            return self.attention_rank
        else:
            return self.r

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "merge_weights": self.merge_weights,
            "bias": self.bias,
            "ups_rank": self.ups_rank,
            "resblock_rank": self.resblock_rank,
            "attention_rank": self.attention_rank,
        }

    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)


# Predefined configurations for different use cases

DEFAULT_CONFIG = LoRAConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules=["ups", "resblocks"],
)

HIGH_QUALITY_CONFIG = LoRAConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    target_modules=["ups", "resblocks", "attn"],
    ups_rank=16,
    resblock_rank=8,
    attention_rank=8,
)

FAST_CONFIG = LoRAConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.0,
    target_modules=["ups"],
    ups_rank=4,
)

BALANCED_CONFIG = LoRAConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["ups", "resblocks"],
    ups_rank=12,
    resblock_rank=6,
)
