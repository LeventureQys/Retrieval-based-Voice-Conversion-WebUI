"""
RVC-LoRA: LoRA Support for RVC Voice Conversion

This package provides LoRA (Low-Rank Adaptation) implementation for
efficient fine-tuning of RVC voice conversion models.
"""

__version__ = "0.1.0"
__author__ = "RVC-LoRA Contributors"

from .lora_layer import LoRALayer, LoRALinear, LoRAConv1d, LoRAConvTranspose1d
from .lora_utils import (
    inject_lora,
    extract_lora_weights,
    merge_lora_weights,
    mark_only_lora_as_trainable,
    load_lora_weights,
    save_lora_checkpoint,
    load_lora_checkpoint,
    print_lora_info,
    count_lora_parameters,
)
from .lora_config import LoRAConfig

__all__ = [
    "LoRALayer",
    "LoRALinear",
    "LoRAConv1d",
    "LoRAConvTranspose1d",
    "inject_lora",
    "extract_lora_weights",
    "merge_lora_weights",
    "mark_only_lora_as_trainable",
    "load_lora_weights",
    "save_lora_checkpoint",
    "load_lora_checkpoint",
    "print_lora_info",
    "count_lora_parameters",
    "LoRAConfig",
]
