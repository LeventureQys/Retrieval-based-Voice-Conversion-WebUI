"""
Model loader for RVC-LoRA inference

This module provides utilities for loading LoRA-enhanced RVC models.
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

# Add parent directory to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lora import LoRAConfig, load_lora_checkpoint, merge_lora_weights

logger = logging.getLogger(__name__)


class LoRAModelLoader:
    """Loader for LoRA-enhanced RVC models.

    Supports loading:
    - Base model + separate LoRA weights
    - Merged model (LoRA weights merged into base)
    - Original RVC models (backward compatible)

    Args:
        device: Device to load models on
        is_half: Whether to use half precision
    """

    def __init__(
        self,
        device: str = "cpu",
        is_half: bool = False,
    ):
        self.device = device
        self.is_half = is_half

    def load_base_model(
        self,
        checkpoint_path: str,
        version: str = "v2",
        f0: bool = True,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load base RVC model without LoRA.

        Args:
            checkpoint_path: Path to model checkpoint
            version: Model version ("v1" or "v2")
            f0: Whether model supports F0

        Returns:
            Tuple of (model, config)
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract config
        if "config" in checkpoint:
            config = checkpoint["config"]
        else:
            raise ValueError("Checkpoint does not contain model config")

        # Import appropriate synthesizer class
        try:
            from infer.lib.infer_pack.models import (
                SynthesizerTrnMs256NSFsid,
                SynthesizerTrnMs768NSFsid,
                SynthesizerTrnMs256NSFsid_nono,
                SynthesizerTrnMs768NSFsid_nono,
            )
        except ImportError:
            raise ImportError(
                "Could not import RVC models. "
                "Make sure you're running from the RVC root directory."
            )

        # Select model class
        if version == "v1":
            if f0:
                ModelClass = SynthesizerTrnMs256NSFsid
            else:
                ModelClass = SynthesizerTrnMs256NSFsid_nono
        else:  # v2
            if f0:
                ModelClass = SynthesizerTrnMs768NSFsid
            else:
                ModelClass = SynthesizerTrnMs768NSFsid_nono

        # Create model
        model = ModelClass(
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
            is_half=self.is_half,
        )

        # Load weights
        if "weight" in checkpoint:
            model.load_state_dict(checkpoint["weight"], strict=False)

        model.to(self.device)
        model.eval()

        if self.is_half:
            model.half()

        return model, {"config": config, "version": version, "f0": f0}

    def load_with_lora(
        self,
        base_model_path: str,
        lora_path: str,
        version: str = "v2",
        f0: bool = True,
        merge: bool = False,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load base model and apply LoRA weights.

        Args:
            base_model_path: Path to base model checkpoint
            lora_path: Path to LoRA weights
            version: Model version
            f0: Whether model supports F0
            merge: Whether to merge LoRA weights into base model

        Returns:
            Tuple of (model, info_dict)
        """
        from models import SynthesizerLoRA

        # Load base model
        base_model, model_info = self.load_base_model(
            base_model_path, version, f0
        )

        # Load LoRA checkpoint
        lora_checkpoint = torch.load(lora_path, map_location=self.device)

        # Get LoRA config
        if "config" in lora_checkpoint:
            lora_config = LoRAConfig.from_dict(lora_checkpoint["config"])
        else:
            # Use default config
            lora_config = LoRAConfig(
                r=8,
                lora_alpha=16,
                target_modules=["ups", "resblocks"],
            )

        # Create SynthesizerLoRA wrapper
        model = SynthesizerLoRA(
            base_synthesizer=base_model,
            lora_config=lora_config,
            freeze_non_lora=False,  # Don't freeze for inference
        )

        # Load LoRA weights
        if "lora_state_dict" in lora_checkpoint:
            model.load_lora_weights(lora_checkpoint["lora_state_dict"])
        elif "state_dict" in lora_checkpoint:
            # Try to extract LoRA weights from full state dict
            lora_weights = {
                k: v for k, v in lora_checkpoint["state_dict"].items()
                if "lora_" in k
            }
            if lora_weights:
                model.load_lora_weights(lora_weights, strict=False)

        # Optionally merge LoRA weights
        if merge:
            merge_lora_weights(model.synthesizer)
            logger.info("LoRA weights merged into base model")

        model.synthesizer.to(self.device)
        model.synthesizer.eval()

        if self.is_half:
            model.synthesizer.half()

        model_info["lora_config"] = lora_config
        model_info["lora_merged"] = merge

        return model, model_info

    def load_merged_model(
        self,
        merged_model_path: str,
        version: str = "v2",
        f0: bool = True,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load a model with LoRA weights already merged.

        Args:
            merged_model_path: Path to merged model checkpoint
            version: Model version
            f0: Whether model supports F0

        Returns:
            Tuple of (model, info_dict)
        """
        # This is the same as loading a regular model
        return self.load_base_model(merged_model_path, version, f0)


def load_model_for_inference(
    model_path: str,
    lora_path: Optional[str] = None,
    device: str = "cpu",
    is_half: bool = False,
    version: str = "v2",
    f0: bool = True,
    merge_lora: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Convenience function to load model for inference.

    Args:
        model_path: Path to base model or merged model
        lora_path: Path to LoRA weights (optional)
        device: Device to load on
        is_half: Whether to use half precision
        version: Model version
        f0: Whether model supports F0
        merge_lora: Whether to merge LoRA weights

    Returns:
        Tuple of (model, info_dict)
    """
    loader = LoRAModelLoader(device=device, is_half=is_half)

    if lora_path is not None:
        return loader.load_with_lora(
            base_model_path=model_path,
            lora_path=lora_path,
            version=version,
            f0=f0,
            merge=merge_lora,
        )
    else:
        return loader.load_base_model(
            checkpoint_path=model_path,
            version=version,
            f0=f0,
        )


def get_model_info(checkpoint_path: str) -> Dict[str, Any]:
    """Get information about a model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint

    Returns:
        Dictionary with model information
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    info = {
        "path": checkpoint_path,
        "keys": list(checkpoint.keys()),
    }

    if "config" in checkpoint:
        config = checkpoint["config"]
        # Check if config is a list (model config) or dict (LoRA config)
        if isinstance(config, list) and len(config) > 16:
            info["config"] = {
                "spec_channels": config[0],
                "segment_size": config[1],
                "inter_channels": config[2],
                "hidden_channels": config[3],
                "resblock": config[9],
                "upsample_rates": config[12],
                "gin_channels": config[16],
            }
        elif isinstance(config, dict):
            # LoRA config
            info["lora_config"] = config

    if "lora_state_dict" in checkpoint:
        info["has_lora"] = True
        info["lora_params"] = len(checkpoint["lora_state_dict"])
    else:
        info["has_lora"] = False

    if "epoch" in checkpoint:
        info["epoch"] = checkpoint["epoch"]

    return info
