"""
LoRA Utility Functions

Provides utilities for injecting, extracting, and merging LoRA weights.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .lora_layer import LoRALayer, LoRALinear, LoRAConv1d, LoRAConvTranspose1d
from .lora_config import LoRAConfig


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Freeze all parameters except LoRA parameters.

    Args:
        model: Model to modify
        bias: Bias training mode ("none", "all", "lora_only")
    """
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    if bias == "none":
        return
    elif bias == "all":
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = True
    elif bias == "lora_only":
        for module in model.modules():
            if isinstance(module, LoRALayer) and hasattr(module, "bias") and module.bias is not None:
                module.bias.requires_grad = True
    else:
        raise ValueError(f"Invalid bias mode: {bias}")


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get all LoRA parameters from a model.

    Args:
        model: Model to extract parameters from

    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            lora_params.append(param)
    return lora_params


def count_lora_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count LoRA and total parameters.

    Args:
        model: Model to count parameters

    Returns:
        Tuple of (lora_params, total_params)
    """
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    total_params = sum(p.numel() for p in model.parameters())
    return lora_params, total_params


def inject_lora(
    model: nn.Module,
    config: LoRAConfig,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Inject LoRA layers into a model.

    Args:
        model: Model to inject LoRA into
        config: LoRA configuration
        target_modules: List of module names to target (e.g., ["ups", "resblocks"])

    Returns:
        Model with LoRA layers injected
    """
    if target_modules is None:
        target_modules = config.target_modules or []

    def should_inject(name: str) -> bool:
        """Check if a module should have LoRA injected."""
        if not target_modules:
            return False
        return any(target in name for target in target_modules)

    def inject_module(parent_module: nn.Module, name: str, module: nn.Module, full_name: str):
        """Inject LoRA into a single module."""
        if not should_inject(full_name):
            return

        # Get appropriate rank for this module
        rank = config.get_rank_for_module(full_name)

        # Replace with LoRA version
        if isinstance(module, nn.Linear):
            lora_module = LoRALinear(
                in_features=module.in_features,
                out_features=module.out_features,
                r=rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                merge_weights=config.merge_weights,
                bias=module.bias is not None,
            )
            # Copy weights
            lora_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_module.bias.data = module.bias.data.clone()

            setattr(parent_module, name, lora_module)
            print(f"Injected LoRA into Linear: {full_name} (rank={rank})")

        elif isinstance(module, nn.Conv1d) and not isinstance(module, nn.ConvTranspose1d):
            lora_module = LoRAConv1d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                r=rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                merge_weights=config.merge_weights,
                bias=module.bias is not None,
            )
            # Copy weights
            lora_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_module.bias.data = module.bias.data.clone()

            setattr(parent_module, name, lora_module)
            print(f"Injected LoRA into Conv1d: {full_name} (rank={rank})")

        elif isinstance(module, nn.ConvTranspose1d):
            lora_module = LoRAConvTranspose1d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                output_padding=module.output_padding,
                groups=module.groups,
                dilation=module.dilation,
                r=rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                merge_weights=config.merge_weights,
                bias=module.bias is not None,
            )
            # Copy weights
            lora_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_module.bias.data = module.bias.data.clone()

            setattr(parent_module, name, lora_module)
            print(f"Injected LoRA into ConvTranspose1d: {full_name} (rank={rank})")

    # Recursively inject LoRA
    def inject_recursive(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            inject_module(module, name, child, full_name)
            inject_recursive(child, full_name)

    inject_recursive(model)

    # Mark only LoRA parameters as trainable
    mark_only_lora_as_trainable(model, bias=config.bias)

    # Print statistics
    lora_params, total_params = count_lora_parameters(model)
    print(f"\nLoRA injection complete:")
    print(f"  LoRA parameters: {lora_params:,} ({lora_params/total_params*100:.2f}%)")
    print(f"  Total parameters: {total_params:,}")

    return model


def extract_lora_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract LoRA weights from a model.

    Args:
        model: Model with LoRA layers

    Returns:
        Dictionary of LoRA weights
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_state_dict[name] = param.data.clone()
    return lora_state_dict


def load_lora_weights(model: nn.Module, lora_state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    """Load LoRA weights into a model.

    Args:
        model: Model with LoRA layers
        lora_state_dict: Dictionary of LoRA weights

    Returns:
        Model with loaded LoRA weights
    """
    model_state_dict = model.state_dict()
    loaded_count = 0
    for name, param in lora_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name] = param
            loaded_count += 1
        else:
            print(f"Warning: LoRA parameter {name} not found in model")

    # Actually load the updated state dict back into the model
    model.load_state_dict(model_state_dict, strict=False)
    print(f"Loaded {loaded_count} LoRA parameters into model")

    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into base weights.

    After merging, the model no longer needs LoRA layers for inference.

    Args:
        model: Model with LoRA layers

    Returns:
        Model with merged weights
    """
    for module in model.modules():
        if isinstance(module, LoRALayer) and hasattr(module, "merge_weights"):
            if not module.merged:
                module.eval()  # This triggers merging
                module.merge_weights = True
                module.merged = True

    print("LoRA weights merged into base model")
    return model


def unmerge_lora_weights(model: nn.Module) -> nn.Module:
    """Unmerge LoRA weights from base weights.

    Args:
        model: Model with merged LoRA weights

    Returns:
        Model with unmerged weights
    """
    for module in model.modules():
        if isinstance(module, LoRALayer) and hasattr(module, "merge_weights"):
            if module.merged:
                module.train()  # This triggers unmerging
                module.merged = False

    print("LoRA weights unmerged from base model")
    return model


def save_lora_checkpoint(
    model: nn.Module,
    path: str,
    config: LoRAConfig,
    optimizer_state: Optional[Dict] = None,
    epoch: Optional[int] = None,
    additional_info: Optional[Dict] = None,
) -> None:
    """Save LoRA checkpoint.

    Args:
        model: Model with LoRA layers
        path: Path to save checkpoint
        config: LoRA configuration
        optimizer_state: Optional optimizer state
        epoch: Optional epoch number
        additional_info: Optional additional information
    """
    checkpoint = {
        "lora_weights": extract_lora_weights(model),
        "lora_config": config.to_dict(),
    }

    if optimizer_state is not None:
        checkpoint["optimizer_state"] = optimizer_state

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if additional_info is not None:
        checkpoint.update(additional_info)

    torch.save(checkpoint, path)
    print(f"LoRA checkpoint saved to {path}")


def load_lora_checkpoint(
    model: nn.Module,
    path: str,
    load_optimizer: bool = False,
) -> Tuple[nn.Module, Optional[Dict], Optional[int], LoRAConfig]:
    """Load LoRA checkpoint.

    Args:
        model: Model with LoRA layers
        path: Path to checkpoint
        load_optimizer: Whether to load optimizer state

    Returns:
        Tuple of (model, optimizer_state, epoch, config)
    """
    checkpoint = torch.load(path, map_location="cpu")

    # Load LoRA weights
    lora_weights = checkpoint["lora_weights"]
    model = load_lora_weights(model, lora_weights)

    # Load config
    config = LoRAConfig.from_dict(checkpoint["lora_config"])

    # Load optimizer state
    optimizer_state = checkpoint.get("optimizer_state") if load_optimizer else None

    # Load epoch
    epoch = checkpoint.get("epoch")

    print(f"LoRA checkpoint loaded from {path}")
    if epoch is not None:
        print(f"  Epoch: {epoch}")

    return model, optimizer_state, epoch, config


def print_lora_info(model: nn.Module) -> None:
    """Print information about LoRA layers in a model.

    Args:
        model: Model to inspect
    """
    print("\n" + "=" * 60)
    print("LoRA Layer Information")
    print("=" * 60)

    lora_layers = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_layers.append((name, module))

    if not lora_layers:
        print("No LoRA layers found in model")
        return

    print(f"\nFound {len(lora_layers)} LoRA layers:\n")

    for name, module in lora_layers:
        layer_type = type(module).__name__
        rank = module.r
        alpha = module.lora_alpha
        scaling = module.scaling

        # Count parameters
        lora_params = sum(p.numel() for n, p in module.named_parameters() if "lora_" in n)

        print(f"  {name}")
        print(f"    Type: {layer_type}")
        print(f"    Rank: {rank}")
        print(f"    Alpha: {alpha}")
        print(f"    Scaling: {scaling:.4f}")
        print(f"    LoRA params: {lora_params:,}")
        print()

    # Total statistics
    lora_params, total_params = count_lora_parameters(model)
    print(f"Total LoRA parameters: {lora_params:,}")
    print(f"Total model parameters: {total_params:,}")
    print(f"LoRA percentage: {lora_params/total_params*100:.2f}%")
    print("=" * 60 + "\n")
