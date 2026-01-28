"""
LoRA Merge Script for RVC

This script merges LoRA weights into the base model to create a standalone model.
"""

import os
import sys
import argparse
import logging

import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lora import LoRAConfig, merge_lora_weights, load_lora_checkpoint
from inference.model_loader import LoRAModelLoader

logger = logging.getLogger(__name__)


def merge_lora_to_base(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    version: str = "v2",
    f0: bool = True,
    device: str = "cpu",
):
    """Merge LoRA weights into base model and save.

    Args:
        base_model_path: Path to base model checkpoint
        lora_path: Path to LoRA weights
        output_path: Path to save merged model
        version: Model version
        f0: Whether model supports F0
        device: Device to use for merging
    """
    logger.info(f"Loading base model from {base_model_path}")
    logger.info(f"Loading LoRA weights from {lora_path}")

    # Load model with LoRA
    loader = LoRAModelLoader(device=device, is_half=False)
    model, model_info = loader.load_with_lora(
        base_model_path=base_model_path,
        lora_path=lora_path,
        version=version,
        f0=f0,
        merge=True,  # Merge LoRA weights
    )

    # Get the synthesizer
    if hasattr(model, 'synthesizer'):
        synthesizer = model.synthesizer
    else:
        synthesizer = model

    # Load original checkpoint to preserve config
    original_checkpoint = torch.load(base_model_path, map_location=device)

    # Create new checkpoint with merged weights
    merged_checkpoint = {
        "weight": synthesizer.state_dict(),
        "config": original_checkpoint.get("config"),
        "info": f"Merged with LoRA from {os.path.basename(lora_path)}",
    }

    # Preserve other metadata
    for key in ["version", "sr", "f0"]:
        if key in original_checkpoint:
            merged_checkpoint[key] = original_checkpoint[key]

    # Save merged model
    logger.info(f"Saving merged model to {output_path}")
    torch.save(merged_checkpoint, output_path)

    # Calculate size reduction info
    base_size = os.path.getsize(base_model_path)
    lora_size = os.path.getsize(lora_path)
    merged_size = os.path.getsize(output_path)

    logger.info(f"Base model size: {base_size / 1024 / 1024:.2f} MB")
    logger.info(f"LoRA size: {lora_size / 1024:.2f} KB")
    logger.info(f"Merged model size: {merged_size / 1024 / 1024:.2f} MB")
    logger.info("Merge completed successfully!")


def extract_lora_from_finetuned(
    base_model_path: str,
    finetuned_model_path: str,
    output_path: str,
    rank: int = 8,
    alpha: int = 16,
    device: str = "cpu",
):
    """Extract LoRA weights by comparing base and fine-tuned models.

    This creates LoRA weights that approximate the difference between
    a base model and a fine-tuned model.

    Args:
        base_model_path: Path to base model
        finetuned_model_path: Path to fine-tuned model
        output_path: Path to save extracted LoRA weights
        rank: LoRA rank for extraction
        alpha: LoRA alpha
        device: Device to use
    """
    logger.info("Loading base model...")
    base_checkpoint = torch.load(base_model_path, map_location=device)
    base_weights = base_checkpoint.get("weight", base_checkpoint)

    logger.info("Loading fine-tuned model...")
    finetuned_checkpoint = torch.load(finetuned_model_path, map_location=device)
    finetuned_weights = finetuned_checkpoint.get("weight", finetuned_checkpoint)

    # Find weight differences
    lora_state_dict = {}
    target_layers = ["ups", "resblocks"]

    for name, finetuned_param in finetuned_weights.items():
        if name not in base_weights:
            continue

        # Check if this is a target layer
        is_target = any(target in name for target in target_layers)
        if not is_target:
            continue

        # Check if it's a weight (not bias)
        if "weight" not in name:
            continue

        base_param = base_weights[name]
        diff = finetuned_param - base_param

        # Skip if no difference
        if torch.allclose(diff, torch.zeros_like(diff), atol=1e-6):
            continue

        # Compute low-rank approximation using SVD
        # Reshape to 2D for SVD
        original_shape = diff.shape
        if len(original_shape) == 1:
            continue

        if len(original_shape) == 3:
            # Conv1d: [out, in, kernel] -> [out, in*kernel]
            diff_2d = diff.reshape(original_shape[0], -1)
        elif len(original_shape) == 2:
            diff_2d = diff
        else:
            continue

        try:
            U, S, Vh = torch.linalg.svd(diff_2d.float(), full_matrices=False)

            # Truncate to rank
            actual_rank = min(rank, min(diff_2d.shape))
            U_r = U[:, :actual_rank]
            S_r = S[:actual_rank]
            Vh_r = Vh[:actual_rank, :]

            # Create LoRA matrices: diff ≈ B @ A where B = U_r @ diag(sqrt(S_r)), A = diag(sqrt(S_r)) @ Vh_r
            sqrt_S = torch.sqrt(S_r)
            lora_B = U_r * sqrt_S.unsqueeze(0)  # [out, rank]
            lora_A = sqrt_S.unsqueeze(1) * Vh_r  # [rank, in*kernel]

            # Store LoRA weights
            lora_name = name.replace(".weight", "")
            lora_state_dict[f"{lora_name}.lora_A"] = lora_A
            lora_state_dict[f"{lora_name}.lora_B"] = lora_B

            logger.info(f"Extracted LoRA for {name}: rank={actual_rank}")

        except Exception as e:
            logger.warning(f"Failed to extract LoRA for {name}: {e}")
            continue

    if not lora_state_dict:
        logger.warning("No LoRA weights extracted!")
        return

    # Create LoRA config
    lora_config = LoRAConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_layers,
    )

    # Save LoRA checkpoint
    checkpoint = {
        "lora_state_dict": lora_state_dict,
        "config": lora_config.to_dict(),
        "info": f"Extracted from {os.path.basename(finetuned_model_path)}",
    }

    torch.save(checkpoint, output_path)
    logger.info(f"Saved extracted LoRA weights to {output_path}")
    logger.info(f"Total LoRA parameters: {sum(p.numel() for p in lora_state_dict.values()):,}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='LoRA Merge/Extract Tool for RVC')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge LoRA into base model')
    merge_parser.add_argument('--base', type=str, required=True,
                              help='Path to base model')
    merge_parser.add_argument('--lora', type=str, required=True,
                              help='Path to LoRA weights')
    merge_parser.add_argument('--output', type=str, required=True,
                              help='Path to save merged model')
    merge_parser.add_argument('--version', type=str, default='v2',
                              choices=['v1', 'v2'],
                              help='Model version')
    merge_parser.add_argument('--f0', action='store_true', default=True,
                              help='F0 model')
    merge_parser.add_argument('--no_f0', action='store_false', dest='f0')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract LoRA from fine-tuned model')
    extract_parser.add_argument('--base', type=str, required=True,
                                help='Path to base model')
    extract_parser.add_argument('--finetuned', type=str, required=True,
                                help='Path to fine-tuned model')
    extract_parser.add_argument('--output', type=str, required=True,
                                help='Path to save LoRA weights')
    extract_parser.add_argument('--rank', type=int, default=8,
                                help='LoRA rank')
    extract_parser.add_argument('--alpha', type=int, default=16,
                                help='LoRA alpha')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show model/LoRA info')
    info_parser.add_argument('path', type=str, help='Path to model or LoRA file')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if args.command == 'merge':
        merge_lora_to_base(
            base_model_path=args.base,
            lora_path=args.lora,
            output_path=args.output,
            version=args.version,
            f0=args.f0,
        )

    elif args.command == 'extract':
        extract_lora_from_finetuned(
            base_model_path=args.base,
            finetuned_model_path=args.finetuned,
            output_path=args.output,
            rank=args.rank,
            alpha=args.alpha,
        )

    elif args.command == 'info':
        from inference.model_loader import get_model_info
        info = get_model_info(args.path)
        print("\nModel/LoRA Information:")
        print("-" * 40)
        for key, value in info.items():
            print(f"  {key}: {value}")

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
