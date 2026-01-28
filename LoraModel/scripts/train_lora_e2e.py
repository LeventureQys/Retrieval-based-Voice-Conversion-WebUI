#!/usr/bin/env python
"""
End-to-End LoRA Training Script for RVC

This script provides a complete pipeline from raw audio to trained LoRA weights:
1. Audio preprocessing (slicing, normalization)
2. Feature extraction (HuBERT, F0, Mel)
3. LoRA training
4. Save LoRA weights

Usage:
    python train_lora_e2e.py --input_dir /path/to/audio --output_dir /path/to/output --base_model /path/to/model.pth
"""

import os
import sys
import argparse
import logging
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any

import torch

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_ROOT = os.path.dirname(SCRIPT_DIR)
RVC_ROOT = os.path.dirname(LORA_ROOT)

sys.path.insert(0, LORA_ROOT)
sys.path.insert(0, RVC_ROOT)

from lora import LoRAConfig, save_lora_checkpoint
from models import load_synthesizer_with_lora
from training.train_lora import LoRATrainer, create_default_config
from training.data_loader import create_dataloader
from preprocessing import PreprocessingPipeline

logger = logging.getLogger(__name__)


def setup_logging(output_dir: str, verbose: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(output_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return log_file


def find_base_model(model_path: str, sample_rate: int = 40000, f0: bool = True) -> str:
    """Find base model path, checking common locations."""
    if os.path.exists(model_path):
        return model_path

    # Check LoraModel/download/pretrained_v2
    pretrained_dir = os.path.join(LORA_ROOT, "download", "pretrained_v2")

    # Determine model name based on sample rate and f0
    sr_suffix = {32000: "32k", 40000: "40k", 48000: "48k"}.get(sample_rate, "40k")
    model_name = f"f0G{sr_suffix}.pth" if f0 else f"G{sr_suffix}.pth"

    candidate = os.path.join(pretrained_dir, model_name)
    if os.path.exists(candidate):
        logger.info(f"Found base model: {candidate}")
        return candidate

    # Check RVC assets
    rvc_pretrained = os.path.join(RVC_ROOT, "assets", "pretrained_v2", model_name)
    if os.path.exists(rvc_pretrained):
        logger.info(f"Found base model: {rvc_pretrained}")
        return rvc_pretrained

    raise FileNotFoundError(
        f"Base model not found: {model_path}\n"
        f"Also checked: {candidate}, {rvc_pretrained}\n"
        f"Please provide a valid model path or download pretrained models."
    )


def find_feature_models(hubert_path: str, rmvpe_path: str) -> tuple:
    """Find HuBERT and RMVPE model paths."""
    # HuBERT
    if not os.path.exists(hubert_path):
        candidates = [
            os.path.join(LORA_ROOT, "download", "hubert_base.pt"),
            os.path.join(RVC_ROOT, "assets", "hubert", "hubert_base.pt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                hubert_path = c
                break

    # RMVPE
    if not os.path.exists(rmvpe_path):
        candidates = [
            os.path.join(LORA_ROOT, "download", "rmvpe.pt"),
            os.path.join(RVC_ROOT, "assets", "rmvpe", "rmvpe.pt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                rmvpe_path = c
                break

    return hubert_path, rmvpe_path


def train_lora_e2e(
    input_dir: str,
    output_dir: str,
    base_model: str,
    # Preprocessing options
    sample_rate: int = 40000,
    version: str = "v2",
    skip_preprocessing: bool = False,
    # Training options
    epochs: int = 100,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    # LoRA options
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    # Model options
    f0: bool = True,
    # Device options
    device: str = "cuda",
    is_half: bool = True,
    # Feature model paths
    hubert_path: str = "assets/hubert/hubert_base.pt",
    rmvpe_path: str = "assets/rmvpe/rmvpe.pt",
    # Other options
    save_interval: int = 10,
    resume_from: Optional[str] = None,
) -> Dict[str, Any]:
    """Run end-to-end LoRA training.

    Args:
        input_dir: Directory containing raw audio files
        output_dir: Directory for all outputs
        base_model: Path to base RVC model
        sample_rate: Target sample rate
        version: RVC version ("v1" or "v2")
        skip_preprocessing: Skip preprocessing if data already exists
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        f0: Whether to use F0 model
        device: Device to train on
        is_half: Whether to use half precision
        hubert_path: Path to HuBERT model
        rmvpe_path: Path to RMVPE model
        save_interval: Checkpoint save interval
        resume_from: Path to checkpoint to resume from

    Returns:
        Dictionary with training results
    """
    start_time = time.time()

    # Setup directories
    preprocess_dir = os.path.join(output_dir, "preprocessed")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Find models
    base_model = find_base_model(base_model, sample_rate, f0)
    hubert_path, rmvpe_path = find_feature_models(hubert_path, rmvpe_path)

    logger.info("=" * 60)
    logger.info("RVC-LoRA End-to-End Training")
    logger.info("=" * 60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Base model: {base_model}")
    logger.info(f"Sample rate: {sample_rate}")
    logger.info(f"Version: {version}")
    logger.info(f"Device: {device}")
    logger.info(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    logger.info("=" * 60)

    # =========================================================================
    # Step 1: Preprocessing
    # =========================================================================
    training_data_dir = os.path.join(preprocess_dir, "training_data")

    if skip_preprocessing and os.path.exists(training_data_dir):
        pt_files = [f for f in os.listdir(training_data_dir) if f.endswith('.pt')]
        if pt_files:
            logger.info(f"Skipping preprocessing, found {len(pt_files)} existing samples")
        else:
            skip_preprocessing = False

    if not skip_preprocessing:
        logger.info("\n" + "=" * 60)
        logger.info("Step 1: Preprocessing audio files")
        logger.info("=" * 60)

        pipeline = PreprocessingPipeline(
            output_dir=preprocess_dir,
            sample_rate=sample_rate,
            version=version,
            device=device,
            is_half=is_half,
            hubert_path=hubert_path,
            rmvpe_path=rmvpe_path,
        )

        preprocess_result = pipeline.run(input_dir)
        num_samples = preprocess_result['num_samples']

        if num_samples == 0:
            raise RuntimeError("No samples were preprocessed!")

        logger.info(f"Preprocessing complete: {num_samples} samples")
    else:
        num_samples = len([f for f in os.listdir(training_data_dir) if f.endswith('.pt')])

    # =========================================================================
    # Step 2: Setup model and LoRA
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Loading model and injecting LoRA")
    logger.info("=" * 60)

    # Create LoRA config
    lora_config = LoRAConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=['ups', 'resblocks'],
    )

    # Load model with LoRA
    model = load_synthesizer_with_lora(
        checkpoint_path=base_model,
        lora_config=lora_config,
        device=device,
        version=version,
        f0=f0,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.synthesizer.parameters())
    lora_params = sum(p.numel() for p in model.get_lora_parameters())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"LoRA parameters: {lora_params:,} ({100*lora_params/total_params:.2f}%)")

    # =========================================================================
    # Step 3: Create data loader
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Creating data loader")
    logger.info("=" * 60)

    dataloader = create_dataloader(
        data_dir=preprocess_dir,
        batch_size=batch_size,
        use_f0=f0,
        num_workers=0 if device == "cuda" else 4,  # Avoid CUDA issues with multiprocessing
        shuffle=True,
        dataset_type="auto",
    )

    logger.info(f"Data loader created: {len(dataloader)} batches")

    # =========================================================================
    # Step 4: Training
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Training LoRA")
    logger.info("=" * 60)

    # Create training config
    config = create_default_config()
    config.update({
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'save_interval': save_interval,
        'use_f0': f0,
        'lora_r': lora_rank,
        'lora_alpha': lora_alpha,
    })

    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Create trainer
    trainer = LoRATrainer(
        model=model,
        config=config,
        output_dir=output_dir,
        device=device,
    )

    # Train
    trainer.train(dataloader, resume_from=resume_from)

    # =========================================================================
    # Step 5: Save final LoRA weights
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Saving final LoRA weights")
    logger.info("=" * 60)

    final_lora_path = os.path.join(output_dir, "lora_final.pth")
    save_lora_checkpoint(
        model=model.synthesizer,
        path=final_lora_path,
        config=lora_config,
        epoch=epochs,
        metadata={
            'base_model': os.path.basename(base_model),
            'sample_rate': sample_rate,
            'version': version,
            'f0': f0,
            'num_samples': num_samples,
            'epochs': epochs,
        }
    )

    total_time = time.time() - start_time

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Final LoRA weights: {final_lora_path}")
    logger.info(f"Checkpoints: {checkpoint_dir}")
    logger.info("=" * 60)

    return {
        'lora_path': final_lora_path,
        'checkpoint_dir': checkpoint_dir,
        'num_samples': num_samples,
        'epochs': epochs,
        'total_time': total_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description='End-to-End LoRA Training for RVC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train_lora_e2e.py -i ./my_voice -o ./output -m ./pretrained/f0G40k.pth

  # Training with custom parameters
  python train_lora_e2e.py -i ./my_voice -o ./output -m ./model.pth \\
      --epochs 200 --batch_size 8 --lora_rank 16

  # Resume training
  python train_lora_e2e.py -i ./my_voice -o ./output -m ./model.pth \\
      --resume ./output/checkpoints/lora_epoch_50.pth
        """
    )

    # Required arguments
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help='Directory containing raw audio files')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Directory for all outputs')
    parser.add_argument('--base_model', '-m', type=str, default="auto",
                        help='Path to base RVC model (default: auto-detect)')

    # Preprocessing options
    parser.add_argument('--sample_rate', '-sr', type=int, default=40000,
                        choices=[32000, 40000, 48000],
                        help='Target sample rate (default: 40000)')
    parser.add_argument('--version', '-v', type=str, default='v2',
                        choices=['v1', 'v2'],
                        help='RVC version (default: v2)')
    parser.add_argument('--skip_preprocess', action='store_true',
                        help='Skip preprocessing if data already exists')

    # Training options
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', '-b', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')

    # LoRA options
    parser.add_argument('--lora_rank', '-r', type=int, default=8,
                        help='LoRA rank (default: 8)')
    parser.add_argument('--lora_alpha', '-a', type=int, default=16,
                        help='LoRA alpha (default: 16)')
    parser.add_argument('--lora_dropout', type=float, default=0.0,
                        help='LoRA dropout (default: 0.0)')

    # Model options
    parser.add_argument('--f0', action='store_true', default=True,
                        help='Use F0 model (default: True)')
    parser.add_argument('--no_f0', action='store_false', dest='f0',
                        help='Use non-F0 model')

    # Device options
    parser.add_argument('--device', '-d', type=str, default='cuda',
                        help='Device to train on (default: cuda)')
    parser.add_argument('--no_half', action='store_true',
                        help='Disable half precision')

    # Feature model paths
    parser.add_argument('--hubert', type=str, default='assets/hubert/hubert_base.pt',
                        help='Path to HuBERT model')
    parser.add_argument('--rmvpe', type=str, default='assets/rmvpe/rmvpe.pt',
                        help='Path to RMVPE model')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Other options
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Setup logging
    log_file = setup_logging(args.output_dir, args.verbose)
    logger.info(f"Log file: {log_file}")

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
        args.no_half = True

    # Auto-detect base model if needed
    if args.base_model == "auto":
        args.base_model = ""  # Will be auto-detected in find_base_model

    try:
        result = train_lora_e2e(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            base_model=args.base_model,
            sample_rate=args.sample_rate,
            version=args.version,
            skip_preprocessing=args.skip_preprocess,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            f0=args.f0,
            device=args.device,
            is_half=not args.no_half,
            hubert_path=args.hubert,
            rmvpe_path=args.rmvpe,
            save_interval=args.save_interval,
            resume_from=args.resume,
        )

        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"LoRA weights saved to: {result['lora_path']}")
        print(f"Training samples: {result['num_samples']}")
        print(f"Total epochs: {result['epochs']}")
        print(f"Total time: {result['total_time']/60:.1f} minutes")
        print("=" * 60)

        return 0

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        print(f"\nERROR: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
