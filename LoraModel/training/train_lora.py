"""
LoRA Training Script for RVC

This script provides the main training loop for LoRA fine-tuning of RVC models.
"""

import os
import sys
import argparse
import logging
import json
import time
import math
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lora import LoRAConfig, save_lora_checkpoint, load_lora_checkpoint
from models import SynthesizerLoRA
from training.losses import LoRAFineTuneLoss, GeneratorLoss, mel_spectrogram_loss
from training.data_loader import create_dataloader

logger = logging.getLogger(__name__)


class LoRATrainer:
    """Trainer for LoRA fine-tuning of RVC models.

    Args:
        model: SynthesizerLoRA model
        config: Training configuration dictionary
        output_dir: Directory for saving checkpoints and logs
        device: Device to train on
    """

    def __init__(
        self,
        model: SynthesizerLoRA,
        config: Dict[str, Any],
        output_dir: str,
        device: str = "cuda",
    ):
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.device = device

        # Training parameters
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.batch_size = config.get('batch_size', 4)
        self.epochs = config.get('epochs', 100)
        self.save_interval = config.get('save_interval', 10)
        self.log_interval = config.get('log_interval', 10)
        self.grad_clip = config.get('grad_clip', 1.0)
        self.use_fp16 = config.get('use_fp16', False)

        # Loss coefficients
        self.c_mel = config.get('c_mel', 45.0)

        # Setup
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss()
        self._setup_logging()

        # Mixed precision
        self.scaler = GradScaler() if self.use_fp16 else None

        # Training state
        self.global_step = 0
        self.current_epoch = 0

    def _setup_optimizer(self):
        """Setup optimizer for LoRA parameters only."""
        lora_params = self.model.get_lora_parameters()

        if len(lora_params) == 0:
            raise ValueError("No LoRA parameters found. Make sure LoRA is injected.")

        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=self.learning_rate,
            betas=(0.8, 0.99),
            eps=1e-9,
            weight_decay=0.01,
        )

        logger.info(f"Optimizer setup with {len(lora_params)} parameter groups")

    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup and cosine annealing."""
        # Warmup + Cosine Annealing scheduler
        warmup_epochs = self.config.get('warmup_epochs', 5)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / max(1, self.epochs - warmup_epochs)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lr_lambda,
        )

    def _setup_loss(self):
        """Setup loss function."""
        self.loss_fn = LoRAFineTuneLoss(
            use_adversarial=False,
            c_mel=self.c_mel,
        )

    def _setup_logging(self):
        """Setup logging and tensorboard."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Tensorboard
        log_dir = os.path.join(self.output_dir, 'logs')
        self.writer = SummaryWriter(log_dir)

        # File logging
        log_file = os.path.join(self.output_dir, 'training.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Batch of training data

        Returns:
            Dictionary of loss values
        """
        self.model.synthesizer.train()

        # Move data to device
        spec = batch['spec'].to(self.device)
        wav = batch['wav'].to(self.device)
        phone = batch['phone'].to(self.device)
        pitch = batch['pitch'].to(self.device)
        pitchf = batch['pitchf'].to(self.device)
        speaker_id = batch['speaker_id'].to(self.device)
        spec_lengths = batch['spec_lengths'].to(self.device)
        wav_lengths = batch['wav_lengths'].to(self.device)

        # Forward pass
        self.optimizer.zero_grad()

        if self.use_fp16:
            with autocast():
                output = self._forward_pass(
                    phone, spec_lengths, pitch, pitchf,
                    spec, spec_lengths, speaker_id  # y_lengths should be spec_lengths, not wav_lengths
                )
                loss, loss_dict = self._compute_loss(output, spec, wav)
        else:
            output = self._forward_pass(
                phone, spec_lengths, pitch, pitchf,
                spec, spec_lengths, speaker_id  # y_lengths should be spec_lengths, not wav_lengths
            )
            loss, loss_dict = self._compute_loss(output, spec, wav)

        # Backward pass
        if self.use_fp16:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.get_lora_parameters(),
                self.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.get_lora_parameters(),
                self.grad_clip
            )
            self.optimizer.step()

        return loss_dict

    def _forward_pass(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        spec: torch.Tensor,
        spec_lengths: torch.Tensor,
        speaker_id: torch.Tensor,
    ):
        """Execute forward pass through the model."""
        return self.model.forward(
            phone, phone_lengths, pitch, pitchf,
            spec, spec_lengths, speaker_id
        )

    def _compute_loss(
        self,
        output,
        target_spec: torch.Tensor,
        target_wav: torch.Tensor,
    ) -> tuple:
        """Compute training loss using mel spectrogram loss."""
        # Add RVC mel_processing to path
        import sys
        import os

        current_file = os.path.abspath(__file__)
        lora_model_dir = os.path.dirname(os.path.dirname(current_file))
        rvc_root = os.path.dirname(lora_model_dir)
        rvc_path = os.path.join(rvc_root, 'infer', 'lib', 'train')

        if rvc_path not in sys.path:
            sys.path.insert(0, rvc_path)

        # Extract generated audio from output
        if isinstance(output, tuple):
            generated_audio = output[0]
        else:
            generated_audio = output

        # Fix dimension mismatch: model outputs [B, 1, T], target is [B, T]
        if generated_audio.dim() == 3 and generated_audio.shape[1] == 1:
            generated_audio = generated_audio.squeeze(1)

        # Align lengths
        min_len = min(generated_audio.shape[-1], target_wav.shape[-1])
        generated_audio = generated_audio[..., :min_len]
        target_wav = target_wav[..., :min_len]

        loss_dict = {}

        try:
            from mel_processing import mel_spectrogram_torch

            # Mel Loss (RVC default config for 40kHz)
            mel_target = mel_spectrogram_torch(
                target_wav, 2048, 128, 40000, 400, 2048, 0, None
            )
            mel_generated = mel_spectrogram_torch(
                generated_audio, 2048, 128, 40000, 400, 2048, 0, None
            )
            loss_mel = torch.nn.functional.l1_loss(mel_generated, mel_target)
            loss_dict['loss_mel'] = loss_mel.item()
            loss_dict['loss_total'] = loss_mel.item()
            return loss_mel, loss_dict

        except Exception as e:
            # Fallback to waveform loss
            logger.warning(f"Mel loss failed: {e}, using waveform loss")
            loss_wav = torch.nn.functional.l1_loss(generated_audio, target_wav)
            loss_dict['loss_wav'] = loss_wav.item()
            loss_dict['loss_total'] = loss_wav.item()
            return loss_wav, loss_dict

        return total_loss, loss_dict

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Dictionary of average loss values
        """
        epoch_losses = {}
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            loss_dict = self.train_step(batch)

            # Accumulate losses
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0
                epoch_losses[key] += value
            num_batches += 1

            # Logging
            if self.global_step % self.log_interval == 0:
                self._log_step(loss_dict, batch_idx, len(dataloader))

            self.global_step += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def _log_step(self, loss_dict: Dict[str, float], batch_idx: int, total_batches: int):
        """Log training step."""
        lr = self.optimizer.param_groups[0]['lr']

        log_str = f"Epoch {self.current_epoch} [{batch_idx}/{total_batches}] "
        log_str += f"Step {self.global_step} "
        log_str += f"LR: {lr:.6f} "
        for key, value in loss_dict.items():
            log_str += f"{key}: {value:.4f} "
            self.writer.add_scalar(f'train/{key}', value, self.global_step)

        logger.info(log_str)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save LoRA weights only
        lora_path = os.path.join(checkpoint_dir, f'lora_epoch_{epoch}.pth')
        save_lora_checkpoint(
            model=self.model.synthesizer,
            path=lora_path,
            config=self.model.lora_config,
            epoch=epoch,
            optimizer_state=self.optimizer.state_dict(),
        )
        logger.info(f"Saved LoRA checkpoint: {lora_path}")

        # Save best model
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'lora_best.pth')
            save_lora_checkpoint(
                model=self.model.synthesizer,
                path=best_path,
                config=self.model.lora_config,
                epoch=epoch,
            )
            logger.info(f"Saved best model: {best_path}")

        # Save training state
        state_path = os.path.join(checkpoint_dir, 'training_state.pth')
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, state_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        # Load LoRA weights
        self.model.synthesizer, _, epoch, _ = load_lora_checkpoint(
            model=self.model.synthesizer,
            path=checkpoint_path,
        )
        self.current_epoch = epoch

        # Load training state if available
        state_path = os.path.join(
            os.path.dirname(checkpoint_path),
            'training_state.pth'
        )
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.global_step = state['global_step']
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
            logger.info(f"Loaded training state from epoch {epoch}")

    def train(self, dataloader, resume_from: Optional[str] = None):
        """Main training loop.

        Args:
            dataloader: Training data loader
            resume_from: Path to checkpoint to resume from
        """
        if resume_from is not None:
            self.load_checkpoint(resume_from)

        logger.info(f"Starting training from epoch {self.current_epoch}")
        logger.info(f"Total epochs: {self.epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Learning rate: {self.learning_rate}")

        best_loss = float('inf')

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Train epoch
            epoch_losses = self.train_epoch(dataloader)

            # Update scheduler
            self.scheduler.step()

            # Log epoch
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch} completed in {epoch_time:.2f}s - "
                f"Loss: {epoch_losses.get('loss_total', 0):.4f}"
            )

            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                is_best = epoch_losses.get('loss_total', float('inf')) < best_loss
                if is_best:
                    best_loss = epoch_losses.get('loss_total', float('inf'))
                self.save_checkpoint(epoch, is_best)

        # Save final checkpoint
        self.save_checkpoint(self.epochs - 1)
        logger.info("Training completed!")

        self.writer.close()


def create_default_config() -> Dict[str, Any]:
    """Create default training configuration."""
    return {
        # Training parameters
        'learning_rate': 1e-4,
        'batch_size': 4,
        'epochs': 100,
        'save_interval': 10,
        'log_interval': 10,
        'grad_clip': 1.0,
        'use_fp16': False,
        'lr_decay': 0.999,

        # Loss coefficients
        'c_mel': 45.0,

        # Data parameters
        'segment_size': 12800,
        'hop_length': 400,
        'use_f0': True,

        # LoRA parameters
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.0,
        'target_modules': ['ups', 'resblocks'],
    }


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description='Train LoRA for RVC')

    # Required arguments
    parser.add_argument('--base_model', type=str, required=True,
                        help='Path to base RVC model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for saving outputs')

    # Optional arguments
    parser.add_argument('--config', type=str, default=None,
                        help='Path to training config JSON file')
    parser.add_argument('--filelist', type=str, default=None,
                        help='Path to file list for training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')

    # LoRA parameters
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha')

    # Model parameters
    parser.add_argument('--version', type=str, default='v2',
                        choices=['v1', 'v2'],
                        help='RVC model version')
    parser.add_argument('--f0', action='store_true', default=True,
                        help='Use F0 model')
    parser.add_argument('--no_f0', action='store_false', dest='f0',
                        help='Use non-F0 model')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load or create config
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()

    # Override config with command line arguments
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['lora_r'] = args.lora_rank
    config['lora_alpha'] = args.lora_alpha
    config['use_f0'] = args.f0

    # Create LoRA config
    lora_config = LoRAConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config.get('lora_dropout', 0.0),
        target_modules=config.get('target_modules', ['ups', 'resblocks']),
    )

    # Load base model and create SynthesizerLoRA
    logger.info(f"Loading base model from {args.base_model}")

    try:
        from models import load_synthesizer_with_lora
        model = load_synthesizer_with_lora(
            checkpoint_path=args.base_model,
            lora_config=lora_config,
            device=args.device,
            version=args.version,
            f0=args.f0,
        )
    except ImportError as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Make sure you're running from the RVC root directory")
        return 1

    # Create data loader
    logger.info(f"Loading training data from {args.data_dir}")
    dataloader = create_dataloader(
        data_dir=args.data_dir,
        filelist_path=args.filelist,
        batch_size=config['batch_size'],
        segment_size=config['segment_size'],
        hop_length=config['hop_length'],
        use_f0=config['use_f0'],
    )

    # Create trainer
    trainer = LoRATrainer(
        model=model,
        config=config,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Save config
    config_path = os.path.join(args.output_dir, 'config.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Train
    trainer.train(dataloader, resume_from=args.resume)

    return 0


if __name__ == '__main__':
    exit(main())
