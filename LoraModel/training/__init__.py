"""
Training module for RVC-LoRA

Contains training scripts and utilities.
"""

from .losses import (
    feature_loss,
    discriminator_loss,
    generator_loss,
    kl_loss,
    mel_spectrogram_loss,
    GeneratorLoss,
    DiscriminatorLoss,
    LoRAFineTuneLoss,
)

from .data_loader import (
    LoRATrainingDataset,
    LoRATrainingCollate,
    SimpleAudioDataset,
    create_dataloader,
)

from .train_lora import (
    LoRATrainer,
    create_default_config,
)

__all__ = [
    # Losses
    'feature_loss',
    'discriminator_loss',
    'generator_loss',
    'kl_loss',
    'mel_spectrogram_loss',
    'GeneratorLoss',
    'DiscriminatorLoss',
    'LoRAFineTuneLoss',
    # Data loading
    'LoRATrainingDataset',
    'LoRATrainingCollate',
    'SimpleAudioDataset',
    'create_dataloader',
    # Training
    'LoRATrainer',
    'create_default_config',
]
