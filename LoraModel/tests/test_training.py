"""
Test script for RVC-LoRA training components

Tests loss functions, data loading, and training utilities.
"""

import os
import sys
import tempfile
import shutil

import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training.losses import (
    feature_loss,
    discriminator_loss,
    generator_loss,
    kl_loss,
    mel_spectrogram_loss,
    GeneratorLoss,
    DiscriminatorLoss,
    LoRAFineTuneLoss,
)


def test_feature_loss():
    """Test feature matching loss."""
    print("\n" + "=" * 60)
    print("Testing Feature Loss")
    print("=" * 60)

    # Create mock feature maps
    batch_size = 2
    fmap_r = [[torch.randn(batch_size, 64, 100) for _ in range(3)] for _ in range(2)]
    fmap_g = [[torch.randn(batch_size, 64, 100) for _ in range(3)] for _ in range(2)]

    loss = feature_loss(fmap_r, fmap_g)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert loss.item() >= 0

    print(f"Feature loss: {loss.item():.4f}")
    print("Feature loss test passed!")
    return True


def test_discriminator_loss():
    """Test discriminator loss."""
    print("\n" + "=" * 60)
    print("Testing Discriminator Loss")
    print("=" * 60)

    # Create mock discriminator outputs
    batch_size = 2
    disc_real = [torch.randn(batch_size, 1, 100) for _ in range(3)]
    disc_fake = [torch.randn(batch_size, 1, 100) for _ in range(3)]

    loss, r_losses, g_losses = discriminator_loss(disc_real, disc_fake)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert len(r_losses) == 3
    assert len(g_losses) == 3

    print(f"Discriminator loss: {loss.item():.4f}")
    print(f"Real losses: {r_losses}")
    print(f"Fake losses: {g_losses}")
    print("Discriminator loss test passed!")
    return True


def test_generator_loss():
    """Test generator adversarial loss."""
    print("\n" + "=" * 60)
    print("Testing Generator Loss")
    print("=" * 60)

    # Create mock discriminator outputs for generated audio
    batch_size = 2
    disc_outputs = [torch.randn(batch_size, 1, 100) for _ in range(3)]

    loss, gen_losses = generator_loss(disc_outputs)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert len(gen_losses) == 3

    print(f"Generator loss: {loss.item():.4f}")
    print("Generator loss test passed!")
    return True


def test_kl_loss():
    """Test KL divergence loss."""
    print("\n" + "=" * 60)
    print("Testing KL Loss")
    print("=" * 60)

    batch_size = 2
    channels = 192
    seq_len = 100

    z_p = torch.randn(batch_size, channels, seq_len)
    logs_q = torch.randn(batch_size, channels, seq_len)
    m_p = torch.randn(batch_size, channels, seq_len)
    logs_p = torch.randn(batch_size, channels, seq_len)
    z_mask = torch.ones(batch_size, 1, seq_len)

    loss = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0

    print(f"KL loss: {loss.item():.4f}")
    print("KL loss test passed!")
    return True


def test_mel_spectrogram_loss():
    """Test mel spectrogram L1 loss."""
    print("\n" + "=" * 60)
    print("Testing Mel Spectrogram Loss")
    print("=" * 60)

    batch_size = 2
    n_mels = 80
    seq_len = 100

    y_mel = torch.randn(batch_size, n_mels, seq_len)
    y_g_hat_mel = torch.randn(batch_size, n_mels, seq_len)

    loss = mel_spectrogram_loss(y_mel, y_g_hat_mel)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() >= 0

    # Test that identical inputs give zero loss
    loss_zero = mel_spectrogram_loss(y_mel, y_mel)
    assert loss_zero.item() < 1e-6

    print(f"Mel loss: {loss.item():.4f}")
    print(f"Mel loss (identical): {loss_zero.item():.6f}")
    print("Mel spectrogram loss test passed!")
    return True


def test_generator_loss_module():
    """Test GeneratorLoss module."""
    print("\n" + "=" * 60)
    print("Testing GeneratorLoss Module")
    print("=" * 60)

    loss_fn = GeneratorLoss(c_mel=45.0, c_kl=1.0, c_fm=2.0)

    batch_size = 2
    n_mels = 80
    seq_len = 100
    channels = 192

    # Create mock inputs
    disc_outputs = [torch.randn(batch_size, 1, 100) for _ in range(3)]
    fmap_r = [[torch.randn(batch_size, 64, 100) for _ in range(3)] for _ in range(2)]
    fmap_g = [[torch.randn(batch_size, 64, 100) for _ in range(3)] for _ in range(2)]
    y_mel = torch.randn(batch_size, n_mels, seq_len)
    y_g_hat_mel = torch.randn(batch_size, n_mels, seq_len)

    # Without KL loss
    total_loss, loss_dict = loss_fn(
        disc_outputs, fmap_r, fmap_g, y_mel, y_g_hat_mel
    )

    assert isinstance(total_loss, torch.Tensor)
    assert 'loss_gen' in loss_dict
    assert 'loss_fm' in loss_dict
    assert 'loss_mel' in loss_dict
    assert 'loss_total' in loss_dict

    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")

    # With KL loss
    z_p = torch.randn(batch_size, channels, seq_len)
    logs_q = torch.randn(batch_size, channels, seq_len)
    m_p = torch.randn(batch_size, channels, seq_len)
    logs_p = torch.randn(batch_size, channels, seq_len)
    z_mask = torch.ones(batch_size, 1, seq_len)

    total_loss_kl, loss_dict_kl = loss_fn(
        disc_outputs, fmap_r, fmap_g, y_mel, y_g_hat_mel,
        z_p, logs_q, m_p, logs_p, z_mask
    )

    assert 'loss_kl' in loss_dict_kl
    print(f"Total loss (with KL): {total_loss_kl.item():.4f}")

    print("GeneratorLoss module test passed!")
    return True


def test_discriminator_loss_module():
    """Test DiscriminatorLoss module."""
    print("\n" + "=" * 60)
    print("Testing DiscriminatorLoss Module")
    print("=" * 60)

    loss_fn = DiscriminatorLoss()

    batch_size = 2
    disc_real = [torch.randn(batch_size, 1, 100) for _ in range(3)]
    disc_fake = [torch.randn(batch_size, 1, 100) for _ in range(3)]

    loss, loss_dict = loss_fn(disc_real, disc_fake)

    assert isinstance(loss, torch.Tensor)
    assert 'loss_disc' in loss_dict
    assert 'loss_disc_real' in loss_dict
    assert 'loss_disc_fake' in loss_dict

    print(f"Discriminator loss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")
    print("DiscriminatorLoss module test passed!")
    return True


def test_lora_finetune_loss():
    """Test LoRAFineTuneLoss module."""
    print("\n" + "=" * 60)
    print("Testing LoRAFineTuneLoss Module")
    print("=" * 60)

    # Without adversarial
    loss_fn = LoRAFineTuneLoss(use_adversarial=False, c_mel=45.0)

    batch_size = 2
    n_mels = 80
    seq_len = 100

    y_mel = torch.randn(batch_size, n_mels, seq_len)
    y_g_hat_mel = torch.randn(batch_size, n_mels, seq_len)

    loss, loss_dict = loss_fn(y_mel, y_g_hat_mel)

    assert isinstance(loss, torch.Tensor)
    assert 'loss_mel' in loss_dict
    assert 'loss_total' in loss_dict

    print(f"LoRA fine-tune loss (no adv): {loss.item():.4f}")

    # With adversarial
    loss_fn_adv = LoRAFineTuneLoss(use_adversarial=True, c_mel=45.0)
    disc_outputs = [torch.randn(batch_size, 1, 100) for _ in range(3)]
    fmap_r = [[torch.randn(batch_size, 64, 100) for _ in range(3)] for _ in range(2)]
    fmap_g = [[torch.randn(batch_size, 64, 100) for _ in range(3)] for _ in range(2)]

    loss_adv, loss_dict_adv = loss_fn_adv(
        y_mel, y_g_hat_mel, disc_outputs, fmap_r, fmap_g
    )

    assert 'loss_gen' in loss_dict_adv
    assert 'loss_fm' in loss_dict_adv

    print(f"LoRA fine-tune loss (with adv): {loss_adv.item():.4f}")
    print("LoRAFineTuneLoss module test passed!")
    return True


def test_data_loader_imports():
    """Test that data loader modules can be imported."""
    print("\n" + "=" * 60)
    print("Testing Data Loader Imports")
    print("=" * 60)

    from training.data_loader import (
        LoRATrainingDataset,
        LoRATrainingCollate,
        SimpleAudioDataset,
        create_dataloader,
    )

    print("All data loader classes imported successfully!")
    print("Data loader import test passed!")
    return True


def test_simple_dataset():
    """Test SimpleAudioDataset with mock data."""
    print("\n" + "=" * 60)
    print("Testing SimpleAudioDataset")
    print("=" * 60)

    from training.data_loader import SimpleAudioDataset, LoRATrainingCollate

    # Create temporary directory with mock data
    temp_dir = tempfile.mkdtemp()

    try:
        # Create mock data files
        for i in range(3):
            base_name = f"audio_{i}"

            # Spec file
            spec = torch.randn(80, 100)
            torch.save(spec, os.path.join(temp_dir, f"{base_name}.spec.pt"))

            # Phone file
            phone = np.random.randn(100, 256).astype(np.float32)
            np.save(os.path.join(temp_dir, f"{base_name}_phone.npy"), phone)

            # Pitch file
            pitch = np.random.randint(0, 256, size=100).astype(np.int64)
            np.save(os.path.join(temp_dir, f"{base_name}_pitch.npy"), pitch)

        # Create dataset
        dataset = SimpleAudioDataset(
            data_dir=temp_dir,
            segment_size=12800,
            hop_length=400,
            use_f0=True,
        )

        assert len(dataset) == 3

        # Get a sample
        sample = dataset[0]
        assert 'spec' in sample
        assert 'phone' in sample
        assert 'pitch' in sample
        assert 'pitchf' in sample
        assert 'speaker_id' in sample

        print(f"Dataset size: {len(dataset)}")
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Spec shape: {sample['spec'].shape}")

        # Test collate function
        collate_fn = LoRATrainingCollate(use_f0=True)
        batch = collate_fn([dataset[i] for i in range(2)])

        assert 'spec' in batch
        assert batch['spec'].shape[0] == 2  # Batch size

        print(f"Batch spec shape: {batch['spec'].shape}")
        print("SimpleAudioDataset test passed!")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

    return True


def test_trainer_config():
    """Test training configuration."""
    print("\n" + "=" * 60)
    print("Testing Training Configuration")
    print("=" * 60)

    from training.train_lora import create_default_config

    config = create_default_config()

    assert 'learning_rate' in config
    assert 'batch_size' in config
    assert 'epochs' in config
    assert 'lora_r' in config
    assert 'lora_alpha' in config
    assert 'target_modules' in config

    print(f"Default config: {config}")
    print("Training configuration test passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RVC-LoRA Training Component Tests")
    print("=" * 60)

    tests = [
        ("Feature Loss", test_feature_loss),
        ("Discriminator Loss", test_discriminator_loss),
        ("Generator Loss", test_generator_loss),
        ("KL Loss", test_kl_loss),
        ("Mel Spectrogram Loss", test_mel_spectrogram_loss),
        ("GeneratorLoss Module", test_generator_loss_module),
        ("DiscriminatorLoss Module", test_discriminator_loss_module),
        ("LoRAFineTuneLoss Module", test_lora_finetune_loss),
        ("Data Loader Imports", test_data_loader_imports),
        ("SimpleAudioDataset", test_simple_dataset),
        ("Training Configuration", test_trainer_config),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\nFAILED: {name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\nAll tests passed!")
        return 0
    else:
        print(f"\n{failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
